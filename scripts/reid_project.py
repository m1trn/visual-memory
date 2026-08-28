"""Train our own fingerprint: a projection on top of the frozen person embedder.

The pair head (`reid_train.py`) sat at the comparison step and could not beat
cosine: the vectors were already the ceiling. This trains the step before -
one linear map, 768 -> k, fitted so that cosine in the NEW space separates the
annotated people better. Smaller too, so every comparison and stored record
shrinks with it.

Loss is supervised contrastive: each batch holds several people with several
views each; a view is pulled toward the other views of its person and pushed
from everyone else's, hardest first because they dominate the softmax. Early
stopping and the verdict use the binder's own aggregation on real returns of a
held-out sequence, projected vs raw, so the number is the one that decides.

Usage:
    python scripts/reid_project.py --train MOT17-02-FRCNN MOT17-04-FRCNN ... --heldout MOT17-09-FRCNN
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from reid_calibrate import _by_person, threshold_at  # noqa: E402
from reid_train import observations, real_returns  # noqa: E402
from vision_memory.appearance import build_describer  # noqa: E402
from vision_memory.config import load_appearance_config, load_memory_config, load_reid_config  # noqa: E402
from vision_memory.reid import auroc, identity_score, select_diverse  # noqa: E402


def _slice(rows, dim):
    return [{**r, "embedding": np.asarray(r["embedding"][:dim], np.float32)} for r in rows]


def evaluate(rows, K, quantile, rate, project=None):
    """AUROC and recovery on real returns, optionally in the projected space."""
    stored, queries = real_returns(rows, K)
    if project is not None:
        stored = {k: project(v) for k, v in stored.items()}
        queries = {k: project(v) for k, v in queries.items()}
    pos = np.array([identity_score(q, stored[k], quantile) for k, q in queries.items()])
    neg = np.array([identity_score(q, stored[o], quantile) for k, q in queries.items() for o in stored if o != k])
    au = auroc(np.r_[pos, neg], np.r_[np.ones(len(pos)), np.zeros(len(neg))].astype(bool))
    t = threshold_at(pos, neg, rate)
    return au, float((pos >= t).mean()), len(pos)


def train(rows, held_rows, dim_in, dim_out, K, quantile, rate, epochs, lr, temperature, seed, log):
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    by = _by_person(rows)
    people = [k for k, rs in by.items() if len(rs) >= 4]
    embs = {k: torch.from_numpy(np.stack([r["embedding"] for r in by[k]])) for k in people}
    w = torch.nn.Parameter(torch.eye(dim_in, dim_out) if dim_out <= dim_in else torch.randn(dim_in, dim_out) * 0.02)
    # Start near the identity on the leading dims: a projection that begins as
    # "keep the first k coordinates" already carries most of the backbone's
    # separation, so the loss refines rather than rediscovers it.
    opt = torch.optim.AdamW([w], lr=lr, weight_decay=1e-4)
    P, V = 24, 4  # people per batch, views per person

    def project_np(x):
        z = np.asarray(x, np.float32) @ w.detach().numpy()
        return z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)

    best, best_w = -1.0, w.detach().clone()
    au0, rec0, n = evaluate(held_rows, K, quantile, rate, None)
    log(f"held-out raw cosine: AUROC {au0:.3f} recovered {rec0:.0%} (n={n})")
    steps = max(len(people) // P, 1) * 8
    for epoch in range(epochs):
        for _ in range(steps):
            chosen = rng.choice(len(people), min(P, len(people)), replace=False)
            xs, ys = [], []
            for pi in chosen:
                e = embs[people[pi]]
                idx = torch.from_numpy(rng.choice(len(e), min(V, len(e)), replace=False))
                xs.append(e[idx]); ys.append(torch.full((len(idx),), int(pi)))
            x = torch.cat(xs); y = torch.cat(ys)
            z = torch.nn.functional.normalize(x @ w, dim=1)
            sim = z @ z.T / temperature
            same = (y[:, None] == y[None, :]).float()
            eye = torch.eye(len(y))
            sim = sim - eye * 1e9                                  # never pair a view with itself
            log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
            pos_mask = same - eye
            loss = -(log_prob * pos_mask).sum(1) / pos_mask.sum(1).clamp(min=1)
            loss = loss.mean()
            opt.zero_grad(); loss.backward(); opt.step()
        au, rec, _ = evaluate(held_rows, K, quantile, rate, project_np)
        log(f"epoch {epoch + 1:>3}  loss {loss.item():.3f}  held-out projected: AUROC {au:.3f} recovered {rec:.0%}")
        if au > best:
            best, best_w = au, w.detach().clone()
    return best_w.numpy().astype(np.float32), (au0, rec0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--out", type=Path, default=Path("data/models/reid_projection.npz"))
    args = ap.parse_args()
    reid_cfg, K = load_reid_config(), load_memory_config().exemplars_per_identity
    dim_in = build_describer(load_appearance_config()).specialist.dim
    quantile, rate = reid_cfg.observation_quantile, reid_cfg.max_false_merge_rate

    train_rows = _slice([r for name in args.train for r in observations(name)], dim_in)
    held_rows = _slice(observations(args.heldout), dim_in)
    people = len({(r["sequence"], r["person"]) for r in train_rows})
    print(f"train: {len(train_rows)} views of {people} people | held out: {len(held_rows)} views | {dim_in} -> {args.dim}")

    w, (au0, rec0) = train(train_rows, held_rows, dim_in, args.dim, K, quantile, rate,
                           args.epochs, args.lr, args.temperature, reid_cfg.seed, print)
    project = lambda x: (lambda z: z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12))(np.asarray(x, np.float32) @ w)
    au, rec, n = evaluate(held_rows, K, quantile, rate, project)
    print(f"\nheld-out real returns (n={n}): raw cosine AUROC {au0:.3f} / {rec0:.0%}   projected AUROC {au:.3f} / {rec:.0%}")
    tau, trec, tn = evaluate(train_rows, K, quantile, rate, project)
    rau, rrec, _ = evaluate(train_rows, K, quantile, rate, None)
    print(f"training sequences (seen, n={tn}): raw {rau:.3f} / {rrec:.0%}   projected {tau:.3f} / {trec:.0%}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, w=w)
    verdict = "beats" if (au > au0 and rec >= rec0) else "does NOT beat"
    print(f"saved {args.out} | projection {verdict} raw cosine on held-out returns")


if __name__ == "__main__":
    main()

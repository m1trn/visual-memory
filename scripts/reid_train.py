"""Train the project's own re-identification verifier on labelled pairs.

SPEC section 20: the pretrained backbone stays frozen; a small model of ours
learns "same person or not" from pairs of its vectors. The earlier attempt
(`LogisticVerifier`) trained on pairs mined from the tracker's own output -
the circular ground truth every measurement since has escaped. This one trains
on human labels.

Data: labelled MOT sequences. Every observation the pipeline makes is
attributed to the annotator's person id (same protocol as reid_calibrate).
Pairs:

    positive   same person, at least `min_gap` frames apart - the returning
               person is the case that matters, so near-duplicates are excluded
    negative   different people in the same sequence, weighted toward the
               hard ones (highest cosine), which is where cosine fails

Training sequences are the ones you name; one held-out sequence is used only
for early stopping and the final report, never for pair mining. The report
compares the trained model against plain cosine on the SAME held-out returns,
with the same identity-score aggregation the binder uses, so the number is the
one that decides whether it ships.

Usage:
    python scripts/reid_train.py --train MOT17-02-FRCNN MOT17-04-FRCNN MOT17-11-FRCNN --heldout MOT17-09-FRCNN
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from reid_calibrate import _by_person, threshold_at  # noqa: E402
from vision_memory.appearance import build_describer, describe_detections  # noqa: E402
from vision_memory.config import (  # noqa: E402
    load_appearance_config, load_detector_config, load_memory_config, load_reid_config,
    load_tracker_config, load_video_config,
)
from vision_memory.detector import YoloOnnxDetector  # noqa: E402
from vision_memory.motchallenge import load_sequence  # noqa: E402
from vision_memory.reid import MlpVerifier, auroc, identity_score, identity_score_with, select_diverse  # noqa: E402
from vision_memory.tracker import ByteTracker, iou_matrix  # noqa: E402

MOT = Path("data/mot")
_ATTRIBUTION_IOU = 0.5
_MIN_VISIBILITY = 0.25


# ------------------------------------------------------------------ data
def detection_cache(name: str) -> list:
    """Per-frame (detections, embeddings) for a sequence, built once and kept."""
    path = MOT / f"detection_cache_{name.split('-')[1]}.pkl"
    if name == "MOT17-02-FRCNN" and (MOT / "detection_cache.pkl").exists():
        path = MOT / "detection_cache.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    print(f"{name}: building detection cache -> {path}", flush=True)
    seq = load_sequence(MOT / name)
    video_cfg, tracker_cfg = load_video_config(), load_tracker_config()
    det = YoloOnnxDetector(load_detector_config())
    desc = build_describer(load_appearance_config())
    frames = []
    for number, frame in seq.frames():
        if (number - 1) % video_cfg.detect_every_n_frames == 0:
            ds = det.detect(frame)
            emb = describe_detections(desc, frame, ds, tracker_cfg.min_exemplar_confidence)
        else:
            ds, emb = None, None
        frames.append((number, ds, emb))
        if number % 100 == 0:
            print(f"  {number}/{seq.length}", flush=True)
    path.write_bytes(pickle.dumps(frames))
    return frames


def observations(name: str) -> list[dict]:
    """Every exemplar the tracker recorded, attributed to the annotator's person."""
    seq = load_sequence(MOT / name)
    frames = detection_cache(name)
    tracker = ByteTracker(load_tracker_config())
    rows, seen = [], {}
    for number, dets, emb in frames:
        active = tracker.update(dets, emb)
        gt = seq.visible(number, _MIN_VISIBILITY)
        if len(gt.ids) == 0:
            continue
        fresh = [t for t in active if t.time_since_update == 0 and t.label == "person"
                 and t.exemplars and len(t.exemplars) != seen.get(t.id)]
        for t in active:
            seen[t.id] = len(t.exemplars)
        if not fresh:
            continue
        ov = iou_matrix(np.array([t.box for t in fresh], float), gt.boxes)
        for r, t in enumerate(fresh):
            j = int(ov[r].argmax())
            if ov[r, j] < _ATTRIBUTION_IOU:
                continue
            rows.append({"sequence": name, "frame": number, "person": int(gt.ids[j]),
                         "track": t.id, "embedding": np.asarray(t.exemplars[-1], np.float32)})
    return rows


def mine_pairs(rows: list[dict], min_gap: int, negatives_per_positive: int, rng: np.random.Generator):
    """Balanced labelled pairs, negatives weighted toward the hardest (highest cosine)."""
    by = _by_person(rows)
    people = list(by)
    a, b, y = [], [], []
    for key in people:
        rs = sorted(by[key], key=lambda r: r["frame"])
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                if rs[j]["frame"] - rs[i]["frame"] >= min_gap:
                    a.append(rs[i]["embedding"]); b.append(rs[j]["embedding"]); y.append(1)
    n_pos = len(y)
    # Negatives: for each positive, sample candidates from other people of the
    # same sequence and keep the hardest few, so the model spends its capacity
    # where cosine is wrong rather than on trivially different strangers.
    embs = {k: np.stack([r["embedding"] for r in by[k]]) for k in people}
    for _ in range(n_pos):
        k1, k2 = rng.choice(len(people), 2, replace=False)
        p1, p2 = people[k1], people[k2]
        if p1[0] != p2[0]:
            continue
        e1 = embs[p1][rng.integers(len(embs[p1]))]
        cands = embs[p2][rng.choice(len(embs[p2]), min(8, len(embs[p2])), replace=False)]
        cos = cands @ e1
        for idx in np.argsort(-cos)[:negatives_per_positive]:
            a.append(e1); b.append(cands[idx]); y.append(0)
    return np.stack(a), np.stack(b), np.asarray(y, np.int64)


# ------------------------------------------------------------ evaluation
def real_returns(rows: list[dict], exemplars: int):
    """Stored (first track) vs query (last track) per person, as the binder sees them."""
    by = _by_person(rows)
    stored, queries = {}, {}
    for key, rs in by.items():
        tracks: dict[int, list] = {}
        for r in rs:
            tracks.setdefault(r["track"], []).append(r)
        if len(tracks) < 2:
            continue
        order = sorted(tracks, key=lambda t: min(x["frame"] for x in tracks[t]))
        first = np.stack([x["embedding"] for x in tracks[order[0]]])
        later = np.stack([x["embedding"] for x in tracks[order[-1]]])
        if len(first) < 2 or len(later) < 2:
            continue
        stored[key] = first[select_diverse(first, exemplars)]
        queries[key] = later
    return stored, queries


def report(label: str, scorer, stored, queries, quantile: float, rate: float) -> tuple[float, float]:
    pos = np.array([scorer(q, stored[k], quantile) for k, q in queries.items()])
    neg = np.array([scorer(q, stored[o], quantile) for k, q in queries.items() for o in stored if o != k])
    au = auroc(np.r_[pos, neg], np.r_[np.ones(len(pos)), np.zeros(len(neg))].astype(bool))
    t = threshold_at(pos, neg, rate)
    rec = float((pos >= t).mean())
    print(f"  {label:>28}  AUROC {au:.3f}  returns recovered @{rate:.0%} wrong merges {rec:4.0%}  "
          f"(n={len(pos)}, boundary {t:.3f})")
    return au, rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--min-gap", type=int, default=30, help="frames between the two views of a positive")
    ap.add_argument("--negatives", type=int, default=3, help="hard negatives kept per positive")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--out", type=Path, default=Path(load_reid_config().verifier_weights))
    args = ap.parse_args()
    rng = np.random.default_rng(load_reid_config().seed)
    reid_cfg, K = load_reid_config(), load_memory_config().exemplars_per_identity

    train_rows = [r for name in args.train for r in observations(name)]
    held_rows = observations(args.heldout)
    slice_dim = build_describer(load_appearance_config()).specialist.dim
    print(f"train: {len(train_rows)} observations from {len(args.train)} sequences | "
          f"held out: {len(held_rows)} from {args.heldout} | person slice {slice_dim}-d")

    a, b, y = mine_pairs(train_rows, args.min_gap, args.negatives, rng)
    va, vb, vy = mine_pairs(held_rows, args.min_gap, args.negatives, rng)
    print(f"pairs: train {len(y)} ({int(y.sum())} same) | held-out {len(vy)} ({int(vy.sum())} same)")

    model = MlpVerifier(slice_dim, hidden=args.hidden, seed=reid_cfg.seed)
    model.fit(a, b, y, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
              validation=(va, vb, vy), log=print)
    cos_pairs = auroc((va[:, :slice_dim] * vb[:, :slice_dim]).sum(1), vy.astype(bool))
    print(f"held-out PAIR AUROC: cosine {cos_pairs:.4f}  (model above)")

    print("\nheld-out real returns, the binder's own aggregation:")
    stored, queries = real_returns(held_rows, K)
    cos_au, cos_rec = report("cosine (shipped)", identity_score, stored, queries,
                             reid_cfg.observation_quantile, reid_cfg.max_false_merge_rate)
    mlp_au, mlp_rec = report("trained mlp", lambda q, e, quant: identity_score_with(model, q, e, quant),
                             stored, queries, reid_cfg.observation_quantile, reid_cfg.max_false_merge_rate)
    print("\ntraining sequences (seen), same protocol:")
    stored_t, queries_t = real_returns(train_rows, K)
    report("cosine", identity_score, stored_t, queries_t, reid_cfg.observation_quantile, reid_cfg.max_false_merge_rate)
    report("trained mlp", lambda q, e, quant: identity_score_with(model, q, e, quant),
           stored_t, queries_t, reid_cfg.observation_quantile, reid_cfg.max_false_merge_rate)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    verdict = "beats" if (mlp_au > cos_au and mlp_rec >= cos_rec) else "does NOT beat"
    print(f"\nsaved {args.out}  |  trained model {verdict} cosine on held-out returns")


if __name__ == "__main__":
    main()

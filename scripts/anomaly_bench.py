"""Phase 3 proof: benchmark anomaly-detection methods on embedded sample images.

Protocol: pick one sample image as the 'normal' object, generate augmented
views of it (crop/flip/jitter/rotate), split into a fit set and a held-out
normal test set. All *other* sample images (plus a few augmentations each)
are anomalies. Embed everything, fit each detector on the fit set, score the
held-out normals and the anomalies, and report AUROC + per-group mean score
+ scoring latency.

Usage: python scripts/anomaly_bench.py [--dir data/samples] [--normal aero1] [--n-aug 20]
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from similarity_demo import fetch_samples  # noqa: E402
from vision_memory.anomaly import build_detector  # noqa: E402
from vision_memory.config import load_anomaly_config, load_encoder_config  # noqa: E402
from vision_memory.encoder import Encoder  # noqa: E402
from vision_memory.metrics import auroc  # noqa: E402



def augment(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """One random-resized-crop + flip + brightness/contrast + rotation view."""
    w, h = img.size
    scale = rng.uniform(0.6, 1.0)
    cw, ch = int(w * scale), int(h * scale)
    x0 = rng.integers(0, w - cw + 1)
    y0 = rng.integers(0, h - ch + 1)
    out = img.crop((x0, y0, x0 + cw, y0 + ch))
    if rng.random() < 0.5:
        out = out.transpose(Image.FLIP_LEFT_RIGHT)
    out = ImageEnhance.Brightness(out).enhance(rng.uniform(0.7, 1.3))
    out = ImageEnhance.Contrast(out).enhance(rng.uniform(0.7, 1.3))
    angle = rng.uniform(-15, 15)
    return out.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(128, 128, 128))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/samples")
    ap.add_argument("--normal", default="aero1")
    ap.add_argument("--n-aug", type=int, default=20)
    ap.add_argument("--held-out", type=int, default=5, help="normal views kept for test")
    ap.add_argument("--anom-aug", type=int, default=3, help="extra views per anomaly image")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    paths = fetch_samples(Path(args.dir))
    by_stem = {p.stem: p for p in paths}
    if args.normal not in by_stem:
        raise SystemExit(f"--normal {args.normal!r} not among samples: {sorted(by_stem)}")

    rng = np.random.default_rng(args.seed)
    normal_img = Image.open(by_stem[args.normal]).convert("RGB")
    if args.n_aug <= args.held_out:
        raise SystemExit("--n-aug must exceed --held-out so the fit set is non-empty")
    normal_views = [augment(normal_img, rng) for _ in range(args.n_aug)]
    fit_views, held_out_views = normal_views[:-args.held_out], normal_views[-args.held_out:]

    anomaly_views: list[Image.Image] = []
    for stem, p in by_stem.items():
        if stem == args.normal:
            continue
        img = Image.open(p).convert("RGB")
        anomaly_views.append(img)
        anomaly_views.extend(augment(img, rng) for _ in range(args.anom_aug))

    enc = Encoder(load_encoder_config())
    fit_emb = enc.encode_batch(fit_views)
    held_out_emb = enc.encode_batch(held_out_views)
    anomaly_emb = enc.encode_batch(anomaly_views)

    print(f"normal={args.normal!r}  fit={len(fit_emb)}  held-out normal={len(held_out_emb)}  "
          f"anomalies={len(anomaly_emb)}  dim={enc.dim}\n")

    eval_emb = np.concatenate([held_out_emb, anomaly_emb], axis=0)
    labels = np.concatenate([np.zeros(len(held_out_emb)), np.ones(len(anomaly_emb))])

    cfg = load_anomaly_config()
    print(f"{'method':<12}{'AUROC':>8}{'mean(normal)':>14}{'mean(anom)':>12}{'us/vec':>10}")
    for method in ("knn", "mahalanobis"):
        det = build_detector(replace(cfg, method=method))
        det.fit(fit_emb)

        t0 = time.perf_counter()
        scores = det.score(eval_emb)
        elapsed = time.perf_counter() - t0
        us_per_vec = 1e6 * elapsed / len(eval_emb)

        score = auroc(scores, labels)
        mean_normal = scores[labels == 0].mean()
        mean_anom = scores[labels == 1].mean()
        print(f"{method:<12}{score:>8.3f}{mean_normal:>14.3f}{mean_anom:>12.3f}{us_per_vec:>10.1f}")


if __name__ == "__main__":
    main()

"""Compare vision backbones on the tests that matter for this system.

Each candidate is scored on the same four axes: latency per crop, resident
memory, how cleanly it separates similar from unrelated images, and anomaly
AUROC. Usage: python scripts/backbone_bakeoff.py
"""

from __future__ import annotations

import argparse
import dataclasses
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_encoder import rss_mb  # noqa: E402
from similarity_demo import fetch_samples  # noqa: E402
from vision_memory.anomaly import build_detector  # noqa: E402
from vision_memory.config import load_anomaly_config, load_encoder_config  # noqa: E402
from vision_memory.encoder import Encoder  # noqa: E402
from vision_memory.metrics import auroc  # noqa: E402

# (backend, torch.hub name, input size) — input size must be a multiple of the patch size.
CANDIDATES = [
    ("dinov2", "dinov2_vits14", 224),
    ("radio", "c-radio_v3-b", 256),
]
_PAIRS = [(0, 1), (2, 3), (4, 5)]  # aero, basketball, chessboard pairs in sample order


def anomaly_auroc(enc: Encoder, images: list[Image.Image], normal_idx: int, seed: int = 0) -> float:
    """Fit on augmented views of one image, score held-out views against other images."""
    from anomaly_bench import augment  # local import: shares the augmentation protocol

    rng = np.random.default_rng(seed)
    views = [augment(images[normal_idx], rng) for _ in range(20)]
    fit, held = views[:15], views[15:]
    anomalies = [img for i, img in enumerate(images) if i != normal_idx]
    fit_emb = enc.encode_batch(fit)
    eval_emb = enc.encode_batch(held + anomalies)
    labels = np.array([0] * len(held) + [1] * len(anomalies))
    det = build_detector(load_anomaly_config())
    det.fit(fit_emb)
    return auroc(det.score(eval_emb), labels)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()

    paths = fetch_samples(Path("data/samples"))[:10]
    images = [Image.open(p).convert("RGB") for p in paths]
    _ = [p.stem for p in paths]
    base = load_encoder_config()

    print(f"{'backend':<10}{'model':<16}{'dim':>6}{'res':>5}{'ms/crop':>9}{'ms/crop(b8)':>13}"
          f"{'RSS MB':>8}{'sim pair':>9}{'sim other':>10}{'margin':>8}{'AUROC':>7}")
    for backend, name, size in CANDIDATES:
        cfg = dataclasses.replace(base, backend=backend, name=name, input_size=size)
        before = rss_mb()
        enc = Encoder(cfg)
        enc.encode(images[0])  # warm up
        rss = rss_mb() - before

        single = []
        for _ in range(args.iters):
            t = time.perf_counter()
            enc.encode(images[0])
            single.append(time.perf_counter() - t)
        t = time.perf_counter()
        enc.encode_batch(images[:8])
        batched = (time.perf_counter() - t) / 8

        embs = enc.encode_batch(images)
        sim = embs @ embs.T
        pair = float(np.mean([sim[i, j] for i, j in _PAIRS]))
        off = sim[np.triu_indices(len(images), k=1)]
        other = float(np.mean([v for k, v in enumerate(off)
                               if k not in {np.ravel_multi_index((i, j), sim.shape) for i, j in _PAIRS}]))
        other = float(np.mean([sim[i, j] for i in range(len(images)) for j in range(i + 1, len(images))
                               if (i, j) not in _PAIRS]))
        au = anomaly_auroc(enc, images, normal_idx=2)

        print(f"{backend:<10}{name:<16}{enc.dim:>6}{size:>5}{statistics.median(single) * 1000:>9.0f}"
              f"{batched * 1000:>13.0f}{rss:>8.0f}{pair:>9.3f}{other:>10.3f}{pair - other:>8.3f}{au:>7.3f}")
        del enc


if __name__ == "__main__":
    main()

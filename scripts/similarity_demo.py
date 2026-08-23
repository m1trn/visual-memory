"""Phase 1 proof: embed sample images, print cosine-similarity matrix, save heatmap.

Usage: python scripts/similarity_demo.py [--dir data/samples]
Downloads OpenCV's public sample images on first run if the dir is empty.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vision_memory.config import load_encoder_config  # noqa: E402
from vision_memory.encoder import Encoder  # noqa: E402

_BASE = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/"
# Ordered so expected-similar pairs are adjacent.
_SAMPLES = [
    "aero1.jpg", "aero3.jpg",            # two airplanes
    "basketball1.png", "basketball2.png",  # consecutive video frames
    "left01.jpg", "left02.jpg",          # chessboard, two viewpoints
    "fruits.jpg", "baboon.jpg", "lena.jpg", "box.png",  # unrelated singletons
]


def fetch_samples(d: Path) -> list[Path]:
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in _SAMPLES:
        p = d / name
        if not p.exists():
            print(f"downloading {name}")
            urllib.request.urlretrieve(_BASE + name, p)
        paths.append(p)
    return paths


def heatmap(sim: np.ndarray, labels: list[str], out: Path, cell: int = 48) -> None:
    n = len(labels)
    pad = 110
    img = Image.new("RGB", (pad + n * cell, pad + n * cell), "white")
    dr = ImageDraw.Draw(img)
    for i in range(n):
        for j in range(n):
            v = float(sim[i, j])
            t = (v + 1) / 2  # [-1,1] -> [0,1]
            color = (int(255 * (1 - t)), int(255 * (1 - t)), 255)
            x, y = pad + j * cell, pad + i * cell
            dr.rectangle([x, y, x + cell, y + cell], fill=color, outline="gray")
            dr.text((x + 8, y + cell // 3), f"{v:.2f}", fill="black" if t < 0.6 else "white")
        dr.text((2, pad + i * cell + cell // 3), labels[i], fill="black")
        dr.text((pad + i * cell + 2, pad - 60), labels[i][:11], fill="black")
    img.save(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/samples")
    args = ap.parse_args()

    paths = fetch_samples(Path(args.dir))
    labels = [p.stem for p in paths]
    enc = Encoder(load_encoder_config())
    embs = enc.encode_batch([Image.open(p) for p in paths])
    sim = embs @ embs.T  # cosine, since rows are unit-norm

    w = max(len(l) for l in labels)
    print(" " * (w + 1) + " ".join(f"{l[:6]:>6}" for l in labels))
    for l, row in zip(labels, sim):
        print(f"{l:>{w}} " + " ".join(f"{v:6.2f}" for v in row))

    print("\nexpected-similar pairs:")
    for i in range(0, 6, 2):
        print(f"  {labels[i]} vs {labels[i+1]}: {sim[i, i+1]:.3f}")
    off = sim[np.triu_indices(len(labels), k=1)]
    print(f"mean off-diagonal: {off.mean():.3f}, median: {np.median(off):.3f}")
    out = Path(args.dir).parent / "similarity.png"
    heatmap(sim, labels, out)
    print(f"heatmap -> {out}")


if __name__ == "__main__":
    main()

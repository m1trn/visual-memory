"""Phase 2 proof: index sample images, query each, show top-k neighbours.

Usage: python scripts/search_demo.py [--dir data/samples] [--k 3]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from similarity_demo import fetch_samples  # noqa: E402
from vision_memory.config import load_encoder_config, load_search_config  # noqa: E402
from vision_memory.encoder import Encoder  # noqa: E402
from vision_memory.search import VectorIndex  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/samples")
    ap.add_argument("--k", type=int, default=load_search_config().default_k)
    args = ap.parse_args()

    paths = fetch_samples(Path(args.dir))
    names = [p.stem for p in paths]
    enc = Encoder(load_encoder_config())
    embs = enc.encode_batch([Image.open(p) for p in paths])

    index = VectorIndex(enc.dim)
    index.add(embs, ids=range(len(names)))
    # A scratch path on purpose: writing to the configured index would leave a
    # stale file beside the real database, whose exemplar ids restart at 1.
    out = Path("data/search_demo.faiss")
    index.save(out)
    index = VectorIndex.load(out)
    print(f"indexed {len(index)} vectors, saved+reloaded {out}\n")

    scores, ids = index.search(embs, k=args.k + 1)  # +1: first hit is the query itself
    for qi, name in enumerate(names):
        hits = [(names[i], s) for s, i in zip(scores[qi], ids[qi]) if i >= 0 and i != qi][: args.k]
        print(f"{name:>12} -> " + ", ".join(f"{n} {s:.2f}" for n, s in hits))


if __name__ == "__main__":
    main()

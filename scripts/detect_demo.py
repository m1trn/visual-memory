"""Phase 4 proof: run the detector on images, print boxes + latency, save annotated copies.

Usage: python scripts/detect_demo.py [images...]   (default: data/samples/bus.jpg, downloaded if missing)
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
import urllib.request
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vision_memory.config import load_detector_config  # noqa: E402
from vision_memory.detector import YoloOnnxDetector  # noqa: E402

_DEFAULT = Path("data/samples/bus.jpg")
_DEFAULT_URL = "https://ultralytics.com/images/bus.jpg"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*", type=Path)
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()
    if not args.images:
        if not _DEFAULT.exists():
            urllib.request.urlretrieve(_DEFAULT_URL, _DEFAULT)
        args.images = [_DEFAULT]

    det = YoloOnnxDetector(load_detector_config())
    out_dir = Path("data/detections")
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in args.images:
        frame = cv2.imread(str(path))
        det.detect(frame)  # warm-up
        times = []
        for _ in range(args.iters):
            t = time.perf_counter()
            dets = det.detect(frame)
            times.append(time.perf_counter() - t)
        print(f"{path.name} {frame.shape[1]}x{frame.shape[0]}: {len(dets)} objects, "
              f"median {statistics.median(times)*1000:.0f} ms")
        for d in dets:
            x1, y1, x2, y2 = (int(v) for v in d.box)
            print(f"  {d.label:<12} {d.score:.2f}  [{x1},{y1},{x2},{y2}]")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{d.label} {d.score:.2f}", (x1, max(y1 - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite(str(out_dir / path.name), frame)
    print(f"annotated -> {out_dir}")


if __name__ == "__main__":
    main()

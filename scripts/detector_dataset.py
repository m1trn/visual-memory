"""Convert labelled MOT sequences into the layout a YOLO fine-tune expects.

The detector is the measured ceiling of this system: on MOT17-02 it finds 69%
of labelled people, and nobody can be re-identified who was never detected. It
ships with stock COCO weights - eighty everyday classes from ordinary
photographs - while this footage is small, distant, overhead pedestrians. That
mismatch is what fine-tuning addresses.

The labels already exist; they are just in the wrong format. MOT gives absolute
xywh in a CSV, YOLO wants one text file per image holding class and box centre
and size as fractions of the frame.

    data/detector/
        dataset.yaml
        images/train/000001.jpg      labels/train/000001.txt
        images/val/000001.jpg        labels/val/000001.txt

TRAIN AND VALIDATION MUST NOT SHARE A SEQUENCE. A model fine-tuned on one
camera and scored on the same camera reports how well it memorised that camera.
The sequence held out of every other decision in this project is held out here
too, and is never passed as `--train`.

Usage:
    python scripts/detector_dataset.py --train data/mot/MOT17-02-FRCNN \
                                       --val   data/mot/MOT17-09-FRCNN
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vision_memory.motchallenge import load_sequence  # noqa: E402

# Only fully-occluded boxes are dropped. A detector SHOULD find a person who is
# half hidden, so the visibility floor used when scoring identity is wrong here:
# teaching it to ignore partly-hidden people is teaching it the failure this
# fine-tune exists to fix.
_MIN_VISIBILITY = 0.1


def write_split(sequence_dir: Path, out: Path, split: str) -> tuple[int, int]:
    """Write one sequence as images and YOLO label files. Returns (frames, boxes)."""
    sequence = load_sequence(sequence_dir)
    img_dir = out / "images" / split
    lbl_dir = out / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    frames = boxes = 0
    for number in range(1, sequence.length + 1):
        source = sequence_dir / "img1" / f"{number:06d}.jpg"
        if not source.exists():
            continue
        truth = sequence.visible(number, _MIN_VISIBILITY)
        if len(truth.ids) == 0:
            continue  # a frame with no labels teaches nothing and dilutes the set
        stem = f"{sequence.name}_{number:06d}"
        shutil.copyfile(source, img_dir / f"{stem}.jpg")

        lines = []
        for x1, y1, x2, y2 in truth.boxes:
            cx = ((x1 + x2) / 2) / sequence.width
            cy = ((y1 + y2) / 2) / sequence.height
            bw = (x2 - x1) / sequence.width
            bh = (y2 - y1) / sequence.height
            if not (0 < bw <= 1 and 0 < bh <= 1):
                continue
            # clamp: an annotated box may run off the edge of the frame
            cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        frames += 1
        boxes += len(lines)
    return frames, boxes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path, nargs="+", required=True)
    ap.add_argument("--val", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=Path("data/detector"))
    args = ap.parse_args()

    overlap = {p.name for p in args.train} & {p.name for p in args.val}
    if overlap:
        raise SystemExit(f"train and val share {overlap}; the result would be meaningless")

    for split, dirs in (("train", args.train), ("val", args.val)):
        total_f = total_b = 0
        for d in dirs:
            f, b = write_split(d, args.out, split)
            print(f"  {split:<5} {d.name}: {f} frames, {b} boxes")
            total_f += f
            total_b += b
        print(f"  {split:<5} total: {total_f} frames, {total_b} boxes")

    (args.out / "dataset.yaml").write_text(
        f"path: {args.out.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\n\nnames:\n  0: person\n", encoding="utf-8")
    print(f"\n-> {args.out / 'dataset.yaml'}")


if __name__ == "__main__":
    main()

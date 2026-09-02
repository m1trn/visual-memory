"""MOT labels to YOLO labels: the conversion, and the split that must hold."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from detector_dataset import write_split  # noqa: E402


def _sequence(tmp_path: Path, name: str, rows: str, w: int = 200, h: int = 100) -> Path:
    seq = tmp_path / name
    (seq / "gt").mkdir(parents=True)
    (seq / "img1").mkdir()
    (seq / "seqinfo.ini").write_text(
        f"[Sequence]\nname={name}\nimDir=img1\nframeRate=30\n"
        f"seqLength=2\nimWidth={w}\nimHeight={h}\nimExt=.jpg\n", encoding="utf-8")
    (seq / "gt" / "gt.txt").write_text(rows, encoding="utf-8")
    import numpy as np, cv2
    for n in (1, 2):
        cv2.imwrite(str(seq / "img1" / f"{n:06d}.jpg"), np.zeros((h, w, 3), np.uint8))
    return seq


def test_boxes_become_fractions_of_the_frame(tmp_path) -> None:
    # frame 1: one person at x 20..60, y 10..50 in a 200x100 frame
    seq = _sequence(tmp_path, "MOT-A", "1,1,20,10,40,40,1,1,1.0\n")
    out = tmp_path / "ds"
    frames, boxes = write_split(seq, out, "train")
    assert (frames, boxes) == (1, 1)

    line = (out / "labels" / "train" / "MOT-A_000001.txt").read_text().strip()
    cls, cx, cy, bw, bh = line.split()
    assert cls == "0"
    assert float(cx) == pytest.approx(40 / 200)     # centre x
    assert float(cy) == pytest.approx(30 / 100)     # centre y
    assert float(bw) == pytest.approx(40 / 200)
    assert float(bh) == pytest.approx(40 / 100)
    assert (out / "images" / "train" / "MOT-A_000001.jpg").exists()


def test_frames_with_no_labels_are_left_out(tmp_path) -> None:
    """An unlabelled frame teaches nothing and dilutes the set."""
    seq = _sequence(tmp_path, "MOT-B", "2,1,20,10,40,40,1,1,1.0\n")
    frames, _ = write_split(seq, tmp_path / "ds", "train")
    assert frames == 1
    assert not (tmp_path / "ds" / "images" / "train" / "MOT-B_000001.jpg").exists()


def test_a_fully_occluded_box_is_dropped_but_a_half_hidden_one_is_kept(tmp_path) -> None:
    """A detector SHOULD find a half-hidden person, so the visibility floor
    here is far lower than the one used when scoring identity."""
    seq = _sequence(tmp_path, "MOT-C",
                    "1,1,20,10,40,40,1,1,0.5\n1,2,80,10,40,40,1,1,0.0\n")
    _, boxes = write_split(seq, tmp_path / "ds", "train")
    assert boxes == 1

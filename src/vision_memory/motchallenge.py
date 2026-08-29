"""Reading MOTChallenge sequences: frames paired with hand-labelled identities.

Every accuracy claim this project has made so far was scored against ground
truth derived from its own tracker, and that circularity has produced wrong
answers repeatedly — two tracks scoring 0.896 turned out to be a woman in a
light blue jacket and a man in a dark coat. A MOTChallenge sequence carries a
human's answer key instead: each frame lists the boxes that are really there
and, decisively, the identity each one belongs to, held consistent across
occlusions and re-entries. That last column is the only thing that can say
whether re-identification actually re-identifies.

Layout of a sequence directory, unchanged from the official archives::

    MOT17-04-FRCNN/
        seqinfo.ini      frame rate, resolution, frame count
        img1/000001.jpg  frames, numbered from 1
        gt/gt.txt        frame, id, x, y, w, h, keep, class, visibility

The ground-truth file is also what a detector's public detections look like
(``det/det.txt``), minus the identity column, so the same reader serves both.
"""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

# MOTChallenge marks every annotated box with a class and a keep flag. Only
# class 1 is an upright pedestrian: the rest are people on vehicles, static
# figures, reflections and distractors, which the benchmark's own evaluation
# excludes rather than counts as misses.
_PEDESTRIAN = 1


@dataclass(frozen=True)
class GroundTruth:
    """Hand-labelled boxes for one frame.

    ``ids`` are stable across the whole sequence: the same person carries the
    same number before and after an occlusion, which is exactly the fact no
    self-derived metric can supply.
    """

    boxes: np.ndarray  # (N, 4) xyxy
    ids: np.ndarray  # (N,) int64
    visibility: np.ndarray  # (N,) float in [0, 1]


@dataclass(frozen=True)
class Sequence:
    """One MOTChallenge sequence on disk."""

    path: Path
    name: str
    fps: float
    width: int
    height: int
    length: int
    truth: dict[int, GroundTruth]

    def frames(self) -> Iterator[tuple[int, np.ndarray]]:
        """Yield ``(frame_number, image)`` in order, numbered from 1 as on disk."""
        for number in range(1, self.length + 1):
            path = self.path / "img1" / f"{number:06d}.jpg"
            image = cv2.imread(str(path))
            if image is None:
                raise FileNotFoundError(f"missing frame {path}")
            yield number, image

    def visible(self, frame: int, min_visibility: float) -> GroundTruth:
        """Ground truth for one frame, dropping boxes mostly hidden behind others.

        A box labelled 10% visible is a sliver of a person behind someone else.
        Requiring a tracker to find it measures the detector's appetite for
        guessing rather than the system's ability to keep an identity, so the
        benchmark's own scripts apply a visibility floor here too.
        """
        got = self.truth.get(frame)
        if got is None:
            empty = np.empty((0, 4), dtype=np.float64)
            return GroundTruth(empty, np.empty(0, np.int64), np.empty(0, np.float64))
        keep = got.visibility >= min_visibility
        return GroundTruth(got.boxes[keep], got.ids[keep], got.visibility[keep])


def metrics_module():
    """Import ``motmetrics``, restoring the one NumPy alias it still relies on.

    motmetrics 1.4 calls ``np.asfarray``, which NumPy removed in 2.0. Pinning
    NumPy back would drag the whole project to an older release to satisfy a
    test-only dependency, so the alias is restored instead — it is exactly
    ``asarray`` with a float dtype, and the shim says only that.
    """
    if not hasattr(np, "asfarray"):
        np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)  # type: ignore[attr-defined]
    import motmetrics

    return motmetrics


def load_sequence(path: str | Path, keep_classes: "tuple[int, ...] | None" = None) -> Sequence:
    """Read a sequence directory, with its ground truth if one is present.

    ``keep_classes`` selects which annotated classes count, defaulting to
    MOT17's upright pedestrian. A VisDrone sequence written by
    ``scripts/fetch_visdrone.py`` uses the same file layout with vehicle
    classes, so passing e.g. ``(4, 5, 6, 9)`` scores cars, vans, trucks and
    buses instead.
    """
    path = Path(path)
    info = configparser.ConfigParser()
    read = info.read(path / "seqinfo.ini")
    if not read:
        raise FileNotFoundError(f"{path} has no seqinfo.ini; is it a MOT sequence?")
    section = info["Sequence"]
    return Sequence(
        path=path,
        name=section.get("name", path.name),
        fps=float(section.get("frameRate", 30)),
        width=int(section.get("imWidth", 0)),
        height=int(section.get("imHeight", 0)),
        length=int(section["seqLength"]),
        truth=_read_labels(path / "gt" / "gt.txt", keep_classes),
    )


def find_sequences(root: str | Path) -> list[Path]:
    """Every sequence directory under ``root``, in name order.

    Accepts either the archive's own layout (``MOT17/train/MOT17-04-FRCNN``) or
    a single sequence directory, so a partial download is still usable.
    """
    root = Path(root)
    if (root / "seqinfo.ini").exists():
        return [root]
    return sorted(p.parent for p in root.rglob("seqinfo.ini"))


def _read_labels(path: Path, keep_classes: "tuple[int, ...] | None" = None) -> dict[int, GroundTruth]:
    """Parse ``gt.txt`` into per-frame boxes, identities and visibility."""
    if not path.exists():
        return {}
    raw = np.loadtxt(path, delimiter=",", ndmin=2)
    if raw.size == 0:
        return {}
    # frame, id, x, y, w, h, keep, class, visibility
    keep = raw[:, 6] > 0
    wanted = (_PEDESTRIAN,) if keep_classes is None else tuple(keep_classes)
    if raw.shape[1] > 7:
        keep &= np.isin(raw[:, 7].astype(np.int64), wanted)
    raw = raw[keep]
    visibility = raw[:, 8] if raw.shape[1] > 8 else np.ones(len(raw))

    boxes = np.stack(
        [raw[:, 2], raw[:, 3], raw[:, 2] + raw[:, 4], raw[:, 3] + raw[:, 5]], axis=1
    )
    out: dict[int, GroundTruth] = {}
    frames = raw[:, 0].astype(np.int64)
    for frame in np.unique(frames):
        rows = frames == frame
        out[int(frame)] = GroundTruth(
            boxes=boxes[rows],
            ids=raw[rows, 1].astype(np.int64),
            visibility=np.asarray(visibility)[rows].astype(np.float64),
        )
    return out

"""How an object's appearance is described: learned features plus clothing colour.

The frozen encoder is good at shape and texture but, on 30-pixel-wide pedestrian
crops, two strangers in dark coats look much alike. Colour is the complementary
evidence — and specifically colour *by body band*, because "dark jacket, blue
jeans" and "dark jacket, dark trousers" are different people even when the
learned features cannot separate them.

Measured on this footage, matching the same person across a three-second gap
against provably-different people:

    DINOv2 on the torso crop        AUROC 0.912, 62.1% recall at 5% false accepts
    colour bands alone              AUROC 0.900, 63.2%
    both, fused                     AUROC 0.930, 68.8%

The two halves are scaled by ``sqrt(1 - w)`` and ``sqrt(w)`` before being
concatenated, which makes the fused vector's cosine exactly the weighted sum of
the two cosines. Everything downstream — the index, memory, re-identification,
the tracker's appearance veto — keeps working on plain dot products and needs to
know nothing about this.
"""

from __future__ import annotations

import cv2
import numpy as np

from vision_memory.config import AppearanceConfig
from vision_memory.encoder import Encoder

_HSV_BINS = (8, 8, 4)  # hue is what identifies clothing; value is mostly lighting


def colour_bands(crop_bgr: np.ndarray, bands: int) -> np.ndarray:
    """Unit-norm HSV histogram per horizontal band of a crop, concatenated."""
    if crop_bgr.size == 0:
        raise ValueError("colour_bands needs a non-empty crop")
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    height = hsv.shape[0]
    edges = [round(height * k / bands) for k in range(bands + 1)]
    parts = []
    for lo, hi in zip(edges, edges[1:]):
        band = hsv[lo:hi] if hi > lo else hsv
        hist = cv2.calcHist([band], [0, 1, 2], None, list(_HSV_BINS), [0, 180, 0, 256, 0, 256])
        parts.append(hist.flatten())
    return _unit(np.concatenate(parts).astype(np.float32))


def fuse(deep: np.ndarray, colour: np.ndarray, weight: float) -> np.ndarray:
    """Combine unit-norm halves so the result's cosine is their weighted sum."""
    return np.concatenate([
        np.sqrt(1.0 - weight) * np.asarray(deep, dtype=np.float32),
        np.sqrt(weight) * np.asarray(colour, dtype=np.float32),
    ]).astype(np.float32)


class AppearanceDescriber:
    """Turns a frame and a box into the vector the rest of the system stores.

    The learned half sees only the top of the box, where a person's identity
    lives; the colour half sees the whole box, because trousers are useless for
    shape and informative for colour.
    """

    def __init__(self, encoder: Encoder, cfg: AppearanceConfig) -> None:
        self.encoder = encoder
        self.cfg = cfg

    @property
    def dim(self) -> int:
        """Width of the fused vector."""
        if self.cfg.colour_weight <= 0.0:
            return self.encoder.dim
        return self.encoder.dim + int(np.prod(_HSV_BINS)) * self.cfg.colour_bands

    def describe(self, frame_bgr: np.ndarray, boxes: list[np.ndarray]) -> dict[int, np.ndarray]:
        """Describe each box, skipping any too small to carry a usable signal.

        Returns a mapping from position in ``boxes`` to its vector, so callers
        can tell which boxes were skipped rather than silently mis-aligning.
        """
        crops_deep, crops_colour, kept = [], [], []
        height, width = frame_bgr.shape[:2]
        for index, box in enumerate(boxes):
            x1, y1, x2, y2 = (float(v) for v in box)
            upper = int(round(y1 + (y2 - y1) * self.cfg.deep_upper_fraction))
            x1i, y1i = max(int(round(x1)), 0), max(int(round(y1)), 0)
            x2i, y2i = min(int(round(x2)), width), min(int(round(y2)), height)
            upper = min(max(upper, y1i), y2i)
            if x2i - x1i < self.cfg.min_crop_px or upper - y1i < self.cfg.min_crop_px:
                continue
            crops_deep.append(cv2.cvtColor(frame_bgr[y1i:upper, x1i:x2i], cv2.COLOR_BGR2RGB))
            crops_colour.append(frame_bgr[y1i:y2i, x1i:x2i])
            kept.append(index)
        if not kept:
            return {}

        vectors = self.encoder.encode_batch(crops_deep)
        if self.cfg.colour_weight <= 0.0:
            return dict(zip(kept, vectors))
        return {
            index: fuse(deep, colour_bands(colour, self.cfg.colour_bands), self.cfg.colour_weight)
            for index, deep, colour in zip(kept, vectors, crops_colour)
        }


def _unit(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    return v if norm == 0.0 else (v / norm).astype(np.float32)

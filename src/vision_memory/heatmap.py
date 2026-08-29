"""Where on an object is unusual, not just how unusual it is.

The anomaly detectors score one vector per object: "this backpack is 91%
strange" and nothing more. A ViT already describes every 14x14 patch of the
crop separately - the global descriptor is only their summary - so the same
question can be asked per patch: how far is THIS patch from the patches of
everything normal? The answer is an image, and the torn strap lights up.

Deliberately built on the general encoder's patch tokens, not the person
specialist: OSNet and YouTu emit a single pooled vector with no spatial grid
to score. This is the anomaly branch, which routes objects to DINOv2 anyway.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

_MIN_NORMAL_PATCHES = 64  # below this, "normal" is a guess, not a distribution


@dataclass(frozen=True)
class PatchAnomaly:
    """Per-patch distances and the same thing as a picture."""

    grid: np.ndarray  # (g, g) float, higher = stranger
    score: float  # the crop's own score: the strangest region, not the average

    def as_image(self, width: int, height: int) -> np.ndarray:
        """Smooth colour overlay at crop size, blue (ordinary) to red (strange)."""
        norm = self.grid - self.grid.min()
        norm = norm / max(norm.max(), 1e-6)
        big = cv2.resize(norm.astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC)
        return cv2.applyColorMap((np.clip(big, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)


class PatchAnomalyDetector:
    """Nearest-neighbour anomaly scoring over patch descriptors.

    Fitted on the patches of objects known to be normal; scoring a crop asks,
    for each of its patches, the cosine distance to the closest normal patch
    ANYWHERE in that bank. A patch that resembles some part of some normal
    object is ordinary even if it sits in an odd place; only genuinely unseen
    texture scores high. That is the PatchCore idea, and it is why a torn strap
    lights up while an unusual pose does not.

    The bank is subsampled to a budget: patches are enormously redundant, and
    the nearest-neighbour distance barely moves once a few thousand are kept.
    """

    def __init__(self, budget: int = 4096, seed: int = 0) -> None:
        self.budget = int(budget)
        self._rng = np.random.default_rng(seed)
        self._bank: np.ndarray | None = None

    @property
    def fitted(self) -> bool:
        return self._bank is not None

    def fit(self, patches: np.ndarray) -> None:
        """Learn what normal looks like from ``(N, g, g, dim)`` or ``(P, dim)`` patches."""
        flat = np.asarray(patches, np.float32).reshape(-1, np.shape(patches)[-1])
        if len(flat) < _MIN_NORMAL_PATCHES:
            raise ValueError(f"need at least {_MIN_NORMAL_PATCHES} normal patches, got {len(flat)}")
        if len(flat) > self.budget:
            flat = flat[self._rng.choice(len(flat), self.budget, replace=False)]
        self._bank = np.ascontiguousarray(flat)

    def score(self, patches: np.ndarray) -> PatchAnomaly:
        """Distance of every patch of one crop to the nearest normal patch."""
        if self._bank is None:
            raise RuntimeError("fit() first")
        p = np.asarray(patches, np.float32)
        grid = p.shape[0]
        flat = p.reshape(-1, p.shape[-1])
        # unit vectors, so cosine distance is 1 - dot
        nearest = (flat @ self._bank.T).max(axis=1)
        d = (1.0 - nearest).reshape(grid, grid)
        # The crop's score is its strangest region, softened over a 2x2
        # neighbourhood: one odd patch is noise, a patch of odd patches is a
        # defect.
        pooled = cv2.blur(d.astype(np.float32), (2, 2))
        return PatchAnomaly(grid=d, score=float(pooled.max()))


def overlay(crop_bgr: np.ndarray, anomaly: PatchAnomaly, strength: float = 0.45) -> np.ndarray:
    """The crop with its heatmap blended over it."""
    h, w = crop_bgr.shape[:2]
    return cv2.addWeighted(anomaly.as_image(w, h), strength, crop_bgr, 1 - strength, 0)

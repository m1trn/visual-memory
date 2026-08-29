"""Patch-level anomaly scoring: the odd region must be the one that lights up."""

from __future__ import annotations

import numpy as np
import pytest

from vision_memory.heatmap import PatchAnomalyDetector, overlay

DIM = 8


def _unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _normal_patches(rng, n=40, grid=8):
    """Crops whose patches all sit near one direction."""
    base = np.zeros(DIM, np.float32); base[0] = 1.0
    p = base + rng.normal(scale=0.05, size=(n, grid, grid, DIM)).astype(np.float32)
    return _unit(p)


def test_the_odd_corner_scores_highest() -> None:
    rng = np.random.default_rng(0)
    det = PatchAnomalyDetector(budget=512, seed=0)
    det.fit(_normal_patches(rng))

    crop = _normal_patches(rng, n=1)[0].copy()
    odd = np.zeros(DIM, np.float32); odd[4] = 1.0        # a texture never seen
    crop[1:3, 1:3] = odd
    result = det.score(_unit(crop))

    assert result.grid.shape == (8, 8)
    hottest = np.unravel_index(result.grid.argmax(), result.grid.shape)
    assert 1 <= hottest[0] <= 2 and 1 <= hottest[1] <= 2, hottest
    assert result.grid[1, 1] > result.grid[6, 6] * 5


def test_an_ordinary_crop_scores_low() -> None:
    rng = np.random.default_rng(1)
    det = PatchAnomalyDetector(budget=512, seed=0)
    det.fit(_normal_patches(rng))
    plain = det.score(_normal_patches(rng, n=1)[0])

    crop = _normal_patches(rng, n=1)[0].copy()
    odd = np.zeros(DIM, np.float32); odd[4] = 1.0
    crop[3:5, 3:5] = odd
    assert det.score(_unit(crop)).score > plain.score * 3


def test_image_and_overlay_come_back_at_crop_size() -> None:
    rng = np.random.default_rng(2)
    det = PatchAnomalyDetector(budget=256, seed=0)
    det.fit(_normal_patches(rng))
    result = det.score(_normal_patches(rng, n=1)[0])
    assert result.as_image(40, 90).shape == (90, 40, 3)
    crop = np.zeros((90, 40, 3), np.uint8)
    assert overlay(crop, result).shape == crop.shape


def test_refuses_to_guess_from_too_few_normals() -> None:
    det = PatchAnomalyDetector()
    with pytest.raises(ValueError):
        det.fit(np.zeros((2, 2, 2, DIM), np.float32))
    with pytest.raises(RuntimeError):
        PatchAnomalyDetector().score(np.zeros((4, 4, DIM), np.float32))

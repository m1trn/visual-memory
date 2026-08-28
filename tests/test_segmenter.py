"""Mask decoding and cleanup for the display outlines."""

from __future__ import annotations

import numpy as np

from vision_memory.segmenter import _largest_piece, outline


def test_largest_piece_drops_a_detached_stray() -> None:
    mask = np.zeros((40, 40), bool)
    mask[5:25, 5:25] = True        # the body
    mask[35:38, 35:38] = True      # a stray blob
    kept = _largest_piece(mask)
    assert kept[10, 10] and not kept[36, 36]
    assert kept.sum() == 400


def test_largest_piece_leaves_a_single_blob_alone() -> None:
    mask = np.zeros((20, 20), bool)
    mask[4:9, 4:9] = True
    assert _largest_piece(mask).sum() == mask.sum()


def test_outline_traces_the_shape() -> None:
    mask = np.zeros((30, 30), bool)
    mask[10:20, 10:20] = True
    contours = outline(mask)
    assert len(contours) == 1
    pts = contours[0].reshape(-1, 2)
    assert pts[:, 0].min() == 10 and pts[:, 0].max() == 19

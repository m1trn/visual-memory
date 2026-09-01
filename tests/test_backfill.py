"""Thumbnails kept per identity, so an object can be described after it leaves."""

from __future__ import annotations

import numpy as np

from vision_memory.engine import TrackView


class _Engine:
    """The thumbnail keeper, exercised without loading four models."""

    from vision_memory.engine import VisionEngine

    _keep_thumbnails = VisionEngine._keep_thumbnails

    def __init__(self) -> None:
        self._thumbnails: dict[int, tuple[int, np.ndarray]] = {}


def _view(identity_id, box, hidden=False):
    return TrackView(track_id=1, box=np.array(box, float), label="person",
                     identity_id=identity_id, is_new=False, score=0.9,
                     hidden=hidden, anomaly=None)


def test_keeps_the_largest_view_of_each_identity() -> None:
    e = _Engine()
    frame = np.zeros((400, 400, 3), np.uint8)
    e._keep_thumbnails(frame, [_view(5, (0, 0, 40, 90))])
    small = e._thumbnails[5][0]
    e._keep_thumbnails(frame, [_view(5, (0, 0, 120, 300))])
    assert e._thumbnails[5][0] > small
    e._keep_thumbnails(frame, [_view(5, (0, 0, 30, 70))])
    assert e._thumbnails[5][0] > small, "a later smaller view must not replace it"
    assert len(e._thumbnails) == 1


def test_skips_what_cannot_be_described() -> None:
    e = _Engine()
    frame = np.zeros((400, 400, 3), np.uint8)
    e._keep_thumbnails(frame, [
        _view(None, (0, 0, 100, 200)),          # unbound
        _view(6, (0, 0, 100, 200), hidden=True),  # a predicted position
        _view(7, (0, 0, 10, 20)),                # too small to read
    ])
    assert e._thumbnails == {}


def test_thumbnails_are_bounded_and_out_of_frame_boxes_clipped() -> None:
    e = _Engine()
    frame = np.zeros((400, 400, 3), np.uint8)
    e._keep_thumbnails(frame, [_view(9, (-50, -50, 900, 900))])
    kept = e._thumbnails[9][1]
    assert max(kept.shape[:2]) <= 224
    assert kept.size > 0

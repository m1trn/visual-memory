from __future__ import annotations

import dataclasses

import numpy as np

from vision_memory.config import TrackerConfig
from vision_memory.detector import Detection
from vision_memory.tracker import ByteTracker, iou_matrix


def _cfg(**overrides) -> TrackerConfig:
    base = dict(
        max_age=30,
        min_hits=3,
        iou_threshold=0.3,
        high_conf=0.5,
        low_conf=0.1,
        appearance_weight=0.0,
    )
    base.update(overrides)
    return TrackerConfig(**base)


def _det(x1: float, y1: float, score: float = 0.9, class_id: int = 0, label: str = "obj") -> Detection:
    return Detection(box=(x1, y1, x1 + 20.0, y1 + 20.0), score=score, class_id=class_id, label=label)


def test_moving_box_keeps_one_id_and_becomes_active() -> None:
    tracker = ByteTracker(_cfg(min_hits=3))
    seen_ids = set()
    active: list = []
    for i in range(10):
        active = tracker.update([_det(10.0 + 5 * i, 10.0)])
        seen_ids |= {t.id for t in tracker._tracks}
    assert seen_ids == {1}
    assert len(active) == 1
    assert active[0].id == 1
    assert active[0].hits == 10


def test_detections_none_predicts_and_moves_box() -> None:
    tracker = ByteTracker(_cfg(min_hits=1))
    for i in range(3):
        tracker.update([_det(10.0 + 5 * i, 10.0)])
    before = tracker._tracks[0].box.copy()
    active = tracker.update(None)
    after = active[0].box
    # constant-velocity prediction should continue moving in +x direction
    assert after[0] > before[0]


def test_low_conf_pass_keeps_id_through_score_drop() -> None:
    tracker = ByteTracker(_cfg(min_hits=1))
    tracker.update([_det(10.0, 10.0, score=0.9)])
    tracker.update([_det(11.0, 10.0, score=0.9)])
    for i in range(3):
        active = tracker.update([_det(12.0 + i, 10.0, score=0.3)])
    assert len(active) == 1
    assert active[0].id == 1


def test_unmatched_track_becomes_lost_after_max_age() -> None:
    tracker = ByteTracker(_cfg(min_hits=1, max_age=3))
    tracker.update([_det(10.0, 10.0)])
    # move the real object far away so nothing matches the old track's box
    for _ in range(5):
        active = tracker.update([_det(500.0, 500.0)])
    lost = tracker.pop_lost()
    assert [t.id for t in lost] == [1]
    assert 1 not in [t.id for t in active]
    # a second pop_lost should be empty (returned exactly once)
    assert tracker.pop_lost() == []


def test_iou_matrix_known_values() -> None:
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[0.0, 0.0, 10.0, 10.0], [10.0, 10.0, 20.0, 20.0], [5.0, 0.0, 15.0, 10.0]])
    m = iou_matrix(a, b)
    assert m.shape == (1, 3)
    assert np.isclose(m[0, 0], 1.0)
    assert np.isclose(m[0, 1], 0.0)
    assert np.isclose(m[0, 2], 50.0 / 150.0)


def test_appearance_weight_breaks_iou_tie_with_embedding() -> None:
    tracker = ByteTracker(_cfg(min_hits=1, appearance_weight=1.0))
    emb_a = np.array([1.0, 0.0], dtype=np.float32)
    emb_b = np.array([0.0, 1.0], dtype=np.float32)
    # Spawn two well-separated tracks (real IoU matching) with distinct embeddings.
    tracker.update([_det(0.0, 0.0), _det(500.0, 500.0)], embeddings={0: emb_a, 1: emb_b})
    id_a = tracker._tracks[0].id
    id_b = tracker._tracks[1].id

    # Now both tracks sit far from any new detection (IoU 0 to everything),
    # so only appearance can disambiguate which detection belongs to which id.
    tracker.cfg = dataclasses.replace(tracker.cfg, iou_threshold=0.0)
    det_for_a = Detection(box=(1000.0, 1000.0, 1020.0, 1020.0), score=0.9, class_id=0, label="obj")
    det_for_b = Detection(box=(2000.0, 2000.0, 2020.0, 2020.0), score=0.9, class_id=0, label="obj")
    active = tracker.update([det_for_a, det_for_b], embeddings={0: emb_a, 1: emb_b})

    # Kalman-corrected boxes land between the prior prediction and the
    # measurement, so compare by which target each track's box moved toward
    # rather than requiring an exact match.
    by_id = {t.id: t.box[:2] for t in active}
    target_a = np.array(det_for_a.box[:2], dtype=np.float32)
    target_b = np.array(det_for_b.box[:2], dtype=np.float32)
    dist_a_to_a = np.linalg.norm(by_id[id_a] - target_a)
    dist_a_to_b = np.linalg.norm(by_id[id_a] - target_b)
    dist_b_to_a = np.linalg.norm(by_id[id_b] - target_a)
    dist_b_to_b = np.linalg.norm(by_id[id_b] - target_b)
    assert dist_a_to_a < dist_a_to_b
    assert dist_b_to_b < dist_b_to_a

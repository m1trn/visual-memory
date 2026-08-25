from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from vision_memory.config import TrackerConfig
from vision_memory.detector import Detection
from vision_memory.tracker import ByteTracker, iou_matrix, should_embed


def _cfg(**overrides) -> TrackerConfig:
    base = dict(
        max_age=30,
        min_hits=3,
        iou_threshold=0.3,
        high_conf=0.5,
        low_conf=0.1,
        contested_iou=0.5,
        max_centre_distance=2.0,
        veto_views=3,
        appearance_veto=0.0,
        appearance_weight=0.0,
        embed_every_n=2,
        max_exemplars=16,
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


def _emb(rng: np.random.Generator, dim: int = 8) -> np.ndarray:
    v = rng.normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


def test_should_embed_follows_embed_every_n() -> None:
    assert should_embed(1, 2)  # first association always embeds
    assert should_embed(2, 2)
    assert not should_embed(3, 2)
    assert should_embed(4, 2)
    assert not should_embed(5, 0)  # 0 disables embedding
    assert not should_embed(0, 2)


def test_exemplars_accumulate_for_embedded_matches() -> None:
    tracker = ByteTracker(_cfg(min_hits=1))
    rng = np.random.default_rng(0)
    for i in range(4):
        tracker.update([_det(10.0 + 5 * i, 10.0)], embeddings={0: _emb(rng)})
    track = tracker._tracks[0]
    assert len(track.exemplars) == 4
    assert all(np.isclose(np.linalg.norm(e), 1.0) for e in track.exemplars)
    assert track.embedding is not None


def test_track_without_embeddings_has_empty_exemplars() -> None:
    tracker = ByteTracker(_cfg(min_hits=1))
    for i in range(4):
        tracker.update([_det(10.0 + 5 * i, 10.0)])
    track = tracker._tracks[0]
    assert track.exemplars == []
    assert track.embedding is None


def test_unmatched_frames_add_no_exemplars() -> None:
    tracker = ByteTracker(_cfg(min_hits=1))
    rng = np.random.default_rng(1)
    tracker.update([_det(10.0, 10.0)], embeddings={0: _emb(rng)})
    for _ in range(3):
        tracker.update(None)
    assert len(tracker._tracks[0].exemplars) == 1
    assert tracker.due_for_embedding() == []


def test_exemplar_buffer_respects_cap() -> None:
    tracker = ByteTracker(_cfg(min_hits=1, max_exemplars=3))
    rng = np.random.default_rng(2)
    for i in range(12):
        tracker.update([_det(10.0 + 5 * i, 10.0)], embeddings={0: _emb(rng)})
    assert len(tracker._tracks[0].exemplars) == 3


def test_due_for_embedding_tracks_the_schedule() -> None:
    tracker = ByteTracker(_cfg(min_hits=1, embed_every_n=2))
    tracker.update([_det(10.0, 10.0)])
    assert [t.hits for t in tracker.due_for_embedding()] == [1]
    tracker.update([_det(15.0, 10.0)])
    assert [t.hits for t in tracker.due_for_embedding()] == [2]
    tracker.update([_det(20.0, 10.0)])
    assert tracker.due_for_embedding() == []


def test_add_embedding_fills_exemplars_after_update() -> None:
    tracker = ByteTracker(_cfg(min_hits=1))
    rng = np.random.default_rng(3)
    active = tracker.update([_det(10.0, 10.0)])
    for track in tracker.due_for_embedding():
        tracker.add_embedding(track, _emb(rng))
    assert len(active[0].exemplars) == 1


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
    # Reach far enough that the centre gate admits both candidates, leaving
    # appearance as the only thing that can tell them apart. (iou_threshold=0
    # would instead disable the gate entirely and accept any pairing.)
    tracker.cfg = dataclasses.replace(tracker.cfg, max_centre_distance=200.0)
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


def test_centre_gate_matches_a_fast_object_that_no_longer_overlaps() -> None:
    """A jump larger than the box still associates when the centres stay close."""
    tracker = ByteTracker(_cfg(min_hits=1))
    tracker.update([_det(0.0, 0.0)])
    # 30 px jump on a 20 px box: zero IoU, so only the centre gate can match it.
    kept = tracker.update([_det(30.0, 0.0)])
    assert [t.id for t in kept] == [1]


def test_centre_gate_refuses_a_jump_beyond_its_reach() -> None:
    tracker = ByteTracker(_cfg(min_hits=1, max_centre_distance=1.0))
    tracker.update([_det(0.0, 0.0)])
    kept = tracker.update([_det(200.0, 200.0)])
    by_id = {t.id: t for t in kept}
    assert 2 in by_id and by_id[2].time_since_update == 0  # the far box is a new object
    assert by_id[1].time_since_update > 0  # the original track went unmatched, not stolen


def test_centre_distance_is_scale_free() -> None:
    from vision_memory.tracker import centre_distance_matrix

    small = np.array([[0.0, 0.0, 10.0, 10.0]])
    small_moved = np.array([[10.0, 0.0, 20.0, 10.0]])
    big = np.array([[0.0, 0.0, 100.0, 100.0]])
    big_moved = np.array([[100.0, 0.0, 200.0, 100.0]])
    # Same displacement in box widths must give the same number at either scale.
    assert centre_distance_matrix(small, small_moved)[0, 0] == pytest.approx(
        centre_distance_matrix(big, big_moved)[0, 0]
    )


def test_a_detection_covering_two_tracks_is_withheld_rather_than_guessed() -> None:
    """A box deep inside two tracks is a coin flip, and losing it swaps two ids."""
    tracker = ByteTracker(_cfg(min_hits=1))
    tracker.update([_det(0.0, 0.0), _det(60.0, 0.0)])
    before = {t.id: t.box.copy() for t in tracker.update([_det(2.0, 0.0), _det(62.0, 0.0)])}
    assert set(before) == {1, 2}

    # One detection now sits squarely on top of both tracks at once.
    tracker.update([_det(1.0, 0.0)])
    tracker._tracks[0]._kf.x[:2] = tracker._tracks[1]._kf.x[:2]  # force full overlap
    kept = tracker.update([_det(1.0, 0.0)])
    # Neither track may claim it: both coast, and no id changes hands.
    assert all(t.time_since_update > 0 for t in kept if t.id in (1, 2))

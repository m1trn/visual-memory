"""The engine composes the evaluated pipeline; the display reads it without moving it."""

from __future__ import annotations

import numpy as np

from vision_memory.config import (
    AnomalyConfig, MemoryConfig, ReidConfig, TrackerConfig, VideoConfig,
)
from vision_memory.detector import Detection
from vision_memory.engine import VisionEngine
from vision_memory.memory import VisualMemory

DIM = 8


class FakeDetector:
    """Two people walking right at half a pixel per frame, fixed confidence."""

    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame):
        k = int(frame[0, 0, 0])  # the frame index is stamped into the pixel
        self.calls += 1
        # Half a pixel per camera frame: a walker covering half a box width
        # during a 20-frame pipeline pass, which is what a live loop sees.
        return [Detection((10.0 + 0.5 * k, 10.0, 30.0 + 0.5 * k, 60.0), 0.9, 0, "person"),
                Detection((200.0 + 0.5 * k, 10.0, 220.0 + 0.5 * k, 60.0), 0.9, 0, "person")]


class FakeDescriber:
    """Each person has a fixed unit-vector look, plus a little noise."""

    dim = DIM

    def describe(self, frame, boxes, labels=None):
        rng = np.random.default_rng(int(frame[0, 0, 0]))
        out = {}
        for i, b in enumerate(boxes):
            base = np.eye(DIM, dtype=np.float32)[1 if b[0] < 100 else 5]
            v = base + rng.normal(scale=0.02, size=DIM).astype(np.float32)
            out[i] = v / np.linalg.norm(v)
        return out


class Cosine:
    def fit(self, *a): pass
    def score(self, a, b): return (np.asarray(a) * np.asarray(b)).sum(1)
    threshold = 0.7
    def predict(self, a, b): return self.score(a, b) >= 0.7


def _engine(tmp_path):
    mem = VisualMemory(MemoryConfig(db_path=str(tmp_path / "m.db"), index_path=str(tmp_path / "m.faiss"),
                                    exemplars_per_identity=5, reid_threshold=0.7), DIM)
    tracker = TrackerConfig(max_age=12, min_hits=2, iou_threshold=0.3, contested_iou=0.5,
                            max_centre_distance=1.5, measurement_noise=0.25, min_exemplar_confidence=0.3,
                            high_conf=0.5, low_conf=0.1, veto_views=3, appearance_veto=0.4, claim_margin=0.0,
                            appearance_weight=0.5, embed_every_n=1, max_exemplars=16)
    reid = ReidConfig(verifier="cosine", verifier_weights="", min_pair_gap=3, max_pairs_per_track=200, observation_quantile=0.9,
                      max_false_merge_rate=0.02, claim_margin=0.08, swap_margin=0.08,
                      min_new_identity_confidence=0.0, still_object_motion=0.5, merge_margin=0.15,
                      reconsider_every=15, continuity_bonus=0.15, spatial_scale=2.0, temporal_scale=2.0,
                      test_fraction=0.3, seed=0)
    video = VideoConfig(detect_every_n_frames=3, min_crop_px=8, crop_upper_fraction=0.6, embed_for_association=True)
    anomaly = AnomalyConfig(method="knn", k=3, shrinkage=0.1, n_estimators=10, nu=0.1, seed=0)
    return VisionEngine(FakeDetector(), FakeDescriber(), mem, Cosine(), 0.7, tracker, reid, video, anomaly, fps=10.0)


def _frame(k: int) -> np.ndarray:
    f = np.zeros((100, 300, 3), dtype=np.uint8)
    f[0, 0, 0] = k
    return f


def test_process_identifies_and_view_projects_without_advancing(tmp_path) -> None:
    eng = _engine(tmp_path)
    # A live loop under load: 20 camera frames between passes, more than
    # max_age. Tracks must move across the gap, not die in it.
    for k in range(0, 80, 20):
        result = eng.process(_frame(k), k)
    assert len(result.views) == 2
    ids = {v.identity_id for v in result.views}
    assert None not in ids and len(ids) == 2, "both people numbered, differently"

    boxes_now = {v.track_id: v.box.copy() for v in eng.view(60)}
    ahead = {v.track_id: v.box for v in eng.view(70)}
    for tid, box in boxes_now.items():
        assert ahead[tid][0] > box[0], "projection follows the estimated motion"
    again = {v.track_id: v.box for v in eng.view(60)}
    for tid in again:
        assert np.allclose(again[tid], boxes_now[tid]), "view must not move the tracker"
    assert eng.detector.calls == 4, "only processed frames reach the detector"
    assert len({v.track_id for v in result.views}) == 2, "the same two tracks survived every gap"


def test_similar_and_anomaly_read_the_same_memory(tmp_path) -> None:
    eng = _engine(tmp_path)
    for k in range(0, 60, 3):
        result = eng.process(_frame(k), k)
    left = min(result.views, key=lambda v: v.box[0])
    hits = eng.similar(left.track_id, k=3)
    assert hits and hits[0][0].id == left.identity_id, "a track is most similar to its own record"
    # Memory holds two people's exemplars; anomaly needs enough normals to say anything.
    a = eng._anomaly_of(next(t for t in eng.tracker._tracks if t.id == left.track_id))
    base = eng.anomaly_baseline("person")
    assert (a is None) == (base is None)
    eng.close()

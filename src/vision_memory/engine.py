"""The whole pipeline behind one object: detect, track, embed, remember, identify.

Everything a camera feed needs is here and nothing the evaluation did not
exercise: `process()` is the loop `scripts/mot_eval.py` scores, frame for frame,
so a live run behaves exactly like the numbers say it will. The three
capabilities all read the same embedding:

    identify   which stored identity does this track wear (the binder)
    similar    which stored identities look most like it (memory search)
    anomaly    how far its appearance sits from everything of its kind

The engine is deliberately synchronous. Concurrency belongs to the caller,
which knows its frame source: a live loop runs `process` on a worker thread
whenever it is free and draws `view()` at camera rate in between.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np

from vision_memory.anomaly import AnomalyDetector, build_detector
from vision_memory.appearance import build_describer, describe_detections
from vision_memory.config import (
    AnomalyConfig, AppearanceConfig, DetectorConfig, MemoryConfig, ReidConfig,
    TrackerConfig, VideoConfig,
)
from vision_memory.detector import Detector, YoloOnnxDetector
from vision_memory.memory import Identity, VisualMemory
from vision_memory.reid import Verifier
from vision_memory.reidentifier import IdentityBinder, ReIdentifier
from vision_memory.tracker import ByteTracker, Track

_MIN_EVIDENCE = 2  # observations before a track is worth identifying
_MIN_NORMALS = 8  # exemplars of a kind before "unusual for its kind" means anything


@dataclass(frozen=True)
class TrackView:
    """One object as the display should show it right now."""

    track_id: int
    box: np.ndarray
    label: str
    identity_id: int | None
    is_new: bool
    score: float
    hidden: bool
    anomaly: float | None


@dataclass
class FrameResult:
    """What `process` found in one frame."""

    frame_idx: int
    views: list[TrackView]
    identities: int
    created: int = 0
    rebound: int = 0
    taken: int = 0


@dataclass
class Counts:
    """Running totals for the on-screen summary."""

    processed: int = 0
    created: int = 0
    rebound: int = 0
    taken: int = 0


class VisionEngine:
    """Detection -> tracking -> embedding -> memory -> {identify, search, anomaly}."""

    def __init__(
        self,
        detector: Detector,
        describer,
        memory: VisualMemory,
        verifier: Verifier,
        threshold: float,
        tracker_cfg: TrackerConfig,
        reid_cfg: ReidConfig,
        video_cfg: VideoConfig,
        anomaly_cfg: AnomalyConfig,
        fps: float,
    ) -> None:
        self.detector = detector
        self.describer = describer
        self.memory = memory
        self.tracker = ByteTracker(tracker_cfg)
        self.reid = ReIdentifier(memory, verifier, threshold=threshold)
        self.binder = IdentityBinder(
            self.reid, fps, video_cfg.detect_every_n_frames, reid_cfg.reconsider_every,
            _MIN_EVIDENCE, swap_margin=reid_cfg.swap_margin,
            min_new_identity_confidence=reid_cfg.min_new_identity_confidence,
            convincing_confidence=tracker_cfg.high_conf, recent_views=tracker_cfg.veto_views,
            still_object_motion=reid_cfg.still_object_motion,
        )
        self.tracker_cfg = tracker_cfg
        self.video_cfg = video_cfg
        self.anomaly_cfg = anomaly_cfg
        self.counts = Counts()
        # One anomaly model per label: "unusual for a person" and "unusual for
        # a backpack" are different questions, and the routed vectors of
        # different kinds live in disjoint slices anyway.
        self._anomaly: dict[str, tuple[AnomalyDetector, int]] = {}
        self._last_frame_idx = -1
        self._lock = threading.Lock()

    @classmethod
    def from_configs(
        cls, detector_cfg: DetectorConfig, appearance_cfg: AppearanceConfig,
        memory_cfg: MemoryConfig, verifier: Verifier, threshold: float,
        tracker_cfg: TrackerConfig, reid_cfg: ReidConfig, video_cfg: VideoConfig,
        anomaly_cfg: AnomalyConfig, fps: float,
    ) -> "VisionEngine":
        describer = build_describer(appearance_cfg)
        memory = VisualMemory(memory_cfg, describer.dim)
        return cls(YoloOnnxDetector(detector_cfg), describer, memory, verifier, threshold,
                   tracker_cfg, reid_cfg, video_cfg, anomaly_cfg, fps)

    # ------------------------------------------------------------------ core
    def process(self, frame_bgr: np.ndarray, frame_idx: int) -> FrameResult:
        """Run the full pipeline on one frame.

        Every frame handed here is a detector frame. A caller that skips
        frames (a live loop dropping what it cannot keep up with) gets the
        offline pipeline with an adaptive skip: the tracker's motion model
        carries every track across the gap, and ``max_age`` counts processed
        frames, so a slow pipeline loses smoothness but not identities.
        """
        # The models run OUTSIDE the lock: they take hundreds of milliseconds
        # and touch no shared state, and the display thread must be able to
        # read `view()` while they run. Measured: holding the lock here pinned
        # the display to the pipeline's rate (0.6 fps against a 30 fps camera).
        detections = self.detector.detect(frame_bgr)
        embeddings = describe_detections(
            self.describer, frame_bgr, detections, self.tracker_cfg.min_exemplar_confidence
        )
        with self._lock:
            gap = frame_idx - self._last_frame_idx - 1 if self._last_frame_idx >= 0 else 0
            # Objects moved during the frames we did not look at; follow them
            # without ageing anyone - there was no detection to miss.
            self.tracker.advance(gap)
            active = self.tracker.update(detections, embeddings)
            events = self.binder.step(active, frame_idx)
            for lost in self.tracker.pop_lost():
                res, born, folded = self.binder.forget(lost.id)
                if res is not None and lost.exemplars and self.memory.get(res.identity_id) is not None:
                    first = (frame_idx if born is None else born) / self.binder.fps
                    last = (frame_idx - lost.time_since_update) / self.binder.fps
                    self.memory.remember(lost.label, lost.exemplars, first, last,
                                         max(lost.hits - folded, 0), identity_id=res.identity_id)
            self._last_frame_idx = frame_idx
            self.counts.processed += 1
            self.counts.created += events.created
            self.counts.rebound += events.rebound
            self.counts.taken += events.taken
            views = [self._view(t, t.box) for t in active if t.time_since_update <= self.video_cfg.detect_every_n_frames]
            return FrameResult(frame_idx, views, len(self.memory),
                               events.created, events.rebound, events.taken)

    def view(self, frame_idx: int) -> list[TrackView]:
        """Where everything is expected to be at ``frame_idx``, without touching state.

        For the display thread: projects each live track forward from the last
        processed frame with the filter's own velocity. Identities are read
        as they stand; nothing is advanced, so a concurrent `process` sees the
        same tracker it left.
        """
        with self._lock:
            steps = max(frame_idx - self._last_frame_idx, 0) if self._last_frame_idx >= 0 else 0
            fresh = self.video_cfg.detect_every_n_frames
            return [self._view(t, box) for t, box in self.tracker.peek(steps)
                    if t.time_since_update <= fresh]

    def _view(self, track: Track, box: np.ndarray) -> TrackView:
        res = self.binder.bound.get(track.id)
        return TrackView(
            track_id=track.id, box=np.asarray(box, dtype=np.float32), label=track.label,
            identity_id=None if res is None else res.identity_id,
            is_new=bool(res.is_new) if res is not None else False,
            score=float(res.score) if res is not None else 0.0,
            hidden=track.time_since_update > 0,
            anomaly=self._anomaly_of(track),
        )

    # --------------------------------------------------------------- search
    def similar(self, track_id: int, k: int = 5) -> list[tuple[Identity, float]]:
        """Stored identities that look most like this track, best first."""
        with self._lock:
            track = next((t for t in self.tracker._tracks if t.id == track_id), None)
            if track is None or not track.recent:
                return []
            query = np.mean(np.stack(track.recent), axis=0)
            out = []
            for identity_id, score in self.memory.search(query, k=k):
                ident = self.memory.get(identity_id)
                if ident is not None and ident.label == track.label:
                    out.append((ident, float(score)))
            return out

    # -------------------------------------------------------------- anomaly
    def _anomaly_of(self, track: Track) -> float | None:
        """Distance of this track's current look from everything of its kind in memory.

        The model for a label is refitted whenever memory has grown by a
        quarter since the last fit, so a fresh session starts with no verdict
        ("not enough seen yet") rather than a confident one built from three
        examples. Scores are the detector's raw distances; the display ranks
        them against the label's own normal distribution.
        """
        if not track.recent:
            return None
        normals = self._normals(track.label)
        if normals is None:
            return None
        model, _ = normals
        return float(model.score(np.mean(np.stack(track.recent), axis=0)[None])[0])

    def _normals(self, label: str) -> tuple[AnomalyDetector, int] | None:
        vectors = [self.memory.exemplar_vectors(i.id) for i in self.memory.all_identities()
                   if i.label == label]
        vectors = [v for v in vectors if len(v)]
        n = int(sum(len(v) for v in vectors))
        if n < _MIN_NORMALS:
            return None
        cached = self._anomaly.get(label)
        if cached is not None and n < cached[1] * 1.25:
            return cached
        model = build_detector(self.anomaly_cfg)
        model.fit(np.concatenate(vectors))
        self._anomaly[label] = (model, n)
        return self._anomaly[label]

    def anomaly_baseline(self, label: str) -> np.ndarray | None:
        """Scores of the stored normals themselves, so a live score can be ranked."""
        normals = self._normals(label)
        if normals is None:
            return None
        model, _ = normals
        vectors = np.concatenate([self.memory.exemplar_vectors(i.id) for i in self.memory.all_identities()
                                  if i.label == label and len(self.memory.exemplar_vectors(i.id))])
        return np.asarray(model.score(vectors), dtype=np.float32)

    # ---------------------------------------------------------------- close
    def close(self) -> None:
        with self._lock:
            self.memory.save()

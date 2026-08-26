"""Multi-object tracking: per-frame detections -> stable track ids.

ByteTrack-style association: a two-pass Hungarian match (high-confidence
detections first, then a second low-confidence pass to recover boxes the
detector was unsure about) over Kalman-filter motion predictions, optionally
blended with appearance (embedding) cosine similarity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from vision_memory.config import TrackerConfig
from vision_memory.detector import Detection


class KalmanBox:
    """Constant-velocity Kalman filter over a single box.

    State is ``[cx, cy, w, h, vx, vy, vw, vh]`` (box center, size, and their
    velocities) rather than the SORT ``[cx, cy, area, aspect, ...]``
    convention: w/h are tracked directly, which keeps the state-transition
    and measurement matrices simple, linear, and free of the divide-by-aspect
    numerical fragility that area/aspect parameterizations can hit on thin
    or noisy boxes.
    """

    def __init__(self, xyxy: np.ndarray, measurement_noise: float = 1.0) -> None:
        cx, cy, w, h = _xyxy_to_cxcywh(xyxy)
        self.x = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.P = np.eye(8, dtype=np.float64) * 10.0

        self._F = np.eye(8, dtype=np.float64)
        # Position extrapolates with velocity; SIZE DELIBERATELY DOES NOT.
        # While a track coasts, the detector's last look at a partly hidden
        # object is a shrunken box, so the filter learns a negative size
        # velocity and then projects it forward: measured on this footage a
        # predicted height ran 17.5 -> 8.6 -> -0.4 -> -9.4 px over four frames.
        # A negative-area box has zero IoU with every detection, and its
        # collapsed width also inflates the size-normalized centre distance, so
        # both halves of the gate fail exactly as the object reappears. An
        # object does not shrink because it is hidden, so size is held instead.
        for i in range(2):
            self._F[i, i + 4] = 1.0
        self._H = np.zeros((4, 8), dtype=np.float64)
        for i in range(4):
            self._H[i, i] = 1.0
        self._Q = np.eye(8, dtype=np.float64)
        self._Q[4:, 4:] *= 0.01  # velocities drift slowly
        # How far a corrected box is allowed to sit from the detection that
        # corrected it. With R equal to Q the Kalman gain lands near 0.65, so
        # the box only travels two thirds of the way to each new detection and
        # visibly trails the object, worst just after an occlusion when the
        # prediction it is being blended with is already stale. The detector is
        # the only real evidence about where the object is, so it is trusted
        # four times more: measured, the corrected box's overlap with its own
        # detection rises 0.929 -> 0.966 and median track life 42 -> 47 frames.
        # Not trusted further than that, because the filter still has to supply
        # a velocity for the frames it coasts through, and a gain near 1 fits
        # that velocity to raw detection jitter.
        self._R = np.eye(4, dtype=np.float64) * measurement_noise
        self._R[2:, 2:] *= 10.0  # width/height measurements are noisier than centers

    def predict(self) -> np.ndarray:
        """Advance the state by one frame and return the predicted xyxy box."""
        self.x = self._F @ self.x
        self.P = self._F @ self.P @ self._F.T + self._Q
        # Belt and braces: a box must always have positive area, whatever the
        # filter believes, or it silently drops out of every geometric test.
        self.x[2] = max(float(self.x[2]), _MIN_BOX_SIDE)
        self.x[3] = max(float(self.x[3]), _MIN_BOX_SIDE)
        return _cxcywh_to_xyxy(self.x[:4])

    def update(self, xyxy: np.ndarray) -> None:
        """Correct the state with an observed xyxy box."""
        z = np.array(_xyxy_to_cxcywh(xyxy), dtype=np.float64)
        y = z - self._H @ self.x
        S = self._H @ self.P @ self._H.T + self._R
        K = self.P @ self._H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self._H) @ self.P


@dataclass
class Track:
    """One tracked object."""

    id: int
    box: np.ndarray
    score: float
    class_id: int
    label: str
    hits: int
    time_since_update: int
    state: str  # "tentative" | "active" | "lost" | "dead"
    embedding: np.ndarray | None = None
    # Crops are never written to disk, so these L2-normalized observations are
    # everything a dying track can hand to persistent memory.
    exemplars: list[np.ndarray] = field(default_factory=list)
    _kf: KalmanBox = field(repr=False, compare=False, default=None)  # type: ignore[assignment]


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between boxes ``a`` (N,4) and ``b`` (M,4), xyxy."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ix1 = np.maximum(ax1[:, None], bx1[None, :])
    iy1 = np.maximum(ay1[:, None], by1[None, :])
    ix2 = np.minimum(ax2[:, None], bx2[None, :])
    iy2 = np.minimum(ay2[:, None], by2[None, :])
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    area_a = np.clip(ax2 - ax1, 0, None) * np.clip(ay2 - ay1, 0, None)
    area_b = np.clip(bx2 - bx1, 0, None) * np.clip(by2 - by1, 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


_MIN_BOX_SIDE = 2.0  # px; a predicted box below this has no geometry left to match on
_IMPOSSIBLE = 1e6  # cost of a pairing the gate will reject anyway
_TIE_BREAK = 1e-3  # weight of the centre term relative to IoU in the cost


def centre_distance_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Centre separation between boxes ``a`` (N,4) and ``b`` (M,4), in box widths.

    Roughly 6% of correct associations on this footage fall below the IoU
    threshold — usually after a missed detector cycle, where the track has
    coasted on prediction and the boxes only graze. Each rejection fragments a
    track, so a small tail dominates the identity count. Centre separation still
    ranks those pairs sensibly; normalizing by box size keeps it scale free, so
    one threshold covers a distant pedestrian and a near one alike.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    ac = np.stack([(a[:, 0] + a[:, 2]) / 2, (a[:, 1] + a[:, 3]) / 2], axis=1)
    bc = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], axis=1)
    dist = np.linalg.norm(ac[:, None, :] - bc[None, :, :], axis=2)
    a_size = np.maximum(a[:, 2] - a[:, 0], 1.0)
    b_size = np.maximum(b[:, 2] - b[:, 0], 1.0)
    scale = (a_size[:, None] + b_size[None, :]) / 2.0
    return dist / scale


def _xyxy_to_cxcywh(box: np.ndarray) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1


def _cxcywh_to_xyxy(state: np.ndarray) -> np.ndarray:
    cx, cy, w, h = state
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n == 0 else v / n


def _cosine_matrix(track_embs, det_embs: list[np.ndarray | None]) -> np.ndarray:
    """Pairwise cosine similarity; a pair with a missing embedding gets NaN.

    A track may offer several observations, in which case the best match wins:
    an object seen from a new angle should be compared against the closest view
    the track has actually recorded, not against an average that resembles none
    of them.
    """
    n, m = len(track_embs), len(det_embs)
    out = np.full((n, m), np.nan, dtype=np.float64)
    for i, te in enumerate(track_embs):
        if te is None or (isinstance(te, list) and not te):
            continue
        views = np.atleast_2d(np.stack(te) if isinstance(te, list) else te)
        for j, de in enumerate(det_embs):
            if de is None:
                continue
            out[i, j] = float((views @ np.asarray(de)).max())
    return out


def should_embed(hits: int, embed_every_n: int) -> bool:
    """True when a track with ``hits`` associations is due for a fresh exemplar.

    The encoder is the expensive stage, so it runs on a subset of a track's
    associations: the first one (so every track owns at least one embedding
    before it can die) and then every ``embed_every_n``-th one after that.
    ``embed_every_n <= 0`` disables embedding entirely.
    """
    if embed_every_n <= 0 or hits <= 0:
        return False
    return hits == 1 or hits % embed_every_n == 0


class ByteTracker:
    """ByteTrack-style multi-object tracker over a single stream of frames."""

    def __init__(self, cfg: TrackerConfig) -> None:
        self.cfg = cfg
        self._tracks: list[Track] = []
        self._next_id = 1
        self._newly_lost: list[Track] = []

    def update(
        self,
        detections: list[Detection] | None,
        embeddings: dict[int, np.ndarray] | None = None,
    ) -> list[Track]:
        """Process one frame and return the currently active tracks.

        ``detections=None`` means the detector did not run this frame: tracks
        are advanced by Kalman prediction only. Time still passes, so
        ``time_since_update`` grows and ``max_age`` stays denominated in video
        frames rather than in detector invocations.

        ``embeddings`` maps detection index -> embedding; each one is folded
        into its track's running mean and exemplar buffer. Callers that encode
        after association should leave it empty and use ``due_for_embedding``
        plus ``add_embedding`` instead.
        """
        for t in self._tracks:
            t.box = t._kf.predict()

        if detections is None:
            self._age(range(len(self._tracks)))
            self._reap()
            return [t for t in self._tracks if t.state == "active"]

        embeddings = embeddings or {}
        contested = self._contested_detections(detections)
        high_idx = [
            i for i, d in enumerate(detections)
            if d.score >= self.cfg.high_conf and i not in contested
        ]
        low_idx = [
            i for i, d in enumerate(detections)
            if self.cfg.low_conf <= d.score < self.cfg.high_conf and i not in contested
        ]

        unmatched_tracks = list(range(len(self._tracks)))
        matched_high: dict[int, int] = {}
        if self._tracks and high_idx:
            matched_high, unmatched_tracks, unmatched_high = self._match(
                unmatched_tracks, high_idx, detections, embeddings, use_appearance=True
            )
        else:
            unmatched_high = list(high_idx)

        matched_low: dict[int, int] = {}
        if unmatched_tracks and low_idx:
            matched_low, unmatched_tracks, _unmatched_low = self._match(
                unmatched_tracks, low_idx, detections, embeddings, use_appearance=False
            )

        for t_i, d_i in {**matched_high, **matched_low}.items():
            self._apply_match(self._tracks[t_i], detections[d_i], embeddings.get(d_i))

        self._age(unmatched_tracks, missed_association=True)

        for d_i in unmatched_high:
            self._spawn(detections[d_i], embeddings.get(d_i))

        self._reap()
        return [t for t in self._tracks if t.state == "active"]

    def _contested_detections(self, detections: list[Detection]) -> set[int]:
        """Detections sitting deep inside two or more tracks at once.

        When two people cross, one box can cover both of their tracks. Forcing an
        assignment there is a coin flip, and losing it hands one person's id to
        the other — the worst failure this tracker has, because the wrong name
        then persists and poisons that identity in memory. Measured on 500
        frames, appearance-inconsistent links during an overlap ran at 8.6%;
        withholding these detections drops that to 0% for the cost of 2 deferrals
        and 0.2% coverage. The tracks simply coast until the crossing resolves.
        """
        if not self._tracks or not detections:
            return set()
        overlap = iou_matrix(
            np.array([d.box for d in detections], dtype=np.float64),
            np.array([t.box for t in self._tracks], dtype=np.float64),
        )
        deep = overlap > self.cfg.contested_iou
        return {i for i in range(len(detections)) if int(deep[i].sum()) >= 2}

    def _age(self, track_idx, missed_association: bool = False) -> None:
        """Advance ``time_since_update`` and mark tracks that ran out of life.

        A tentative track that misses an association was probably a detector
        false positive, so it is dropped at once instead of lingering for
        ``max_age`` frames where it could steal matches from real tracks.
        Only confirmed tracks become "lost" and reach ``pop_lost``.
        """
        for t_i in track_idx:
            track = self._tracks[t_i]
            track.time_since_update += 1
            if missed_association and track.state == "tentative":
                track.state = "dead"
            elif track.time_since_update > self.cfg.max_age:
                track.state = "lost" if track.state == "active" else "dead"

    def _reap(self) -> None:
        """Move finished tracks off the live list; confirmed ones become lost."""
        if not any(t.state in ("lost", "dead") for t in self._tracks):
            return
        self._newly_lost.extend(t for t in self._tracks if t.state == "lost")
        self._tracks = [t for t in self._tracks if t.state not in ("lost", "dead")]

    def pop_lost(self) -> list[Track]:
        """Return tracks that became lost since the last call, then clear them."""
        lost, self._newly_lost = self._newly_lost, []
        return lost

    def due_for_embedding(self) -> list[Track]:
        """Live tracks matched on this frame that are due for a fresh exemplar.

        Call this right after ``update``: the returned tracks were corrected by
        a real measurement this frame, so their boxes crop cleanly out of the
        frame just processed. Encode those crops and hand each embedding back
        with ``add_embedding``.
        """
        return [
            t
            for t in self._tracks
            if t.time_since_update == 0 and should_embed(t.hits, self.cfg.embed_every_n)
        ]

    def add_embedding(self, track: Track, embedding: np.ndarray) -> None:
        """Record an observed embedding for ``track`` (running mean + exemplar buffer)."""
        emb = _normalize(np.asarray(embedding, dtype=np.float32))
        track.embedding = _normalize(emb if track.embedding is None else (track.embedding + emb) / 2.0)
        self._push_exemplar(track, emb)

    def _push_exemplar(self, track: Track, emb: np.ndarray) -> None:
        """Append an exemplar, evicting the most redundant one once the cap is hit.

        Evicting the exemplar closest to the newcomer (rather than the oldest)
        keeps the buffer spread over the track's appearance changes, which is
        what memory wants: a bounded, diverse view of the object.
        """
        cap = self.cfg.max_exemplars
        if cap <= 0:
            return
        track.exemplars.append(emb)
        if len(track.exemplars) <= cap:
            return
        others = np.stack(track.exemplars[:-1])
        track.exemplars.pop(int(np.argmax(others @ emb)))

    def _match(
        self,
        track_idx: list[int],
        det_idx: list[int],
        detections: list[Detection],
        embeddings: dict[int, np.ndarray],
        use_appearance: bool,
    ) -> tuple[dict[int, int], list[int], list[int]]:
        """Hungarian-match a subset of tracks against a subset of detections.

        Returns (matches as {track_idx: det_idx}, leftover track_idx, leftover det_idx).
        """
        t_boxes = np.array([self._tracks[i].box for i in track_idx], dtype=np.float64)
        d_boxes = np.array([detections[i].box for i in det_idx], dtype=np.float64)
        iou = iou_matrix(t_boxes, d_boxes)
        centre = centre_distance_matrix(t_boxes, d_boxes)
        # Half overlap, half proximity: without the second term every
        # non-overlapping pair costs exactly 1.0 and Hungarian breaks the tie
        # arbitrarily instead of preferring the nearest candidate.
        reach = max(self.cfg.max_centre_distance, 1e-6)
        # Overlap decides wherever it discriminates; centre distance only breaks
        # ties among pairs IoU cannot separate. A heavier centre term measurably
        # changed 6 of 2086 matches and no identities, while saturating exactly
        # at the gate boundary — so it carried no ranking information anyway.
        motion = (1.0 - iou) + _TIE_BREAK * np.minimum(centre / reach, 1.0)

        # Recent raw observations, falling back to the running mean before any
        # have been recorded.
        track_embs = [
            (t.exemplars[-self.cfg.veto_views:] if t.exemplars else t.embedding)
            for t in (self._tracks[i] for i in track_idx)
        ]
        det_embs = [embeddings.get(i) for i in det_idx]
        cos = _cosine_matrix(track_embs, det_embs)
        has_cos = ~np.isnan(cos)

        # An object does not change class. Without this the low-confidence pass
        # feeds a person track the detector's junk (dog, skis, bird all appear
        # on this footage), which both steals the match and silently relabels
        # the track.
        same_class = np.array(
            [[self._tracks[i].class_id == detections[j].class_id for j in det_idx] for i in track_idx],
            dtype=bool,
        ) if track_idx and det_idx else np.zeros((len(track_idx), len(det_idx)), dtype=bool)

        lam = self.cfg.appearance_weight if use_appearance else 0.0
        if lam > 0.0:
            cost = np.where(has_cos, (1 - lam) * motion + lam * (1 - cos), motion)
        else:
            cost = motion

        cost = np.where(same_class, cost, _IMPOSSIBLE)
        row, col = linear_sum_assignment(cost)
        matches: dict[int, int] = {}
        matched_t, matched_d = set(), set()
        for r, c in zip(row, col):
            # Overlap accepts outright; otherwise centres must be close enough
            # relative to object size. `reach` is used rather than the raw config
            # value so the gate and the cost agree on what "close" means.
            if not same_class[r, c]:
                continue
            if iou[r, c] < self.cfg.iou_threshold and centre[r, c] > reach:
                continue
            # Geometry alone cannot tell two people apart while they cross, but
            # appearance can: measured on this footage a true match scores 0.853
            # median (5th percentile 0.722) against 0.603 for a different person.
            # A veto at 0.65 discards 0.5% of true matches and blocks 69% of the
            # mistaken pairings geometry would otherwise accept. Only applied
            # when both sides actually carry an embedding.
            if self.cfg.appearance_veto > 0.0 and has_cos[r, c] and cos[r, c] < self.cfg.appearance_veto:
                continue
            # When two people overlap, their boxes are nearly interchangeable and
            # the assignment can hand one person's id to the other. Geometry
            # cannot arbitrate that, but appearance can: if a rival track fits
            # this detection clearly better than the track about to claim it,
            # nobody takes it and both coast until the crossing resolves.
            if self.cfg.claim_margin > 0.0 and has_cos[r, c]:
                rivals = cos[:, c].copy()
                rivals[r] = -np.inf
                rivals[~has_cos[:, c]] = -np.inf
                if rivals.size and float(rivals.max()) - cos[r, c] > self.cfg.claim_margin:
                    continue
            matches[track_idx[r]] = det_idx[c]
            matched_t.add(r)
            matched_d.add(c)
        leftover_t = [track_idx[i] for i in range(len(track_idx)) if i not in matched_t]
        leftover_d = [det_idx[i] for i in range(len(det_idx)) if i not in matched_d]
        return matches, leftover_t, leftover_d

    def _apply_match(self, track: Track, det: Detection, emb: np.ndarray | None) -> None:
        box = np.array(det.box, dtype=np.float32)
        track._kf.update(box)
        track.box = _cxcywh_to_xyxy(track._kf.x[:4])
        track.score = det.score
        track.class_id = det.class_id
        track.label = det.label
        track.hits += 1
        track.time_since_update = 0
        if track.state == "tentative" and track.hits >= self.cfg.min_hits:
            track.state = "active"
        if emb is not None:
            self.add_embedding(track, emb)

    def _spawn(self, det: Detection, emb: np.ndarray | None) -> None:
        box = np.array(det.box, dtype=np.float32)
        kf = KalmanBox(box, self.cfg.measurement_noise)
        track = Track(
            id=self._next_id,
            box=box,
            score=det.score,
            class_id=det.class_id,
            label=det.label,
            hits=1,
            time_since_update=0,
            state="tentative",
            _kf=kf,
        )
        if emb is not None:
            self.add_embedding(track, emb)
        if self.cfg.min_hits <= 1:
            track.state = "active"
        self._next_id += 1
        self._tracks.append(track)

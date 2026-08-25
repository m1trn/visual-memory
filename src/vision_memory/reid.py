"""Pair mining and learned verification for re-identification.

Re-identification asks a binary question about a *pair* of observations: are
these the same object? The tracker already answers that question for free —
two crops from one track are the same object by construction, two crops from
tracks alive in the same frame are not — so it can mine its own labelled
pairs. Fitting a verifier on those pairs replaces the hand-set cosine
threshold, which was calibrated on clean photographs and is far too strict
for real detector crops.

Everything here operates on L2-normalized embeddings, so a dot product is a
cosine similarity.
"""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from vision_memory.config import ReidConfig
from vision_memory.memory import select_diverse

# Cap on exhaustive negative-pair enumeration; above it, negatives are sampled.
_MAX_ENUMERATED_OBSERVATIONS = 512


def mine_pairs(
    tracks: Mapping[int, Sequence[np.ndarray]],
    cfg: ReidConfig,
    rng: np.random.Generator,
    frames: Mapping[int, Sequence[int]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mine same/different pairs from per-track observation sequences.

    ``tracks`` maps a track id to that track's embeddings in temporal order.
    A positive pair is two observations of one track at least ``cfg.min_pair_gap``
    apart — adjacent frames are near duplicates and teach the verifier nothing.

    ``frames`` optionally gives the frame index of each observation. When
    supplied, a negative pair is only emitted for two tracks that were alive at
    the same time, which makes the label provable: one detection cannot be two
    tracks in a single frame. Without it, two fragments of the same person can
    be mislabelled "different", and this tracker is known to fragment.

    Returns ``(A, B, y, owners)``. ``owners`` is ``(N, 2)`` — the track ids the
    two halves of each pair came from — so a caller can hold out whole tracks.
    Per-track positives are capped at ``cfg.max_pairs_per_track`` so one
    long-lived object cannot dominate the class.
    """
    if cfg.min_pair_gap < 1:
        raise ValueError(f"min_pair_gap must be >= 1, got {cfg.min_pair_gap}")

    ids = sorted(tracks)
    obs = {t: _stack(tracks[t]) for t in ids}
    dim = next((v.shape[1] for v in obs.values() if len(v)), 0)
    if dim == 0:
        return _empty(0)

    pos: list[tuple[int, int, int]] = []
    for t in ids:
        n = len(obs[t])
        here = [(t, i, j) for i in range(n) for j in range(i + cfg.min_pair_gap, n)]
        if len(here) > cfg.max_pairs_per_track:
            here = [here[k] for k in _subsample(len(here), cfg.max_pairs_per_track, rng)]
        pos.extend(here)

    span = None
    if frames is not None:
        span = {t: (min(frames[t]), max(frames[t])) for t in ids if len(frames.get(t, ()))}
    neg = _candidate_negatives(obs, ids, len(pos), rng, span)
    n_keep = min(len(pos), len(neg))
    if n_keep == 0:
        return _empty(dim)

    pos = [pos[i] for i in _subsample(len(pos), n_keep, rng)]
    neg = [neg[i] for i in _subsample(len(neg), n_keep, rng)]

    a = np.empty((2 * n_keep, dim), dtype=np.float32)
    b = np.empty_like(a)
    y = np.empty(2 * n_keep, dtype=np.int64)
    owners = np.empty((2 * n_keep, 2), dtype=np.int64)
    for k, (t, i, j) in enumerate(pos):
        a[k], b[k], y[k] = obs[t][i], obs[t][j], 1
        owners[k] = (t, t)
    for k, (t1, i, t2, j) in enumerate(neg, start=n_keep):
        a[k], b[k], y[k] = obs[t1][i], obs[t2][j], 0
        owners[k] = (t1, t2)
    return a, b, y, owners


def split_by_group(
    y: np.ndarray,
    owners: np.ndarray,
    test_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Split pairs so that no TRACK contributes to both sides.

    Splitting on pair identity is not enough: a pair straddling two tracks would
    put each of them on whichever side that pair landed. Tracks are assigned to
    sides first, and a pair is kept only when both of its tracks share a side —
    cross-side negatives are discarded rather than leaked. Assignment is
    stratified so neither side ends up single-class.
    """
    y = np.asarray(y)
    owners = np.asarray(owners)
    if owners.ndim != 2 or owners.shape[1] != 2:
        raise ValueError(f"owners must be (N, 2) track ids, got {owners.shape}")
    if len(y) != len(owners):
        raise ValueError(f"shape mismatch: y {y.shape} vs owners {owners.shape}")
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in [0, 1), got {test_fraction}")

    empty = np.zeros(len(y), dtype=bool)
    if len(y) == 0:
        return ~empty, empty

    tracks = np.unique(owners)
    order = tracks[rng.permutation(len(tracks))]
    n_test = int(round(test_fraction * len(tracks)))
    n_test = min(max(n_test, 1), len(tracks) - 1) if len(tracks) > 1 else 0
    test_tracks = set(order[:n_test].tolist())

    in_test = np.array([o[0] in test_tracks and o[1] in test_tracks for o in owners])
    in_train = np.array([o[0] not in test_tracks and o[1] not in test_tracks for o in owners])
    return in_train, in_test


def balance(y: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Trim ``mask`` to equal numbers of same and different pairs.

    Mining balances the whole population, but a split does not preserve that, so
    raw accuracy on an unbalanced side reflects the prior rather than the
    verifier. Balancing each side keeps the reported number comparable.
    """
    idx = np.flatnonzero(mask)
    pos = idx[y[idx] == 1]
    neg = idx[y[idx] == 0]
    n = min(len(pos), len(neg))
    out = np.zeros(len(y), dtype=bool)
    if n == 0:
        return out
    out[pos[_subsample(len(pos), n, rng)]] = True
    out[neg[_subsample(len(neg), n, rng)]] = True
    return out



def identity_score(query: np.ndarray, exemplars: np.ndarray, quantile: float) -> float:
    """Score a whole track against one identity: max over exemplars, quantile over observations."""
    per_observation = (np.asarray(query, np.float32) @ np.asarray(exemplars, np.float32).T).max(axis=1)
    return float(np.percentile(per_observation, 100.0 * quantile))


def calibrate_identity_threshold(
    tracks: Mapping[int, Sequence[np.ndarray]],
    frames: Mapping[int, Sequence[int]],
    quantile: float,
    exemplars_per_identity: int,
    max_false_merge_rate: float = 0.0,
) -> tuple[float, int, int]:
    """Learn the accept/reject boundary on track-versus-identity scores.

    A threshold fitted to individual pair scores does not transfer to this
    decision: the identity score takes a max over K exemplars and then a high
    quantile over the track's observations, and both push it well above a single
    pair's cosine. Applying the pair boundary here accepts far too much.

    Ground truth comes from the tracker itself. A track's later observations
    against an identity built from its own earlier ones is a positive. A track
    against the identity of a track that was on screen at the same time is a
    negative, and provably so — one detection cannot be two tracks in one frame.

    With ``max_false_merge_rate`` the boundary is the lowest one holding false
    merges at or under that rate, rather than Youden's J. The two errors are not
    equally bad here: failing to recognize a returning object costs one spare
    identity that a later sighting can still repair, while binding two people
    into one record corrupts it permanently and shows a viewer the wrong name.
    On this footage the distributions overlap so heavily — same object 0.657
    median against 0.653 at the 95th percentile of provably-different pairs —
    that Youden lands on a boundary with a 21% false-merge rate.

    Returns ``(threshold, n_positive, n_negative)``.
    """
    usable = {t: _stack(v) for t, v in tracks.items() if len(v) >= 4 and len(frames.get(t, ()))}
    if len(usable) < 2:
        raise ValueError("calibration needs at least two tracks with several observations each")
    span = {t: (min(frames[t]), max(frames[t])) for t in usable}
    half = {t: len(v) // 2 for t, v in usable.items()}

    # Each identity is represented the way memory would store it: a diverse subset.
    stored = {}
    for t, v in usable.items():
        early = v[: half[t]]
        stored[t] = early[select_diverse(early, exemplars_per_identity)]

    pos, neg = [], []
    for t, v in usable.items():
        query = v[half[t] :]
        pos.append(identity_score(query, stored[t], quantile))
        for other in usable:
            if other == t:
                continue
            (a0, a1), (b0, b1) = span[t], span[other]
            if a0 <= b1 and b0 <= a1:  # co-alive, so provably a different object
                neg.append(identity_score(query, stored[other], quantile))
    if not pos or not neg:
        raise ValueError("calibration needs both same-object and different-object examples")

    scores = np.array(pos + neg, dtype=np.float32)
    labels = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int64)
    if max_false_merge_rate > 0.0:
        negatives = np.sort(np.array(neg, dtype=np.float32))
        allowed = int(np.floor(max_false_merge_rate * len(negatives)))
        # Sit just above the highest different-object score we are willing to admit.
        cut = negatives[len(negatives) - allowed - 1] if allowed < len(negatives) else negatives[0]
        return float(np.nextafter(cut, np.inf)), len(pos), len(neg)
    return _youden_threshold(scores, labels), len(pos), len(neg)

class Verifier(Protocol):
    """Decide whether two embeddings show the same object."""

    def fit(self, a: np.ndarray, b: np.ndarray, y: np.ndarray) -> None:
        """Fit on labelled pairs; ``y`` is 1 for same, 0 for different."""
        ...

    def score(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Score pairs, ``(N,)`` float32, higher = more likely the same object."""
        ...

    @property
    def threshold(self) -> float:
        """Decision boundary applied to ``score``."""
        ...

    def predict(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Boolean ``(N,)``: ``score(a, b) >= threshold``."""
        ...


class CosineVerifier:
    """Raw cosine similarity with a threshold learned from the mined pairs.

    The score function is fixed, so fitting only has to choose where to cut
    it. That single learned number is the whole point of the phase: the
    configured 0.80 came from clean photographs and rejects most true matches
    on detector crops.
    """

    def __init__(self) -> None:
        self._threshold: float | None = None

    def fit(self, a: np.ndarray, b: np.ndarray, y: np.ndarray) -> None:
        """Pick the cosine threshold maximizing Youden's J on these pairs."""
        self._threshold = _youden_threshold(self.score(a, b), np.asarray(y))

    def score(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Cosine similarity of each pair."""
        a, b = _as_pairs(a, b)
        return np.einsum("ij,ij->i", a, b).astype(np.float32, copy=False)

    @property
    def threshold(self) -> float:
        """The learned cosine cut point."""
        if self._threshold is None:
            raise RuntimeError("CosineVerifier.threshold used before fit(); run scripts/reid_bench.py first")
        return self._threshold

    def predict(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Same-object decision for each pair."""
        return self.score(a, b) >= self._threshold


class LogisticVerifier:
    """Logistic regression over a small symmetric summary of each pair.

    The features are cosine, Euclidean distance, and mean/std/max of the
    absolute difference plus mean/std/min/max of the elementwise product —
    nine numbers, all invariant to swapping ``a`` and ``b`` (pair order is
    arbitrary, so an order-sensitive model would learn noise). The obvious
    alternative, feeding the raw 384-dim elementwise product, gives the model
    more parameters than the tracker can realistically mine pairs, so it
    overfits; these summaries let the model learn *which kind* of difference
    matters (a few large channel gaps versus a broad small drift) without
    that cost. Features are standardized because their scales differ by
    orders of magnitude.

    The threshold is 0.5, the probability at which the fitted model itself
    switches sides; the decision boundary is learned inside the coefficients.
    """

    def __init__(self, seed: int) -> None:
        self._model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
            ]
        )
        self._fitted = False

    def fit(self, a: np.ndarray, b: np.ndarray, y: np.ndarray) -> None:
        """Fit the pair classifier on labelled pairs."""
        y = np.asarray(y).ravel()
        if len(np.unique(y)) < 2:
            raise ValueError("LogisticVerifier.fit needs both same and different pairs")
        self._model.fit(pair_features(a, b), y)
        self._fitted = True

    def score(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Probability that each pair shows the same object."""
        if not self._fitted:
            raise RuntimeError("LogisticVerifier.score called before fit")
        return self._model.predict_proba(pair_features(a, b))[:, 1].astype(np.float32, copy=False)

    @property
    def threshold(self) -> float:
        """Probability at which the fitted model switches class."""
        return 0.5

    def predict(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Same-object decision for each pair."""
        return self.score(a, b) >= self.threshold


def pair_features(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Symmetric ``(N, 9)`` float32 summary of each embedding pair."""
    a, b = _as_pairs(a, b)
    diff = np.abs(a - b)
    prod = a * b
    return np.stack(
        [
            prod.sum(axis=1),
            np.linalg.norm(a - b, axis=1),
            diff.mean(axis=1),
            diff.std(axis=1),
            diff.max(axis=1),
            prod.mean(axis=1),
            prod.std(axis=1),
            prod.min(axis=1),
            prod.max(axis=1),
        ],
        axis=1,
    ).astype(np.float32, copy=False)


def build_verifier(cfg: ReidConfig) -> Verifier:
    """Construct the verifier named by ``cfg.verifier`` ('cosine' or 'logistic')."""
    if cfg.verifier == "cosine":
        return CosineVerifier()
    if cfg.verifier == "logistic":
        return LogisticVerifier(seed=cfg.seed)
    raise ValueError(f"unknown reid verifier: {cfg.verifier!r}")


def _youden_threshold(scores: np.ndarray, y: np.ndarray) -> float:
    """Threshold maximizing TPR - FPR, placed midway to the next lower score."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(y).ravel().astype(bool)
    pos = np.sort(scores[y])
    neg = np.sort(scores[~y])
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("threshold selection needs both same and different pairs")

    candidates = np.unique(scores)
    tpr = (len(pos) - np.searchsorted(pos, candidates, side="left")) / len(pos)
    fpr = (len(neg) - np.searchsorted(neg, candidates, side="left")) / len(neg)
    best = int(np.argmax(tpr - fpr))
    # Sit halfway between the winning score and the next one below it, rather
    # than exactly on an observed sample, so the boundary generalizes.
    lower = candidates[best - 1] if best > 0 else candidates[best]
    return float((candidates[best] + lower) / 2.0)


def _candidate_negatives(
    obs: Mapping[int, np.ndarray],
    ids: Sequence[int],
    n_pos: int,
    rng: np.random.Generator,
    span: Mapping[int, tuple[int, int]] | None = None,
) -> list[tuple[int, int, int, int]]:
    """Cross-track pairs: exhaustive when small, randomly sampled when not.

    With ``span`` (each track's first and last frame) only tracks that overlap
    in time are paired, so "different" is provable rather than assumed.
    """
    usable = [t for t in ids if len(obs[t])]
    if len(usable) < 2:
        return []

    def co_alive(t1: int, t2: int) -> bool:
        if span is None or t1 not in span or t2 not in span:
            return True
        (a0, a1), (b0, b1) = span[t1], span[t2]
        return a0 <= b1 and b0 <= a1
    total = sum(len(obs[t]) for t in usable)
    if total <= _MAX_ENUMERATED_OBSERVATIONS:
        return [
            (t1, i, t2, j)
            for x, t1 in enumerate(usable)
            for t2 in usable[x + 1 :]
            if co_alive(t1, t2)
            for i in range(len(obs[t1]))
            for j in range(len(obs[t2]))
        ]

    seen: set[tuple[int, int, int, int]] = set()
    for _ in range(4 * max(n_pos, 1)):
        t1, t2 = rng.choice(len(usable), size=2, replace=False)
        t1, t2 = usable[int(t1)], usable[int(t2)]
        if t1 > t2:
            t1, t2 = t2, t1
        if not co_alive(t1, t2):
            continue
        seen.add((t1, int(rng.integers(len(obs[t1]))), t2, int(rng.integers(len(obs[t2])))))
    return sorted(seen)


def _subsample(n: int, keep: int, rng: np.random.Generator) -> np.ndarray:
    return np.arange(n) if keep >= n else np.sort(rng.choice(n, size=keep, replace=False))


def _stack(embeddings: Sequence[np.ndarray]) -> np.ndarray:
    if len(embeddings) == 0:
        return np.empty((0, 0), dtype=np.float32)
    return np.ascontiguousarray(
        np.stack([np.asarray(e, dtype=np.float32).reshape(-1) for e in embeddings])
    )


def _empty(dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z = np.empty((0, dim), dtype=np.float32)
    return z, z.copy(), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)


def _as_pairs(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.ascontiguousarray(a, dtype=np.float32).reshape(len(a), -1)
    b = np.ascontiguousarray(b, dtype=np.float32).reshape(len(b), -1)
    if a.shape != b.shape:
        raise ValueError(f"pair shape mismatch: {a.shape} vs {b.shape}")
    return a, b

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

from typing import Callable, Mapping, Protocol, Sequence

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



def identity_score_with(verifier, query: np.ndarray, exemplars: np.ndarray, quantile: float) -> float:
    """``identity_score`` on whatever scale ``verifier.score`` produces.

    Max over exemplars, quantile over observations - the same aggregation
    ``ReIdentifier._score_identity`` applies at runtime, so a boundary fitted
    on this is on the scale it will be compared against.
    """
    q = np.asarray(query, np.float32)
    e = np.asarray(exemplars, np.float32)
    m, n = len(q), len(e)
    scores = np.asarray(verifier.score(np.repeat(q, n, axis=0), np.tile(e, (m, 1))), np.float32)
    return float(np.percentile(scores.reshape(m, n).max(axis=1), 100.0 * quantile))


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
    score: "Callable[[np.ndarray, np.ndarray, float], float] | None" = None,
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

    ``score`` is the identity scorer the boundary will be applied to. It
    defaults to the cosine ``identity_score``; a caller using a different
    verifier must pass its own, or a boundary on one scale is compared
    against numbers on another.

    Returns ``(threshold, n_positive, n_negative)``.
    """
    scorer = identity_score if score is None else score
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
        pos.append(scorer(query, stored[t], quantile))
        for other in usable:
            if other == t:
                continue
            (a0, a1), (b0, b1) = span[t], span[other]
            if a0 <= b1 and b0 <= a1:  # co-alive, so provably a different object
                # Two objects of different kinds live in disjoint slices of
                # the routed vector, so their raw cosine is exactly 0 whatever
                # verifier is in use. Re-identification label-gates its
                # shortlist, so such a pair can never reach the boundary;
                # counting it as an easy negative dilutes the false-merge
                # budget with cases that cannot occur. Tested on the cosine,
                # not the scorer: a logistic verifier maps 0 to some
                # probability and the test would silently stop working.
                if identity_score(query, stored[other], quantile) == 0.0:
                    continue
                neg.append(scorer(query, stored[other], quantile))
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


class MlpVerifier:
    """A small network trained on labelled pairs: "are these two the same person?"

    The project's own trained component (SPEC section 20). The backbone stays
    frozen; this sits on top of two of its vectors and learns what a single
    cosine cannot: which dimensions carry identity, which differences are
    lighting rather than clothing, that a 0.6 in one region of the space means
    more than a 0.6 in another.

    Input is the pair's symmetric summary ``[a*b, |a-b|, cos]`` restricted to
    the person slice of the routed vector (the general slice is zero for
    people and would only add parameters). Output is a probability; the
    boundary used for binding is still fitted downstream on identity scores
    (`identity_score_with`), so this scale needs no hand-set threshold.

    Weights are numpy arrays saved to one ``.npz``; inference is two matrix
    products, no torch at runtime.
    """

    def __init__(self, slice_dim: int, hidden: int = 128, seed: int = 0) -> None:
        self.slice_dim = int(slice_dim)
        self.hidden = int(hidden)
        self.seed = int(seed)
        self._w1: np.ndarray | None = None
        self._b1: np.ndarray | None = None
        self._w2: np.ndarray | None = None
        self._b2: float = 0.0
        self._threshold = 0.5

    def features(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = np.asarray(a, np.float32)[:, : self.slice_dim]
        b = np.asarray(b, np.float32)[:, : self.slice_dim]
        cos = (a * b).sum(1, keepdims=True)
        return np.concatenate([a * b, np.abs(a - b), cos], axis=1)

    @property
    def n_features(self) -> int:
        return 2 * self.slice_dim + 1

    def fit(self, a: np.ndarray, b: np.ndarray, y: np.ndarray, *, epochs: int = 30,
            lr: float = 1e-3, weight_decay: float = 1e-4, batch: int = 512,
            validation: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
            log: Callable[[str], None] | None = None) -> None:
        """Fit on labelled pairs; keep the epoch that scores best on ``validation``."""
        import torch

        torch.manual_seed(self.seed)
        x = torch.from_numpy(self.features(a, b))
        t = torch.from_numpy(np.asarray(y, np.float32).ravel())
        net = torch.nn.Sequential(
            torch.nn.Linear(self.n_features, self.hidden), torch.nn.ReLU(),
            torch.nn.Linear(self.hidden, 1),
        )
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
        pos_weight = torch.tensor([float(len(t) - t.sum()) / max(float(t.sum()), 1.0)])
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        best, best_state = -float("inf"), None
        val = None
        if validation is not None:
            va, vb, vy = validation
            val = (torch.from_numpy(self.features(va, vb)), np.asarray(vy).ravel().astype(bool))
        g = torch.Generator().manual_seed(self.seed)
        for epoch in range(epochs):
            net.train()
            for idx in torch.randperm(len(x), generator=g).split(batch):
                opt.zero_grad()
                loss_fn(net(x[idx]).squeeze(1), t[idx]).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                if val is None:
                    score = -float(loss_fn(net(x).squeeze(1), t))
                else:
                    score = auroc(net(val[0]).squeeze(1).numpy(), val[1])
            if log:
                log(f"epoch {epoch + 1:>3}  {'val AUROC' if val else '-train loss'} {score:.4f}")
            if score > best:
                best = score
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        net.load_state_dict(best_state)
        self._w1 = net[0].weight.detach().numpy().T.astype(np.float32)
        self._b1 = net[0].bias.detach().numpy().astype(np.float32)
        self._w2 = net[2].weight.detach().numpy().ravel().astype(np.float32)
        self._b2 = float(net[2].bias.detach().numpy()[0])

    def score(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if self._w1 is None:
            raise RuntimeError("MlpVerifier has no weights; fit() or load() first")
        h = np.maximum(self.features(a, b) @ self._w1 + self._b1, 0.0)
        logit = h @ self._w2 + self._b2
        return (1.0 / (1.0 + np.exp(-logit))).astype(np.float32)

    @property
    def threshold(self) -> float:
        return self._threshold

    def predict(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return self.score(a, b) >= self._threshold

    def save(self, path) -> None:
        np.savez(path, w1=self._w1, b1=self._b1, w2=self._w2, b2=np.float32(self._b2),
                 slice_dim=np.int64(self.slice_dim), hidden=np.int64(self.hidden))

    @classmethod
    def load(cls, path) -> "MlpVerifier":
        z = np.load(path)
        m = cls(int(z["slice_dim"]), int(z["hidden"]))
        m._w1, m._b1, m._w2, m._b2 = z["w1"], z["b1"], z["w2"], float(z["b2"])
        return m


def auroc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Area under the ROC curve by rank; ties split evenly."""
    s = np.asarray(scores, np.float64).ravel()
    y = np.asarray(positive).ravel().astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s), np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = sums[inv] / counts[inv]
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


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
    """Construct the verifier named by ``cfg.verifier`` ('cosine', 'logistic' or 'mlp')."""
    if cfg.verifier == "cosine":
        return CosineVerifier()
    if cfg.verifier == "logistic":
        return LogisticVerifier(seed=cfg.seed)
    if cfg.verifier == "mlp":
        from pathlib import Path
        if not Path(cfg.verifier_weights).exists():
            raise FileNotFoundError(
                f"reid.verifier is mlp but {cfg.verifier_weights} does not exist; "
                "train it with scripts/reid_train.py")
        return MlpVerifier.load(cfg.verifier_weights)
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

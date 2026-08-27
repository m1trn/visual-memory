import numpy as np
import pytest

from vision_memory.config import ReidConfig
from vision_memory.reid import (
    CosineVerifier,
    LogisticVerifier,
    build_verifier,
    mine_pairs,
    pair_features,
    split_by_group,
)

DIM = 16


def cfg(min_pair_gap: int = 2, verifier: str = "cosine", test_fraction: float = 0.3) -> ReidConfig:
    return ReidConfig(
        verifier=verifier, min_pair_gap=min_pair_gap, max_pairs_per_track=200, observation_quantile=0.75, max_false_merge_rate=0.02, claim_margin=0.08, swap_margin=0.08, min_new_identity_confidence=0.5, merge_margin=0.05, reconsider_every=15, continuity_bonus=0.15, spatial_scale=2.0, temporal_scale=2.0,
        test_fraction=test_fraction, seed=0
    )


def unit(n: int, dim: int = DIM, seed: int = 0) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def pairs_at_cosine(cosines: np.ndarray, seed: int, dim: int = DIM) -> tuple[np.ndarray, np.ndarray]:
    """Unit pairs whose cosine is exactly each requested value."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((len(cosines), dim)).astype(np.float32)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    u = rng.standard_normal((len(cosines), dim)).astype(np.float32)
    u -= (np.einsum("ij,ij->i", u, a))[:, None] * a  # orthogonal component
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    c = np.asarray(cosines, dtype=np.float32)[:, None]
    b = c * a + np.sqrt(1.0 - c**2) * u
    return a, b.astype(np.float32)


def make_tracks(n_tracks: int = 4, n_obs: int = 6, seed: int = 0) -> dict[int, np.ndarray]:
    """Tracks as tight clusters, each track's own observations drifting slowly."""
    rng = np.random.default_rng(seed)
    centers = unit(n_tracks, seed=seed + 100)
    tracks = {}
    for t in range(n_tracks):
        v = centers[t][None, :] + 0.15 * rng.standard_normal((n_obs, DIM)).astype(np.float32)
        tracks[t + 1] = v / np.linalg.norm(v, axis=1, keepdims=True)
    return tracks


def locate(vec: np.ndarray, tracks: dict[int, np.ndarray]) -> set[int]:
    """Which track ids contain this exact observation."""
    return {t for t, v in tracks.items() if np.any(np.all(np.isclose(v, vec), axis=1))}


def test_positives_come_from_one_track_and_negatives_from_two() -> None:
    tracks = {1: list(unit(6, seed=1)), 2: list(unit(6, seed=2)), 3: list(unit(6, seed=3))}
    a, b, y, owners = mine_pairs(tracks, cfg(), np.random.default_rng(0))
    assert len(a) == len(b) == len(y) == len(owners)
    for i in range(len(y)):
        ta, tb = int(owners[i, 0]), int(owners[i, 1])
        if y[i] == 1:
            assert ta == tb  # a positive pair is one track with itself
        else:
            assert ta != tb  # a negative pair spans two tracks

def test_min_pair_gap_is_respected() -> None:
    # One track with distinguishable observations: e_0 .. e_7 in order.
    track = {1: np.eye(8, dtype=np.float32), 2: -np.eye(8, dtype=np.float32)}
    for gap in (1, 3, 5):
        a, b, y, _ = mine_pairs(track, cfg(min_pair_gap=gap), np.random.default_rng(0))
        pos = y == 1
        assert pos.sum() > 0
        idx_a = np.argmax(np.abs(a[pos]), axis=1)
        idx_b = np.argmax(np.abs(b[pos]), axis=1)
        assert np.all(np.abs(idx_b - idx_a) >= gap)


def test_classes_are_balanced() -> None:
    for seed in range(3):
        _, _, y, _ = mine_pairs(make_tracks(seed=seed), cfg(), np.random.default_rng(seed))
        assert int((y == 1).sum()) == int((y == 0).sum()) > 0


def test_no_track_contributes_to_both_sides_of_the_split() -> None:
    """The invariant that matters: a held-out track must be unseen in training.

    Splitting on pair identity is not enough, because a negative pair straddles
    two tracks and would carry whichever side it landed on into both.
    """
    tracks = {t: list(unit(6, seed=t)) for t in range(1, 7)}
    for seed in range(6):
        rng = np.random.default_rng(seed)
        _, _, y, owners = mine_pairs(tracks, cfg(), rng)
        train, test = split_by_group(y, owners, 0.3, rng)
        assert not (train & test).any()  # no pair counted twice
        train_tracks = set(np.unique(owners[train]).tolist())
        test_tracks = set(np.unique(owners[test]).tolist())
        assert train_tracks.isdisjoint(test_tracks)
        assert test_tracks  # the held-out side is not empty

def test_split_keeps_both_sides_populated_at_a_large_fraction() -> None:
    tracks = {t: list(unit(6, seed=t)) for t in range(1, 7)}
    rng = np.random.default_rng(0)
    _, _, y, owners = mine_pairs(tracks, cfg(test_fraction=0.9), rng)
    train, test = split_by_group(y, owners, 0.9, rng)
    assert train.sum() > 0 and test.sum() > 0
    assert set(np.unique(owners[train]).tolist()).isdisjoint(np.unique(owners[test]).tolist())

def test_mining_returns_empty_when_no_positive_pair_is_long_enough() -> None:
    a, b, y, group = mine_pairs(
        {1: unit(2, seed=1), 2: unit(2, seed=2)}, cfg(min_pair_gap=5), np.random.default_rng(0)
    )
    assert len(a) == len(b) == len(y) == len(group) == 0


def test_cosine_verifier_separates_an_easy_case() -> None:
    same_a, same_b = pairs_at_cosine(np.full(60, 0.95), seed=1)
    diff_a, diff_b = pairs_at_cosine(np.full(60, 0.10), seed=2)
    a = np.concatenate([same_a, diff_a])
    b = np.concatenate([same_b, diff_b])
    y = np.concatenate([np.ones(60, int), np.zeros(60, int)])

    v = CosineVerifier()
    v.fit(a, b, y)
    assert 0.10 < v.threshold < 0.95
    assert np.array_equal(v.predict(a, b), y.astype(bool))


def test_learned_threshold_tracks_the_true_boundary_not_the_configured_one() -> None:
    rng = np.random.default_rng(7)
    boundary = 0.7
    same_c = rng.uniform(boundary + 0.02, 0.99, size=200)
    diff_c = rng.uniform(0.05, boundary - 0.02, size=200)
    same_a, same_b = pairs_at_cosine(same_c, seed=11)
    diff_a, diff_b = pairs_at_cosine(diff_c, seed=12)
    a = np.concatenate([same_a, diff_a])
    b = np.concatenate([same_b, diff_b])
    y = np.concatenate([np.ones(200, int), np.zeros(200, int)])

    v = CosineVerifier()
    v.fit(a, b, y)
    assert v.threshold == pytest.approx(boundary, abs=0.05)
    assert abs(v.threshold - 0.80) > 0.05  # not the hand-set config value


def test_logistic_scores_are_monotonic_in_similarity() -> None:
    train_c = np.concatenate([np.linspace(0.75, 0.99, 120), np.linspace(0.05, 0.55, 120)])
    ta, tb = pairs_at_cosine(train_c, seed=21)
    y = (train_c > 0.65).astype(int)
    v = LogisticVerifier(seed=0)
    v.fit(ta, tb, y)

    probe_c = np.linspace(0.05, 0.99, 40)
    pa, pb = pairs_at_cosine(probe_c, seed=22)
    scores = v.score(pa, pb)
    # Rank correlation with cosine: strictly increasing up to sampling noise.
    rank = np.argsort(np.argsort(scores)).astype(float)
    truth = np.arange(len(probe_c), dtype=float)
    corr = np.corrcoef(rank, truth)[0, 1]
    assert corr > 0.95


def test_predict_agrees_with_score_and_threshold() -> None:
    tracks = make_tracks(n_tracks=5, n_obs=8, seed=3)
    a, b, y, _ = mine_pairs(tracks, cfg(), np.random.default_rng(0))
    for v in (CosineVerifier(), LogisticVerifier(seed=0)):
        v.fit(a, b, y)
        assert np.array_equal(v.predict(a, b), v.score(a, b) >= v.threshold)


def test_pair_features_are_symmetric_in_argument_order() -> None:
    a, b = pairs_at_cosine(np.linspace(0.1, 0.9, 10), seed=31)
    assert np.allclose(pair_features(a, b), pair_features(b, a), atol=1e-6)


def test_build_verifier_dispatches_and_rejects_unknown() -> None:
    assert isinstance(build_verifier(cfg(verifier="cosine")), CosineVerifier)
    assert isinstance(build_verifier(cfg(verifier="logistic")), LogisticVerifier)
    with pytest.raises(ValueError):
        build_verifier(cfg(verifier="nope"))


def test_config_section_loads() -> None:
    from vision_memory.config import load_reid_config

    c = load_reid_config()
    assert c.verifier in {"cosine", "logistic"}
    assert c.min_pair_gap >= 1
    assert 0.0 <= c.test_fraction < 1.0


def test_cross_class_negatives_are_dropped_whatever_the_scorer() -> None:
    """Round-2 audit: the s != 0 filter only worked on the cosine scale."""
    from vision_memory.reid import calibrate_identity_threshold
    rng = np.random.default_rng(0)
    d = 8

    def unit(v):
        v = np.asarray(v, np.float32); return v / np.linalg.norm(v)
    # Two 'people' in the first slice, one 'car' in the disjoint second slice, all co-alive.
    a = [unit(rng.normal(size=d) * [1, 1, 1, 1, 0, 0, 0, 0]) for _ in range(6)]
    b = [unit(rng.normal(size=d) * [1, 1, 1, 1, 0, 0, 0, 0]) for _ in range(6)]
    c = [unit(rng.normal(size=d) * [0, 0, 0, 0, 1, 1, 1, 1]) for _ in range(6)]
    tracks = {1: a, 2: b, 3: c}
    frames = {1: list(range(6)), 2: list(range(6)), 3: list(range(6))}

    seen = []
    def logistic_like(q, e, quant):
        # Maps cosine 0 to 0.5: a scorer on another scale.
        from vision_memory.reid import identity_score
        s = 0.5 + 0.5 * identity_score(q, e, quant)
        seen.append(s); return s

    calibrate_identity_threshold(tracks, frames, 0.9, 5, 0.0, score=logistic_like)
    # Person-vs-car pairs (raw cosine exactly 0 -> 0.5 here) must never have been scored as negatives.
    negatives = [s for s in seen]
    assert all(abs(s - 0.5) > 1e-6 for s in negatives), negatives

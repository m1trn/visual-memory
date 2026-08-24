import numpy as np

from vision_memory.metrics import auroc


def test_perfect_separation() -> None:
    scores = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)
    labels = np.array([0, 0, 1, 1])
    assert auroc(scores, labels) == 1.0


def test_reversed_separation() -> None:
    scores = np.array([0.9, 0.8, 0.2, 0.1], dtype=np.float32)
    labels = np.array([0, 0, 1, 1])
    assert auroc(scores, labels) == 0.0


def test_random_ish_is_near_half() -> None:
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=2000)
    scores = rng.standard_normal(2000)  # uncorrelated with labels
    assert abs(auroc(scores, labels) - 0.5) < 0.05


def test_ties_count_as_half_win() -> None:
    # One positive and one negative share the same score -> that pair
    # contributes 0.5 instead of 0 or 1.
    scores = np.array([0.5, 0.5], dtype=np.float32)
    labels = np.array([0, 1])
    assert auroc(scores, labels) == 0.5


def test_all_same_class_returns_chance() -> None:
    scores = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    labels = np.array([1, 1, 1])
    assert auroc(scores, labels) == 0.5

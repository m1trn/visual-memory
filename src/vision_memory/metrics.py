"""Evaluation metrics for anomaly detection.

Numpy-only; no sklearn dependency.
"""

from __future__ import annotations

import numpy as np


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve via the Mann-Whitney U rank statistic.

    ``scores`` are anomaly scores (higher = more anomalous), ``labels`` are
    binary with 1 = anomaly (positive class), 0 = normal. Equivalent to the
    probability that a randomly chosen positive scores higher than a
    randomly chosen negative; ties are handled with average ranks and count
    as half a win. Returns 0.5 (chance) if either class is empty.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel().astype(bool)
    if scores.shape != labels.shape:
        raise ValueError(f"shape mismatch: scores {scores.shape} vs labels {labels.shape}")

    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]

    # Average-rank ties: assign the mean of the tied block's 1-based rank range.
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1

    sum_ranks_pos = ranks[labels].sum()
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))

"""Anomaly scoring over L2-normalized embeddings.

Every detector answers the same question: "how far is this embedding from
the distribution of normal embeddings I was fit on?" Higher score = more
anomalous. All detectors share the ``AnomalyDetector`` protocol so callers
can swap methods via config without touching call sites.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from vision_memory.config import AnomalyConfig


class AnomalyDetector(Protocol):
    """Fit on normal embeddings, then score new ones."""

    def fit(self, normal: np.ndarray) -> None:
        """Fit the detector on normal embeddings, shape ``(N, dim)``."""
        ...

    def score(self, x: np.ndarray) -> np.ndarray:
        """Score embeddings, shape ``(M, dim)`` -> ``(M,)`` float32, higher = more anomalous."""
        ...


class KNNDetector:
    """Mean cosine distance to the ``k`` nearest normal embeddings.

    Plain numpy: with N normals in the low hundreds and dim=384, a brute
    force ``(M, N)`` similarity matrix is cheap and avoids building a FAISS
    index for what is usually a small, static reference set.
    """

    def __init__(self, k: int) -> None:
        self.k = k
        self._normal: np.ndarray | None = None

    def fit(self, normal: np.ndarray) -> None:
        """Store the normal embeddings to search against."""
        normal = np.ascontiguousarray(normal, dtype=np.float32)
        if len(normal) == 0:
            raise ValueError("KNNDetector.fit needs at least one normal embedding")
        self._normal = normal

    def score(self, x: np.ndarray) -> np.ndarray:
        """Mean cosine distance to the k nearest fitted normals."""
        if self._normal is None:
            raise RuntimeError("KNNDetector.score called before fit")
        x = np.ascontiguousarray(x, dtype=np.float32)
        k = min(self.k, len(self._normal))
        sims = x @ self._normal.T  # (M, N) cosine similarity (unit vectors)
        top_sims = np.sort(sims, axis=1)[:, -k:]
        dist = 1.0 - top_sims
        return dist.mean(axis=1).astype(np.float32, copy=False)


class MahalanobisDetector:
    """Mahalanobis distance to the fitted normal distribution.

    N normals is typically far less than dim (384), which makes the sample
    covariance singular. Shrinkage toward a diagonal (mean-variance) matrix
    regularizes it: ``Sigma = (1 - s) * cov + s * diag(mean variance) * I``.
    """

    def __init__(self, shrinkage: float) -> None:
        self.shrinkage = shrinkage
        self._mu: np.ndarray | None = None
        self._chol: np.ndarray | None = None

    def fit(self, normal: np.ndarray) -> None:
        """Fit mean and Cholesky factor of the shrunk covariance."""
        normal = np.ascontiguousarray(normal, dtype=np.float64)
        if len(normal) < 2:
            raise ValueError("MahalanobisDetector.fit needs at least two normal embeddings")
        self._mu = normal.mean(axis=0)
        centered = normal - self._mu
        cov = (centered.T @ centered) / max(len(normal) - 1, 1)
        mean_var = float(np.mean(np.diag(cov)))
        if mean_var == 0.0:
            raise ValueError("normal embeddings are all identical; covariance is zero")
        dim = cov.shape[0]
        shrunk = (1.0 - self.shrinkage) * cov + self.shrinkage * mean_var * np.eye(dim)
        # Cholesky (not pinv): fails loudly if Sigma is singular, and solve is O(d^2) per vector.
        self._chol = np.linalg.cholesky(shrunk)

    def score(self, x: np.ndarray) -> np.ndarray:
        """Mahalanobis distance sqrt((x - mu)^T Sigma^-1 (x - mu))."""
        if self._mu is None or self._chol is None:
            raise RuntimeError("MahalanobisDetector.score called before fit")
        x = np.ascontiguousarray(x, dtype=np.float64)
        # Sigma = L L^T  =>  d^2 = ||L^-1 (x - mu)||^2
        z = np.linalg.solve(self._chol, (x - self._mu).T)
        return np.sqrt((z * z).sum(axis=0)).astype(np.float32, copy=False)


def build_detector(cfg: AnomalyConfig) -> AnomalyDetector:
    """Construct the detector named by ``cfg.method`` ('knn' or 'mahalanobis')."""
    if cfg.method == "knn":
        return KNNDetector(k=cfg.k)
    if cfg.method == "mahalanobis":
        return MahalanobisDetector(shrinkage=cfg.shrinkage)
    raise ValueError(f"unknown anomaly method: {cfg.method!r}")

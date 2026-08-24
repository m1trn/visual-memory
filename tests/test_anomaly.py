import numpy as np
import pytest

from vision_memory.anomaly import (IsolationForestDetector, KNNDetector, MahalanobisDetector,
                                   OneClassSVMDetector, build_detector)
from vision_memory.config import AnomalyConfig


def unit(n: int, d: int, seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def clustered(n: int, d: int, seed: int, spread: float = 0.05) -> np.ndarray:
    """Unit vectors tightly clustered around a fixed direction."""
    rng = np.random.default_rng(seed)
    base = np.zeros(d, dtype=np.float32)
    base[0] = 1.0
    v = base[None, :] + spread * rng.standard_normal((n, d)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.fixture
def data():
    dim = 32
    normals = clustered(100, dim, seed=0)
    normal_test = clustered(10, dim, seed=1)
    outliers = unit(10, dim, seed=2)  # random directions, far from the cluster
    return normals, normal_test, outliers


def test_knn_scores_outliers_higher(data) -> None:
    normals, normal_test, outliers = data
    det = KNNDetector(k=5)
    det.fit(normals)
    normal_scores = det.score(normal_test)
    outlier_scores = det.score(outliers)
    assert normal_scores.shape == (10,)
    assert normal_scores.dtype == np.float32
    assert outlier_scores.mean() > normal_scores.mean()


def test_mahalanobis_scores_outliers_higher(data) -> None:
    normals, normal_test, outliers = data
    det = MahalanobisDetector(shrinkage=0.1)
    det.fit(normals)
    normal_scores = det.score(normal_test)
    outlier_scores = det.score(outliers)
    assert normal_scores.shape == (10,)
    assert normal_scores.dtype == np.float32
    assert outlier_scores.mean() > normal_scores.mean()


def test_mahalanobis_handles_n_less_than_dim() -> None:
    dim = 384
    normals = clustered(20, dim, seed=0)  # N << dim, singular sample covariance
    det = MahalanobisDetector(shrinkage=0.1)
    det.fit(normals)
    scores = det.score(clustered(5, dim, seed=1))
    assert scores.shape == (5,)
    assert np.all(np.isfinite(scores))


def test_build_detector_knn_and_mahalanobis() -> None:
    knn = build_detector(AnomalyConfig(method="knn", k=5, shrinkage=0.1, n_estimators=50, nu=0.1, seed=0))
    assert isinstance(knn, KNNDetector)
    maha = build_detector(AnomalyConfig(method="mahalanobis", k=5, shrinkage=0.1, n_estimators=50, nu=0.1, seed=0))
    assert isinstance(maha, MahalanobisDetector)


def test_build_detector_rejects_unknown_method() -> None:
    with pytest.raises(ValueError):
        build_detector(AnomalyConfig(method="bogus", k=5, shrinkage=0.1, n_estimators=50, nu=0.1, seed=0))


def test_degenerate_fit_sets_raise() -> None:
    v = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    with pytest.raises(ValueError):
        KNNDetector(k=1).fit(v[:0])
    with pytest.raises(ValueError):
        MahalanobisDetector(shrinkage=0.1).fit(v)
    with pytest.raises(ValueError):
        MahalanobisDetector(shrinkage=0.1).fit(np.repeat(v, 3, axis=0))


@pytest.mark.parametrize("det", [IsolationForestDetector(n_estimators=50, seed=0), OneClassSVMDetector(nu=0.1)])
def test_sklearn_detectors_separate_outliers(det) -> None:
    rng = np.random.default_rng(0)
    normal = rng.normal(0, 0.05, (60, 8)) + np.array([1.0] + [0.0] * 7)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    outliers = rng.standard_normal((10, 8))
    outliers /= np.linalg.norm(outliers, axis=1, keepdims=True)
    det.fit(normal[:50])
    s_norm, s_out = det.score(normal[50:]), det.score(outliers)
    assert s_out.dtype == np.float32 and s_out.shape == (10,)
    # IsolationForest only splits inside the training range, so single far points can
    # look like boundary normals; require separation on average, not per point.
    assert s_out.mean() > s_norm.mean()

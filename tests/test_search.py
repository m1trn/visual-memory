import numpy as np
import pytest

from vision_memory.search import VectorIndex


def unit(n: int, d: int, seed: int = 0) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_exact_self_match_and_custom_ids() -> None:
    v = unit(20, 8)
    idx = VectorIndex(8)
    idx.add(v, ids=range(100, 120))
    scores, ids = idx.search(v[3], k=1)
    assert ids[0, 0] == 103 and scores[0, 0] == pytest.approx(1.0, abs=1e-5)


def test_pads_when_k_exceeds_size_and_empty() -> None:
    idx = VectorIndex(4)
    s, i = idx.search(unit(1, 4), k=3)
    assert (i == -1).all() and np.isinf(s).all()
    idx.add(unit(2, 4), ids=[7, 8])
    s, i = idx.search(unit(1, 4), k=3)
    assert i[0, 2] == -1


def test_remove_and_roundtrip(tmp_path) -> None:
    v = unit(5, 4)
    idx = VectorIndex(4)
    idx.add(v, ids=[1, 2, 3, 4, 5])
    assert idx.remove([2, 4]) == 2 and len(idx) == 3
    idx.save(tmp_path / "x.faiss")
    idx2 = VectorIndex.load(tmp_path / "x.faiss")
    assert len(idx2) == 3 and idx2.dim == 4
    assert np.allclose(idx2.get(3), v[2])
    with pytest.raises(ValueError):
        idx2.add(unit(1, 5), ids=[9])


def test_a_short_result_is_padded_with_minus_inf_like_the_empty_case() -> None:
    from vision_memory.search import VectorIndex
    idx = VectorIndex(4)
    idx.add(np.eye(4, dtype=np.float32)[:2], [10, 11])
    scores, ids = idx.search(np.eye(4, dtype=np.float32)[0], k=5)
    assert list(ids[0][:2]) == [10, 11] and (ids[0][2:] == -1).all()
    assert np.isneginf(scores[0][2:]).all()

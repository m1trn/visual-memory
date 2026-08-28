"""The trained projection: sits after the frozen embedder, shrinks the vector, keeps it unit."""

from __future__ import annotations

import numpy as np
import pytest

from vision_memory.appearance import ProjectedEmbedder


class FakeBase:
    dim = 6

    def encode_batch(self, crops):
        rng = np.random.default_rng(len(crops))
        v = rng.normal(size=(len(crops), 6)).astype(np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_projection_changes_dim_and_stays_unit(tmp_path) -> None:
    w = np.random.default_rng(0).normal(size=(6, 3)).astype(np.float32)
    np.savez(tmp_path / "p.npz", w=w)
    emb = ProjectedEmbedder.load(FakeBase(), str(tmp_path / "p.npz"))
    assert emb.dim == 3
    z = emb.encode_batch([None] * 4)
    assert z.shape == (4, 3)
    assert np.allclose(np.linalg.norm(z, axis=1), 1.0, atol=1e-5)


def test_projection_rejects_a_mismatched_matrix() -> None:
    with pytest.raises(ValueError):
        ProjectedEmbedder(FakeBase(), np.zeros((5, 3), np.float32))


def test_identity_projection_is_a_no_op() -> None:
    emb = ProjectedEmbedder(FakeBase(), np.eye(6, dtype=np.float32))
    assert np.allclose(emb.encode_batch([None] * 3), FakeBase().encode_batch([None] * 3), atol=1e-6)

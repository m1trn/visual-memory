"""The trained pair verifier: learns, scores through the protocol, survives a save/load."""

from __future__ import annotations

import numpy as np

from vision_memory.reid import MlpVerifier, auroc, identity_score_with

DIM = 16


def _pairs(rng, n=600):
    """Same-person pairs share a base direction; different-person pairs do not."""
    bases = rng.normal(size=(12, DIM)).astype(np.float32)
    bases /= np.linalg.norm(bases, axis=1, keepdims=True)
    a, b, y = [], [], []
    for _ in range(n):
        i = rng.integers(12)
        j = i if rng.random() < 0.5 else (i + rng.integers(1, 12)) % 12
        va = bases[i] + rng.normal(scale=0.3, size=DIM); vb = bases[j] + rng.normal(scale=0.3, size=DIM)
        a.append(va / np.linalg.norm(va)); b.append(vb / np.linalg.norm(vb)); y.append(int(i == j))
    return np.stack(a).astype(np.float32), np.stack(b).astype(np.float32), np.asarray(y)


def test_auroc_is_rank_based() -> None:
    assert auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert auroc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0
    assert abs(auroc([0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0]) - 0.5) < 1e-9


def test_mlp_learns_pairs_and_round_trips(tmp_path) -> None:
    rng = np.random.default_rng(0)
    a, b, y = _pairs(rng)
    va, vb, vy = _pairs(rng, 300)
    m = MlpVerifier(DIM, hidden=32, seed=0)
    m.fit(a, b, y, epochs=15, validation=(va, vb, vy))
    held = auroc(m.score(va, vb), vy.astype(bool))
    cosine = auroc((va * vb).sum(1), vy.astype(bool))
    assert held > 0.8 and held >= cosine - 0.02, (held, cosine)
    # Through the Verifier protocol used by the binder.
    assert m.predict(va[:5], vb[:5]).dtype == bool
    assert 0.0 <= m.threshold <= 1.0
    s = identity_score_with(m, va[:3], vb[:4], 0.9)
    assert 0.0 <= s <= 1.0

    path = tmp_path / "w.npz"
    m.save(path)
    back = MlpVerifier.load(path)
    assert np.allclose(back.score(va, vb), m.score(va, vb), atol=1e-6)


def test_mlp_only_reads_the_person_slice() -> None:
    """Routed vectors carry a zero general slice for people; it must not matter."""
    rng = np.random.default_rng(1)
    a, b, y = _pairs(rng, 200)
    m = MlpVerifier(DIM, hidden=16, seed=0)
    m.fit(a, b, y, epochs=5)
    padded_a = np.concatenate([a, rng.normal(size=(len(a), 7)).astype(np.float32)], axis=1)
    padded_b = np.concatenate([b, rng.normal(size=(len(b), 7)).astype(np.float32)], axis=1)
    assert np.allclose(m.score(padded_a, padded_b), m.score(a, b))

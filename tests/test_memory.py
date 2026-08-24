import numpy as np
import pytest

from vision_memory.config import MemoryConfig
from vision_memory.memory import VisualMemory, select_diverse

DIM = 8


def cfg(tmp_path, k: int = 5) -> MemoryConfig:
    return MemoryConfig(
        db_path=str(tmp_path / "memory.db"),
        index_path=str(tmp_path / "memory.faiss"),
        exemplars_per_identity=k,
        reid_threshold=0.8,
    )


def unit(n: int, d: int = DIM, seed: int = 0) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def cluster(center: np.ndarray, n: int, seed: int, spread: float = 0.01) -> np.ndarray:
    noise = np.random.default_rng(seed).standard_normal((n, len(center))).astype(np.float32)
    v = center[None, :] + spread * noise
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_roundtrip_survives_close_and_reopen(tmp_path) -> None:
    c = cfg(tmp_path)
    a, b = unit(4, seed=1), unit(3, seed=2)
    with VisualMemory(c, DIM) as mem:
        id_a = mem.remember("person", a, first_seen=1.0, last_seen=2.0, appearances=4)
        id_b = mem.remember("bag", b, first_seen=3.0, last_seen=4.0, appearances=3)
        proto_a = mem.get(id_a).prototype.copy()
        vecs_a = mem.exemplar_vectors(id_a).copy()

    with VisualMemory(c, DIM) as mem2:
        assert len(mem2) == 2
        again = mem2.get(id_a)
        assert again.label == "person" and again.appearances == 4
        assert again.first_seen == 1.0 and again.last_seen == 2.0
        assert np.allclose(again.prototype, proto_a)
        assert np.allclose(mem2.exemplar_vectors(id_a), vecs_a)
        assert {i.id for i in mem2.all_identities()} == {id_a, id_b}
        assert mem2.search(a[0], k=1)[0][0] == id_a


def test_search_dedupes_exemplars_of_one_identity(tmp_path) -> None:
    center = unit(1, seed=3)[0]
    with VisualMemory(cfg(tmp_path), DIM) as mem:
        id_a = mem.remember("a", cluster(center, 5, seed=4), 0.0, 1.0, 5)
        assert len(mem.exemplar_vectors(id_a)) == 5
        hits = mem.search(center, k=5)
        assert [h[0] for h in hits] == [id_a]
        assert hits[0][1] == pytest.approx(1.0, abs=0.05)


def test_search_picks_the_right_identity(tmp_path) -> None:
    e = np.eye(DIM, dtype=np.float32)
    with VisualMemory(cfg(tmp_path), DIM) as mem:
        id_a = mem.remember("a", cluster(e[0], 4, seed=5), 0.0, 1.0, 4)
        id_b = mem.remember("b", cluster(e[3], 4, seed=6), 0.0, 1.0, 4)
        assert mem.search(e[3], k=2)[0][0] == id_b
        assert mem.search(e[0], k=2)[0][0] == id_a
        ranked = mem.search(e[3], k=2)
        assert [r[0] for r in ranked] == [id_b, id_a]
        assert ranked[0][1] > ranked[1][1]


def test_merge_updates_metadata_and_prototype(tmp_path) -> None:
    e = np.eye(DIM, dtype=np.float32)
    with VisualMemory(cfg(tmp_path), DIM) as mem:
        id_a = mem.remember("a", e[0][None, :], first_seen=1.0, last_seen=2.0, appearances=1)
        before = mem.get(id_a).prototype.copy()
        mem.remember("a", e[1][None, :], first_seen=5.0, last_seen=9.0, appearances=3, identity_id=id_a)
        after = mem.get(id_a)
        assert len(mem) == 1
        assert after.appearances == 4
        assert after.first_seen == 1.0 and after.last_seen == 9.0
        assert not np.allclose(after.prototype, before)
        assert float(np.linalg.norm(after.prototype)) == pytest.approx(1.0, abs=1e-5)
        # 1*e0 + 3*e1, normalized -> weights 0.25 / 0.75 before normalization
        assert after.prototype[1] > after.prototype[0]
        expected = np.array([1.0, 3.0] + [0.0] * (DIM - 2), dtype=np.float32)
        assert np.allclose(after.prototype, expected / np.linalg.norm(expected), atol=1e-6)


def test_exemplars_never_exceed_cap(tmp_path) -> None:
    c = cfg(tmp_path, k=3)
    center = unit(1, seed=7)[0]
    with VisualMemory(c, DIM) as mem:
        id_a = mem.remember("a", cluster(center, 10, seed=8), 0.0, 1.0, 10)
        assert len(mem.exemplar_vectors(id_a)) == 3
        for step in range(4):
            mem.remember("a", cluster(center, 6, seed=20 + step), 1.0, 2.0, 6, identity_id=id_a)
            assert len(mem.exemplar_vectors(id_a)) == 3
        assert len(mem._index) == 3


def test_select_diverse_spans_clusters() -> None:
    e = np.eye(16, dtype=np.float32)
    groups = [cluster(e[0], 6, seed=11), cluster(e[5], 6, seed=12), cluster(e[9], 6, seed=13)]
    vectors = np.concatenate(groups)
    picked = select_diverse(vectors, 3)
    assert len(picked) == 3
    assert {int(i) // 6 for i in picked} == {0, 1, 2}


def test_select_diverse_returns_all_when_fewer_than_k() -> None:
    v = unit(3, seed=14)
    assert np.array_equal(select_diverse(v, 5), np.arange(3))


def test_empty_memory_searches_cleanly(tmp_path) -> None:
    with VisualMemory(cfg(tmp_path), DIM) as mem:
        assert len(mem) == 0
        assert mem.search(unit(1, seed=15)[0], k=5) == []
        assert mem.all_identities() == []
        assert mem.get(1) is None


def _unit(n: int, dim: int, seed: int = 0) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_stale_index_beside_fresh_database_is_reconciled(tmp_path) -> None:
    """A leftover index must not collide with exemplar ids that restart at 1."""
    cfg = MemoryConfig(db_path=str(tmp_path / "m.db"), index_path=str(tmp_path / "m.faiss"),
                       exemplars_per_identity=5, reid_threshold=0.8)
    with VisualMemory(cfg, 8) as mem:
        mem.remember("thing", list(_unit(4, 8)), 0.0, 1.0, 4)
    (tmp_path / "m.db").unlink()  # index survives, database does not

    with VisualMemory(cfg, 8) as mem:
        assert len(mem) == 0
        new_id = mem.remember("other", list(_unit(3, 8, seed=1)), 0.0, 1.0, 3)
        # Vectors must be the ones just stored, not the stale index's leftovers.
        assert len(mem.exemplar_vectors(new_id)) == 3


def test_orphaned_exemplar_rows_are_dropped_on_open(tmp_path) -> None:
    """A database written after the index it points at must still open cleanly."""
    cfg = MemoryConfig(db_path=str(tmp_path / "m.db"), index_path=str(tmp_path / "m.faiss"),
                       exemplars_per_identity=5, reid_threshold=0.8)
    with VisualMemory(cfg, 8) as mem:
        ident = mem.remember("thing", list(_unit(4, 8)), 0.0, 1.0, 4)
    # Simulate a crash after the sqlite commit but before the index was written.
    import sqlite3
    con = sqlite3.connect(cfg.db_path)
    con.execute("INSERT INTO exemplars (identity_id) VALUES (?)", (ident,))
    con.commit()
    con.close()

    with VisualMemory(cfg, 8) as mem:
        assert mem.exemplar_vectors(ident).shape[0] == 4  # the phantom row is gone
        assert mem.search(mem.get(ident).prototype, k=1)[0][0] == ident


def test_unnormalized_input_is_normalized_before_storage(tmp_path) -> None:
    cfg = MemoryConfig(db_path=str(tmp_path / "m.db"), index_path=str(tmp_path / "m.faiss"),
                       exemplars_per_identity=5, reid_threshold=0.8)
    with VisualMemory(cfg, 8) as mem:
        ident = mem.remember("thing", list(_unit(3, 8) * 7.0), 0.0, 1.0, 3)
        vecs = mem.exemplar_vectors(ident)
        assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)
        assert abs(np.linalg.norm(mem.get(ident).prototype) - 1.0) < 1e-5


def test_prototype_is_writeable(tmp_path) -> None:
    cfg = MemoryConfig(db_path=str(tmp_path / "m.db"), index_path=str(tmp_path / "m.faiss"),
                       exemplars_per_identity=5, reid_threshold=0.8)
    with VisualMemory(cfg, 8) as mem:
        ident = mem.remember("thing", list(_unit(2, 8)), 0.0, 1.0, 2)
        proto = mem.get(ident).prototype
        proto += 1.0  # must not raise
        assert proto.flags.writeable

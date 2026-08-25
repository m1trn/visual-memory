import numpy as np
import pytest

from vision_memory.config import MemoryConfig
from vision_memory.memory import VisualMemory

reidentifier = pytest.importorskip("vision_memory.reidentifier")
ReIdentifier = reidentifier.ReIdentifier
Resolution = reidentifier.Resolution

DIM = 8


class FakeVerifier:
    """Cosine verifier with a threshold fixed by the test, no fitting involved."""

    def __init__(self, threshold: float = 0.8) -> None:
        self._threshold = threshold

    def fit(self, a: np.ndarray, b: np.ndarray, y: np.ndarray) -> None:
        self._threshold = float(np.mean(y))

    def score(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.sum(a * b, axis=1).astype(np.float32)

    @property
    def threshold(self) -> float:
        return self._threshold

    def predict(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return self.score(a, b) >= self._threshold


def cfg(tmp_path, k: int = 5) -> MemoryConfig:
    return MemoryConfig(
        db_path=str(tmp_path / "memory.db"),
        index_path=str(tmp_path / "memory.faiss"),
        exemplars_per_identity=k,
        reid_threshold=0.8,
    )


def cluster(center: np.ndarray, n: int, seed: int, spread: float = 0.01) -> np.ndarray:
    noise = np.random.default_rng(seed).standard_normal((n, len(center))).astype(np.float32)
    v = center[None, :] + spread * noise
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_empty_memory_always_creates(tmp_path) -> None:
    e = np.eye(DIM, dtype=np.float32)
    with VisualMemory(cfg(tmp_path), DIM) as mem:
        r = ReIdentifier(mem, FakeVerifier(0.8)).resolve("person", cluster(e[0], 3, seed=1), 0.0, 1.0, 3)
        assert r.is_new is True
        assert r.score == float("-inf")
        assert mem.get(r.identity_id).label == "person"
        assert len(mem) == 1


def test_returning_object_binds_to_its_original_identity(tmp_path) -> None:
    e = np.eye(DIM, dtype=np.float32)
    with VisualMemory(cfg(tmp_path), DIM) as mem:
        rid = ReIdentifier(mem, FakeVerifier(0.8))
        first = rid.resolve("person", cluster(e[0], 4, seed=2), 0.0, 1.0, 4)
        again = rid.resolve("person", cluster(e[0], 4, seed=3), 5.0, 6.0, 4)
        assert again.is_new is False
        assert again.identity_id == first.identity_id
        assert again.score > 0.9
        assert len(mem) == 1
        merged = mem.get(first.identity_id)
        assert merged.appearances == 8
        assert merged.first_seen == 0.0 and merged.last_seen == 6.0


def test_different_object_with_same_label_creates_a_new_identity(tmp_path) -> None:
    e = np.eye(DIM, dtype=np.float32)
    with VisualMemory(cfg(tmp_path), DIM) as mem:
        rid = ReIdentifier(mem, FakeVerifier(0.8))
        first = rid.resolve("person", cluster(e[0], 4, seed=4), 0.0, 1.0, 4)
        other = rid.resolve("person", cluster(e[5], 4, seed=5), 2.0, 3.0, 4)
        assert other.is_new is True
        assert other.identity_id != first.identity_id
        assert len(mem) == 2


def test_label_mismatch_never_binds(tmp_path) -> None:
    """Identical embeddings must not merge a person into a car."""
    e = np.eye(DIM, dtype=np.float32)
    with VisualMemory(cfg(tmp_path), DIM) as mem:
        rid = ReIdentifier(mem, FakeVerifier(0.5))
        car = rid.resolve("car", cluster(e[0], 4, seed=6), 0.0, 1.0, 4)
        person = rid.resolve("person", cluster(e[0], 4, seed=7), 2.0, 3.0, 4)
        assert person.is_new is True
        assert person.identity_id != car.identity_id
        assert mem.get(car.identity_id).label == "car"
        assert mem.get(person.identity_id).label == "person"
        assert len(mem) == 2


def test_resolving_twice_in_a_row_does_not_duplicate(tmp_path) -> None:
    e = np.eye(DIM, dtype=np.float32)
    obs = cluster(e[2], 4, seed=8)
    with VisualMemory(cfg(tmp_path), DIM) as mem:
        rid = ReIdentifier(mem, FakeVerifier(0.8))
        first = rid.resolve("bag", obs, 0.0, 1.0, 4)
        second = rid.resolve("bag", obs, 1.0, 2.0, 4)
        third = rid.resolve("bag", obs, 2.0, 3.0, 4)
        assert second.identity_id == first.identity_id
        assert third.identity_id == first.identity_id
        assert [second.is_new, third.is_new] == [False, False]
        assert len(mem) == 1
        assert len(mem.all_identities()) == 1


def test_verifier_threshold_decides_not_the_raw_search_score(tmp_path) -> None:
    """The same pair binds or splits purely on where the learned boundary sits."""
    e = np.eye(DIM, dtype=np.float32)
    drifted = (e[0] + 0.6 * e[1]) / np.linalg.norm(e[0] + 0.6 * e[1])  # ~0.86 cosine

    with VisualMemory(cfg(tmp_path / "strict"), DIM) as mem:
        rid = ReIdentifier(mem, FakeVerifier(0.95))
        rid.resolve("person", cluster(e[0], 3, seed=9), 0.0, 1.0, 3)
        assert rid.resolve("person", cluster(drifted, 3, seed=10), 2.0, 3.0, 3).is_new is True
        assert len(mem) == 2

    with VisualMemory(cfg(tmp_path / "loose"), DIM) as mem:
        rid = ReIdentifier(mem, FakeVerifier(0.60))
        first = rid.resolve("person", cluster(e[0], 3, seed=9), 0.0, 1.0, 3)
        second = rid.resolve("person", cluster(drifted, 3, seed=10), 2.0, 3.0, 3)
        assert second.is_new is False and second.identity_id == first.identity_id
        assert len(mem) == 1


def test_matching_one_stored_view_is_enough(tmp_path) -> None:
    """Aggregation is max over exemplars: a multi-view identity stays findable."""
    e = np.eye(DIM, dtype=np.float32)
    views = np.concatenate([cluster(e[0], 3, seed=11), cluster(e[4], 3, seed=12)])
    with VisualMemory(cfg(tmp_path), DIM) as mem:
        rid = ReIdentifier(mem, FakeVerifier(0.9))
        first = rid.resolve("person", views, 0.0, 1.0, 6)
        back = rid.resolve("person", cluster(e[4], 3, seed=13), 5.0, 6.0, 3)
        assert back.is_new is False and back.identity_id == first.identity_id


def test_a_lone_lucky_frame_does_not_carry_the_track(tmp_path) -> None:
    """Aggregation is mean over observations: most of the track must agree."""
    e = np.eye(DIM, dtype=np.float32)
    with VisualMemory(cfg(tmp_path), DIM) as mem:
        rid = ReIdentifier(mem, FakeVerifier(0.9))
        rid.resolve("person", cluster(e[0], 3, seed=14), 0.0, 1.0, 3)
        mixed = np.concatenate([cluster(e[0], 1, seed=15), cluster(e[6], 3, seed=16)])
        assert rid.resolve("person", mixed, 2.0, 3.0, 4).is_new is True
        assert len(mem) == 2


def test_binding_survives_close_and_reopen(tmp_path) -> None:
    e = np.eye(DIM, dtype=np.float32)
    c = cfg(tmp_path)
    with VisualMemory(c, DIM) as mem:
        first = ReIdentifier(mem, FakeVerifier(0.8)).resolve("person", cluster(e[3], 4, seed=17), 0.0, 1.0, 4)
    with VisualMemory(c, DIM) as mem:
        again = ReIdentifier(mem, FakeVerifier(0.8)).resolve("person", cluster(e[3], 4, seed=18), 9.0, 10.0, 2)
        assert again.is_new is False and again.identity_id == first.identity_id
        assert mem.get(first.identity_id).appearances == 6


def test_rejects_empty_observations(tmp_path) -> None:
    with VisualMemory(cfg(tmp_path), DIM) as mem:
        with pytest.raises(ValueError):
            ReIdentifier(mem, FakeVerifier()).resolve("person", [], 0.0, 1.0, 0)


def test_an_identity_alive_at_the_same_time_is_never_bound(tmp_path) -> None:
    """Two things on screen together are provably not the same thing."""
    e = np.eye(DIM, dtype=np.float32)
    obs = cluster(e[3], 4, seed=11)
    with VisualMemory(cfg(tmp_path), DIM) as mem:
        rid = ReIdentifier(mem, FakeVerifier(0.0))  # accepts anything on appearance alone
        first = rid.resolve("bag", obs, 0.0, 10.0, 4)
        # Identical embeddings, but this track overlaps the stored one in time.
        overlapping = rid.resolve("bag", obs, 5.0, 15.0, 4)
        assert overlapping.is_new
        assert overlapping.identity_id != first.identity_id
        # The same appearance after the first has finished does bind.
        later = rid.resolve("bag", obs, 20.0, 25.0, 4)
        assert not later.is_new

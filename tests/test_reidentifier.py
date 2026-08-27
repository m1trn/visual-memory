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


def test_continuity_lets_a_weak_appearance_match_bind(tmp_path) -> None:
    """Reappearing where you vanished, moments later, is evidence in its own right."""
    e = np.eye(DIM, dtype=np.float32)
    stored = [e[4].copy() for _ in range(4)]
    # Same object, seen differently enough that appearance alone will not do it.
    returning = np.sqrt(0.80) * e[4] + np.sqrt(0.20) * e[5]
    returning = [(returning / np.linalg.norm(returning)).astype(np.float32) for _ in range(4)]
    box = np.array([100.0, 100.0, 140.0, 200.0])
    strict = 0.95  # above the ~0.894 these two score against each other

    with VisualMemory(cfg(tmp_path), DIM) as mem:
        rid = ReIdentifier(mem, FakeVerifier(strict))
        first = rid.resolve("bag", stored, 0.0, 5.0, 4, box=box)
        near = rid.resolve("bag", returning, 6.0, 7.0, 4, box=box + 5.0)
        assert near.identity_id == first.identity_id and not near.is_new

    with VisualMemory(cfg(tmp_path / "far"), DIM) as mem:
        rid = ReIdentifier(mem, FakeVerifier(strict))
        rid.resolve("bag", stored, 0.0, 5.0, 4, box=box)
        # Same appearance evidence, but far away and long after: nothing to lend.
        far = rid.resolve("bag", returning, 400.0, 401.0, 4, box=box + 900.0)
        assert far.is_new


def test_a_track_can_reclaim_an_earlier_record_it_first_missed(tmp_path) -> None:
    """The first binding rests on two observations; it must be allowed to improve."""
    e = np.eye(DIM, dtype=np.float32)
    earlier = [e[6].copy() for _ in range(4)]
    with VisualMemory(cfg(tmp_path), DIM) as mem:
        rid = ReIdentifier(mem, FakeVerifier(0.9))
        old = rid.resolve("bag", earlier, 0.0, 5.0, 4)

        # The same object returns, but its first two views are unlike anything
        # stored, so it is called new.
        weak = [(np.sqrt(0.5) * e[6] + np.sqrt(0.5) * e[7]).astype(np.float32)] * 2
        provisional = rid.resolve("bag", weak, 20.0, 21.0, 2)
        assert provisional.is_new and provisional.identity_id != old.identity_id
        assert len(mem) == 2

        # It keeps being watched, and now plainly matches the earlier record.
        revised = rid.reconsider(provisional.identity_id, "bag", earlier, 20.0, 30.0, appearances=6)
        assert revised is not None
        assert revised.identity_id == old.identity_id      # the older record wins
        assert len(mem) == 1                               # and the duplicate is gone
        merged = mem.get(old.identity_id)
        assert merged.first_seen == 0.0 and merged.last_seen == 30.0


def test_reconsider_leaves_a_sound_binding_alone(tmp_path) -> None:
    e = np.eye(DIM, dtype=np.float32)
    with VisualMemory(cfg(tmp_path), DIM) as mem:
        rid = ReIdentifier(mem, FakeVerifier(0.9))
        a = rid.resolve("bag", [e[6].copy() for _ in range(4)], 0.0, 5.0, 4)
        b = rid.resolve("bag", [e[7].copy() for _ in range(4)], 20.0, 25.0, 4)
        assert b.identity_id != a.identity_id
        # b looks nothing like a, so there is nothing to merge.
        assert rid.reconsider(b.identity_id, "bag", [e[7].copy() for _ in range(4)], 20.0, 30.0, appearances=4) is None
        assert len(mem) == 2


def test_a_live_identity_cannot_be_claimed_by_someone_on_screen_with_it(tmp_path) -> None:
    """The classic failure: one number ending up on two people at once.

    An identity's stored end-time must follow the object while it is still
    visible. If it stays frozen at creation, a track that starts later looks
    sequential rather than simultaneous, and binds to a person still on screen.
    """
    e = np.eye(DIM, dtype=np.float32)
    obs = cluster(e[5], 4, seed=21)
    with VisualMemory(cfg(tmp_path), DIM) as mem:
        rid = ReIdentifier(mem, FakeVerifier(0.0))  # appearance accepts anything
        first = rid.resolve("bag", obs, 0.0, 1.0, 4, box=np.array([0.0, 0.0, 40.0, 80.0]))
        # It stays on screen until t=20.
        for t in range(2, 21):
            rid.note_seen(first.identity_id, float(t), np.array([0.0, 0.0, 40.0, 80.0]))
        # A second object appears at t=10, while the first is still visible.
        second = rid.resolve("bag", obs, 10.0, 12.0, 4, box=np.array([0.0, 0.0, 40.0, 80.0]))
        assert second.is_new
        assert second.identity_id != first.identity_id


def test_reconsider_will_not_fuse_two_records_that_coexisted(tmp_path) -> None:
    """A merge joins two records, so those two must be compatible with each other.

    Checking only the track against the candidate is not enough: a track that
    arrives late need not overlap an old identity, but merging would still fuse
    that old identity with the one the track holds.
    """
    e = np.eye(DIM, dtype=np.float32)
    obs = cluster(e[1], 4, seed=31)
    with VisualMemory(cfg(tmp_path), DIM) as mem:
        rid = ReIdentifier(mem, FakeVerifier(0.0))  # appearance accepts anything
        early = rid.resolve("bag", obs, 0.0, 5.0, 4)
        overlapping = rid.resolve("bag", obs, 4.0, 27.0, 4)   # coexisted with `early`
        assert overlapping.is_new and len(mem) == 2

        # A late track holding `overlapping` must not drag it into `early`.
        assert rid.reconsider(overlapping.identity_id, "bag", obs, 26.0, 30.0, appearances=4) is None
        assert len(mem) == 2


def test_an_identity_someone_is_wearing_now_cannot_be_taken(tmp_path) -> None:
    """Interval arithmetic cannot settle this; the caller knows who is on screen."""
    e = np.eye(DIM, dtype=np.float32)
    obs = cluster(e[2], 4, seed=41)
    with VisualMemory(cfg(tmp_path), DIM) as mem:
        rid = ReIdentifier(mem, FakeVerifier(0.0))  # appearance accepts anything
        first = rid.resolve("bag", obs, 0.0, 2.1, 4)
        # Another object appears at the very instant the first was last seen.
        taken = rid.resolve("bag", obs, 2.1, 2.1, 4, unavailable={first.identity_id})
        assert taken.is_new and taken.identity_id != first.identity_id
        # Once the first is gone, the identity is available again.
        later = rid.resolve("bag", obs, 30.0, 31.0, 4, unavailable=set())
        assert not later.is_new


def test_an_identity_cannot_crowd_itself_out_of_the_shortlist(tmp_path) -> None:
    """``k`` counts identities, so exemplars of one must not consume the width.

    A decoy sits marginally closer to the query than the true match. Both belong
    on a shortlist of two, and choosing between them is the verifier's job; the
    index must not spend both slots on the decoy's five exemplars.
    """
    rng = np.random.default_rng(7)
    q = np.zeros(DIM, dtype=np.float32); q[0] = 1.0
    o = np.zeros(DIM, dtype=np.float32); o[1] = 1.0

    def near(weight: float, n: int) -> list[np.ndarray]:
        out = []
        for _ in range(n):
            v = weight * q + np.sqrt(1 - weight ** 2) * o
            v = v + rng.normal(scale=0.005, size=DIM).astype(np.float32)
            out.append((v / np.linalg.norm(v)).astype(np.float32))
        return out

    with VisualMemory(cfg(tmp_path), DIM) as mem:
        rid = ReIdentifier(mem, FakeVerifier(0.0), candidates=2)
        decoy = mem.remember("bag", near(0.99, 5), 0.0, 1.0, 5)
        wanted = mem.remember("bag", near(0.95, 5), 2.0, 3.0, 5)

        shortlist = rid._shortlist("bag", np.stack([q, q]))
        assert decoy in shortlist
        assert wanted in shortlist, "one identity consumed the whole shortlist"


def test_a_claim_cannot_take_an_identity_that_was_co_alive(tmp_path) -> None:
    """Two objects on screen at once are different objects, however alike.

    ``_best_candidate`` refuses a co-alive identity outright; the takeover path
    must apply the same fact, or a strong appearance match can steal an
    identity from an object that was provably visible at the same time.
    """
    e = np.eye(DIM, dtype=np.float32)
    obs = cluster(e[4], 4, seed=21)
    with VisualMemory(cfg(tmp_path), DIM) as mem:
        rid = ReIdentifier(mem, FakeVerifier(0.0))  # appearance accepts anything
        holder = rid.resolve("bag", obs, 0.0, 10.0, 4)
        # A second object, alive from 5.0 - overlapping the holder's lifetime -
        # scores perfectly against the stored record and offers a huge margin.
        taken = rid.resolve("bag", obs, 5.0, 6.0, 4,
                            unavailable={holder.identity_id},
                            held_by_others={holder.identity_id: 0.1})
        assert taken.is_new and taken.identity_id != holder.identity_id


def test_a_dead_track_releases_its_identity_so_the_person_can_return(tmp_path) -> None:
    """Only live tracks make an identity unavailable.

    A track that has died must let go of its number. Without that, every
    identity ever bound stays "in use" forever and a returning person is
    always called new - which is exactly what the demo did for a day.
    """
    from vision_memory.reidentifier import IdentityBinder
    from vision_memory.tracker import Track

    e = np.eye(DIM, dtype=np.float32)

    def track(tid: int, views: list[np.ndarray], tsu: int = 0) -> Track:
        return Track(id=tid, box=np.array([0.0, 0.0, 10.0, 20.0], dtype=np.float32),
                     score=0.9, class_id=0, label="person", hits=len(views),
                     time_since_update=tsu, state="active", exemplars=views)

    with VisualMemory(cfg(tmp_path), DIM) as mem:
        rid = ReIdentifier(mem, FakeVerifier(0.8))
        binder = IdentityBinder(rid, fps=10.0, fresh=3, reconsider_every=1000)

        first = track(1, cluster(e[2], 3, seed=31))
        binder.step([first], frame_idx=0)
        original = binder.identity_of(1)
        assert original is not None

        # The track died. Even before the caller forgets it, a track that is
        # no longer live must not hold its number against a returning person.
        again = track(2, cluster(e[2], 3, seed=32))
        events = binder.step([again], frame_idx=100)
        assert events.rebound == 1 and events.created == 0
        assert binder.identity_of(2) == original, "a returning person must get their number back"

        binder.forget(1)
        assert binder.identity_of(1) is None


def test_two_people_holding_each_others_numbers_swap_back_only_when_mutual(tmp_path) -> None:
    """Takeover never runs for already-bound tracks, so this is the only repair.

    Two live tracks wear each other's identities. Both fit the other's record
    better than their own, so they swap. A one-sided case must not swap: one
    confused frame on one side cannot be allowed to flip two people.
    """
    from vision_memory.reidentifier import IdentityBinder, Resolution
    from vision_memory.tracker import Track

    e = np.eye(DIM, dtype=np.float32)

    def track(tid: int, views) -> Track:
        views = [np.asarray(v, dtype=np.float32) for v in views]
        return Track(id=tid, box=np.array([0.0, 0.0, 10.0, 20.0], dtype=np.float32),
                     score=0.9, class_id=0, label="person", hits=len(views),
                     time_since_update=0, state="active", exemplars=views)

    with VisualMemory(cfg(tmp_path), DIM) as mem:
        rid = ReIdentifier(mem, FakeVerifier(0.5))
        alice = mem.remember("person", cluster(e[1], 4, seed=41), 0.0, 1.0, 4)
        bob = mem.remember("person", cluster(e[5], 4, seed=42), 0.0, 1.0, 4)
        binder = IdentityBinder(rid, fps=10.0, fresh=3, reconsider_every=1, swap_margin=0.1)

        # A crossing left them wearing each other's numbers.
        a = track(1, cluster(e[1], 3, seed=43))   # looks like alice
        b = track(2, cluster(e[5], 3, seed=44))   # looks like bob
        binder.bound[1] = Resolution(identity_id=bob, score=0.6, is_new=False)
        binder.bound[2] = Resolution(identity_id=alice, score=0.6, is_new=False)
        binder.first_frame.update({1: 0, 2: 0})

        events = binder.step([a, b], frame_idx=1)
        assert events.swapped == 1
        assert binder.identity_of(1) == alice and binder.identity_of(2) == bob

        # One-sided: track 2 now looks like NEITHER, so no swap however well
        # track 1 fits the other record.
        binder.bound[1] = Resolution(identity_id=bob, score=0.6, is_new=False)
        binder.bound[2] = Resolution(identity_id=alice, score=0.6, is_new=False)
        neither = track(2, cluster(e[7], 3, seed=45))
        events = binder.step([a, neither], frame_idx=2)
        assert events.swapped == 0
        assert binder.identity_of(1) == bob and binder.identity_of(2) == alice


def _live_track(tid, views, tsu=0, peak=0.9):
    from vision_memory.tracker import Track
    views = [np.asarray(v, dtype=np.float32) for v in views]
    return Track(id=tid, box=np.array([0.0, 0.0, 10.0, 20.0], dtype=np.float32),
                 score=peak, class_id=0, label="person", hits=len(views),
                 time_since_update=tsu, state="active", exemplars=views, peak_score=peak)


def test_a_ghost_whose_identity_was_taken_gets_no_new_number(tmp_path) -> None:
    """A coasting track that loses its identity to a fresh detection is the same object.

    The person was briefly hidden; the tracker coasted their box and then
    spawned a new track on the real detection. Re-id rightly gives the new
    track the identity. The coasting ghost must NOT then be re-identified as
    somebody new - that is a phantom box with a fresh number.
    """
    from vision_memory.reidentifier import IdentityBinder
    e = np.eye(DIM, dtype=np.float32)
    with VisualMemory(cfg(tmp_path), DIM) as mem:
        rid = ReIdentifier(mem, FakeVerifier(0.8))
        binder = IdentityBinder(rid, fps=10.0, fresh=3, reconsider_every=1000,
                                convincing_confidence=0.5)
        ghost = _live_track(1, cluster(e[2], 3, seed=51))
        binder.step([ghost], 0)
        ident = binder.identity_of(1)

        # Hidden for a while: the ghost coasts, the real detection spawns track 2.
        ghost = _live_track(1, cluster(e[2], 3, seed=51), tsu=5)
        fresh = _live_track(2, cluster(e[2], 3, seed=52))
        events = binder.step([ghost, fresh], 10)
        assert events.taken == 1 and binder.identity_of(2) == ident

        # Next frames: the ghost is still unseen and must stay unnumbered.
        events = binder.step([ghost, fresh], 11)
        assert events.created == 0 and binder.identity_of(1) is None
        assert len(mem) == 1

        # A weak second-pass match is not a sighting either.
        weak = _live_track(1, cluster(e[2], 3, seed=51), tsu=0, peak=0.18)
        events = binder.step([weak, fresh], 11)
        assert events.created == 0 and binder.identity_of(1) is None

        # If it IS seen again, it is a real track and may be identified.
        seen = _live_track(1, cluster(e[6], 3, seed=53), tsu=0)
        events = binder.step([seen, fresh], 12)
        assert events.created == 1 and binder.identity_of(1) not in (None, ident)


def test_a_track_never_seen_convincingly_cannot_create_an_identity(tmp_path) -> None:
    """A sign post the detector called a person at 0.36 must not become somebody."""
    from vision_memory.reidentifier import IdentityBinder
    e = np.eye(DIM, dtype=np.float32)
    with VisualMemory(cfg(tmp_path), DIM) as mem:
        rid = ReIdentifier(mem, FakeVerifier(0.8))
        binder = IdentityBinder(rid, fps=10.0, fresh=3, reconsider_every=1000,
                                min_new_identity_confidence=0.5)
        post = _live_track(1, cluster(e[4], 3, seed=61), peak=0.36)
        events = binder.step([post], 0)
        assert events.created == 0 and binder.identity_of(1) is None
        assert len(mem) == 0, "the withdrawn record must not linger in memory"

        # A convincing sighting later makes it eligible.
        post = _live_track(1, cluster(e[4], 3, seed=61), peak=0.7)
        events = binder.step([post], 1)
        assert events.created == 1 and len(mem) == 1
        ident = binder.identity_of(1)

        # Binding to an EXISTING identity never needed the sighting.
        weak = _live_track(2, cluster(e[4], 3, seed=62), peak=0.36)
        binder.forget(1)
        events = binder.step([weak], 50)
        assert events.rebound == 1 and binder.identity_of(2) == ident


def test_a_newcomer_cannot_take_a_number_from_a_holder_who_still_fits_it(tmp_path) -> None:
    """The holder's claim is his live fit to the record, not the score he bound at.

    A track that created an identity holds it at -inf, so a newcomer merely
    clearing the threshold could take a long-lived person's number the moment
    he was occluded. On the demo clip a man carrying a sign took #2 from the
    dark-jacketed man who had worn it for 13 hits. The challenger must beat
    the holder's actual fit by the claim margin.
    """
    from vision_memory.reidentifier import IdentityBinder
    e = np.eye(DIM, dtype=np.float32)
    with VisualMemory(cfg(tmp_path), DIM) as mem:
        rid = ReIdentifier(mem, FakeVerifier(0.6))
        binder = IdentityBinder(rid, fps=10.0, fresh=3, reconsider_every=1000)
        holder = _live_track(1, cluster(e[2], 4, seed=71))
        binder.step([holder], 0)
        ident = binder.identity_of(1)

        # A newcomer whose views resemble the record enough to clear 0.6, but
        # less than the holder himself does.
        near = [(np.sqrt(0.7) * e[2] + np.sqrt(0.3) * e[5]).astype(np.float32)] * 3
        newcomer = _live_track(2, near)
        events = binder.step([holder, newcomer], 1)
        assert events.taken == 0
        assert binder.identity_of(1) == ident and binder.identity_of(2) not in (None, ident)

        # But a holder who has drifted onto something else can be displaced by
        # someone who plainly fits the record better.
        binder.forget(2)
        drifted = _live_track(1, cluster(e[7], 4, seed=72))
        rightful = _live_track(3, cluster(e[2], 3, seed=73))
        events = binder.step([drifted, rightful], 20)
        assert events.taken == 1 and binder.identity_of(3) == ident

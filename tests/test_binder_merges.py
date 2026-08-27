"""Merges and displacements in the binder: the cases the third audit round found."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_reidentifier import DIM, FakeVerifier, _live_track, cfg, cluster  # noqa: E402

from vision_memory.memory import VisualMemory  # noqa: E402
from vision_memory.reidentifier import IdentityBinder, ReIdentifier  # noqa: E402


def test_a_merge_displaces_a_ghost_wearing_the_absorbed_record_too(tmp_path) -> None:
    """The mirror of the surviving-id case: the ghost's record is the one the merge deletes."""
    e = np.eye(DIM, dtype=np.float32)
    with VisualMemory(cfg(tmp_path), DIM) as mem:
        rid = ReIdentifier(mem, FakeVerifier(0.6))
        binder = IdentityBinder(rid, fps=10.0, fresh=3, reconsider_every=10,
                                convincing_confidence=0.5)
        # Same frame, one person seen twice: the keeper's record gets the SMALLER id.
        near = [(np.sqrt(0.92) * e[3] + np.sqrt(0.08) * e[5]).astype(np.float32)] * 3
        keeper = _live_track(1, near, box=(0, 0, 10, 20))
        dup = _live_track(2, [e[3].copy() for _ in range(3)], box=(300, 300, 310, 320))
        binder.step([keeper, dup], 0)
        k, g = binder.identity_of(1), binder.identity_of(2)
        assert k is not None and g is not None and k < g
        # The duplicate coasts; the keeper keeps being seen and, at the tick,
        # plainly matches the duplicate's record -> merge, keeping k, deleting g.
        ghost = _live_track(2, [e[3].copy() for _ in range(3)], tsu=10, box=(300, 300, 310, 320))
        keeper = _live_track(1, [e[3].copy() for _ in range(6)], box=(0, 0, 10, 20))
        events = binder.step([keeper, ghost], 10)
        assert events.reclaimed == 1 and binder.identity_of(1) == k
        assert mem.get(g) is None
        # The ghost must not be left wearing a deleted number.
        assert binder.identity_of(2) is None and 2 in binder._dormant
        res, _, _ = binder.forget(2)
        assert res is None


def test_a_displaced_track_does_not_recredit_hits_it_already_folded(tmp_path) -> None:
    """Per-track hits were counted into two identities after a displacement."""
    e = np.eye(DIM, dtype=np.float32)
    with VisualMemory(cfg(tmp_path), DIM) as mem:
        rid = ReIdentifier(mem, FakeVerifier(0.6))
        binder = IdentityBinder(rid, fps=10.0, fresh=3, reconsider_every=1000, recent_views=3)
        holder = _live_track(1, cluster(e[1], 4, seed=111))          # 4 hits folded into A
        binder.step([holder], 0)
        a = binder.identity_of(1)
        holder = _live_track(1, cluster(e[7], 4, seed=112))          # slid onto someone else
        holder.hits = 9
        rightful = _live_track(2, cluster(e[1], 3, seed=113))
        binder.step([holder, rightful], 50)                          # A taken back; holder displaced
        assert binder.identity_of(1) is None
        holder = _live_track(1, cluster(e[7], 4, seed=112))
        holder.hits = 12
        binder.step([holder, rightful], 51)                          # re-identified as someone new
        b = binder.identity_of(1)
        assert b not in (None, a)
        # The 4 hits folded at the first bind were told to memory (into A) and
        # stay told; the 8 since were never folded anywhere, so they go to B.
        # Counted once, whichever record they land in.
        assert mem.get(b).appearances == 12 - 4
        assert mem.get(a).appearances == 4 + 3

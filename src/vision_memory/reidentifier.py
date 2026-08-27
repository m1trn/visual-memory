"""Re-identification: bind a lost track to a remembered identity, or mint a new one.

A track that dies still carries its diverse exemplar embeddings. This module
asks memory "have I seen this before?" in two stages: the FAISS index proposes
candidate identities (cheap, approximate, label-agnostic), then the learned
:class:`~vision_memory.reid.Verifier` makes the actual accept/reject call
against each candidate's stored exemplar vectors. The index is only a
shortlisting device — the decision boundary always comes from the verifier, so
re-id behaviour changes with the trained threshold and not with FAISS ranking.

Labels are a hard gate: a "person" track can never merge into a "car" identity,
whatever the embeddings say.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

from vision_memory.config import load_reid_config, load_search_config
from vision_memory.memory import VisualMemory

if TYPE_CHECKING:  # a structural Protocol; needed for typing, not at runtime
    from vision_memory.reid import Verifier


@dataclass
class Resolution:
    """Outcome of resolving one lost track against memory."""

    identity_id: int
    score: float
    is_new: bool


class ReIdentifier:
    """Decides whether a dead track is a known identity or a brand new one."""

    def __init__(
        self,
        memory: VisualMemory,
        verifier: "Verifier",
        candidates: int | None = None,
        observation_quantile: float | None = None,
        threshold: float | None = None,
    ) -> None:
        self.memory = memory
        self.verifier = verifier
        # How many identities the index shortlists per query observation. The
        # index cannot filter by label, so a shortlist that is too tight can be
        # entirely consumed by wrong-label identities; searching once per
        # observation and taking the union widens it for free.
        self.candidates = candidates if candidates is not None else load_search_config().default_k
        # A threshold fitted to individual pair scores does not transfer to the
        # identity score, which maxes over exemplars and then takes a quantile
        # over observations - both push it well above a single pair's cosine.
        # `reid.calibrate_identity_threshold` fits the boundary on that statistic.
        self._threshold = threshold
        # Where and when each identity was last seen, for this session only.
        # Deliberately not persisted: "last seen at these coordinates three days
        # ago" says nothing about whether the person in front of you now is the
        # same one, whereas "vanished half a box width from here, one second
        # ago" very nearly settles it.
        self._recent: dict[int, tuple[float, np.ndarray]] = {}
        self.cfg = load_reid_config()
        self.observation_quantile = (
            observation_quantile if observation_quantile is not None
            else load_reid_config().observation_quantile
        )

    @property
    def threshold(self) -> float:
        """Boundary the accept/reject decision uses, calibrated when one was supplied."""
        return self._threshold if self._threshold is not None else self.verifier.threshold

    def resolve(
        self,
        label: str,
        embeddings: Sequence[np.ndarray],
        first_seen: float,
        last_seen: float,
        appearances: int,
        box: np.ndarray | None = None,
        unavailable: frozenset[int] | set[int] | None = None,
        held_by_others: dict[int, float] | None = None,
    ) -> Resolution:
        """Bind these observations into the best matching identity, or create one.

        ``box`` is where the object is now. Appearance alone is weak across a
        gap — measured on this footage, two views of the same person a few
        seconds apart score barely better than two different people — but an
        object reappearing where another vanished moments earlier is strong
        evidence on its own. This is exactly the continuity the tracker uses and
        re-identification was discarding, which is why the tracker handles a
        brief occlusion better than re-identification does.
        """
        obs = _stack_unit(embeddings, self.memory.dim)
        best_id, best_score = self._best_candidate(
            label, obs, first_seen, last_seen, unavailable=unavailable
        )

        # An identity worn by someone else is not simply off limits: the current
        # holder may have been the first to ask rather than the best match. Two
        # people who resemble one another both score against the same stored
        # record, and whichever track happened to appear first would otherwise
        # keep it for good, leaving the real person permanently renamed. A
        # clearly stronger claim takes it, and the displaced holder is returned
        # so the caller can re-identify them.
        contested = self._better_claim(label, obs, first_seen, last_seen,
                                       held_by_others or {}, best_id, best_score)
        if contested is not None:
            taken_id, taken_score = contested
            identity_id = self.memory.remember(
                label, list(obs), first_seen, last_seen, appearances, identity_id=taken_id
            )
            self._remember_where(identity_id, last_seen, box)
            return Resolution(identity_id=identity_id, score=taken_score, is_new=False)

        required = self.threshold
        if best_id is not None and box is not None:
            required -= self.cfg.continuity_bonus * self._continuity(best_id, first_seen, box)

        if best_id is not None and best_score >= required:
            identity_id = self.memory.remember(
                label, obs, first_seen, last_seen, appearances, identity_id=best_id
            )
            self._remember_where(identity_id, last_seen, box)
            return Resolution(identity_id=identity_id, score=best_score, is_new=False)

        identity_id = self.memory.remember(label, obs, first_seen, last_seen, appearances)
        self._remember_where(identity_id, last_seen, box)
        return Resolution(identity_id=identity_id, score=best_score, is_new=True)

    def reconsider(
        self,
        held: int,
        label: str,
        embeddings: Sequence[np.ndarray],
        first_seen: float,
        last_seen: float,
        appearances: int = 0,
        box: np.ndarray | None = None,
        unavailable: frozenset[int] | set[int] | None = None,
    ) -> Resolution | None:
        """Revisit a binding now that the track has more to say for itself.

        A track is identified early so the number on screen is stable from the
        moment the object appears, but that first decision rests on two
        observations. Without a way back, an object called new can never reclaim
        the record of its earlier visit however obvious the match later becomes,
        which is exactly how one person ends up holding two identities.

        If a different identity now matches better than the one being held, the
        two records are merged. The older record wins, so the object stays
        anchored to when it was really first seen. Returns the new resolution,
        or None when nothing changed.
        """
        obs = _stack_unit(embeddings, self.memory.dim)
        best_id, best_score = self._best_candidate(
            label, obs, first_seen, last_seen, exclude=held, unavailable=unavailable
        )
        if best_id is None:
            return None
        required = self.threshold
        if box is not None:
            required -= self.cfg.continuity_bonus * self._continuity(best_id, first_seen, box)
        # Merging rewrites history, so demand clearly more than a fresh
        # binding — but never more than evidence can supply: cosine tops out at
        # 1.0, so an uncapped bar above ~0.98 would refuse even a byte-identical
        # duplicate and silently turn this mechanism off.
        if best_score < min(required + self.cfg.merge_margin, 0.98):
            return None
        # A merge joins two RECORDS, so the two records must themselves be
        # compatible. Checking only the track against the candidate misses the
        # case that matters: a late-arriving track need not overlap an old
        # identity, yet merging would still fuse that old identity with the one
        # the track is holding - and those two may plainly have coexisted.
        mine, theirs = self.memory.get(held), self.memory.get(best_id)
        if mine is None or theirs is None:
            return None
        if mine.first_seen < theirs.last_seen and theirs.first_seen < mine.last_seen:
            return None
        keep, absorb = sorted((held, best_id))
        identity_id = self.memory.merge(keep, absorb)
        # The merge combines what was already stored; the track has since seen
        # more, and that belongs to the surviving record too.
        self.memory.remember(label, list(obs), first_seen, last_seen, appearances,
                             identity_id=identity_id)
        self._recent.pop(absorb, None)
        self._remember_where(identity_id, last_seen, box)
        return Resolution(identity_id=identity_id, score=best_score, is_new=False)

    def _better_claim(self, label: str, obs: np.ndarray, first_seen: float, last_seen: float,
                      held_by_others: dict[int, float], best_free: int | None,
                      best_free_score: float) -> tuple[int, float] | None:
        """An identity someone else holds, which this object matches clearly better.

        ``held_by_others`` maps an identity to the score its current holder
        achieved when it claimed it. Taking one requires beating that score by
        ``claim_margin`` and also beating whatever is freely available, so an
        identity only changes hands on clear evidence rather than on a tie.
        """
        if not held_by_others:
            return None
        best: tuple[int, float] | None = None
        for identity_id, incumbent in held_by_others.items():
            identity = self.memory.get(identity_id)
            if identity is None or identity.label != label:
                continue
            # The same hard fact _best_candidate enforces: an identity whose
            # lifetime overlaps this track's was on screen at the same time as
            # it, so however well the embeddings agree they are two different
            # objects, and no strength of claim can take it.
            if identity.first_seen < last_seen and first_seen < identity.last_seen:
                continue
            exemplars = self.memory.exemplar_vectors(identity_id)
            if len(exemplars) == 0:
                continue
            score = self._score_identity(obs, exemplars)
            if score < self.threshold or score < incumbent + self.cfg.claim_margin:
                continue
            if best is None or score > best[1]:
                best = (identity_id, score)
        if best is None:
            return None
        if best_free is not None and best_free_score >= best[1]:
            return None
        return best

    def score_against(self, identity_id: int, embeddings: Sequence[np.ndarray]) -> float | None:
        """How well these observations fit one stored identity, or None if it has no exemplars."""
        exemplars = self.memory.exemplar_vectors(identity_id)
        if len(exemplars) == 0:
            return None
        return self._score_identity(_stack_unit(embeddings, self.memory.dim), exemplars)

    def _continuity(self, identity_id: int, now: float, box: np.ndarray) -> float:
        """How strongly position and timing say this is the same object, in [0, 1].

        Both terms decay: an object seen a moment ago a fraction of its own width
        away scores near 1, one seen long ago or far away scores near 0. Distance
        is in box widths so it is scale free, the same reason the association
        gate normalizes that way.
        """
        seen = self._recent.get(identity_id)
        if seen is None:
            return 0.0
        when, where = seen
        gap = max(now - when, 0.0)
        width = max(float(where[2] - where[0]), 1.0)
        here = np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])
        there = np.array([(where[0] + where[2]) / 2.0, (where[1] + where[3]) / 2.0])
        distance = float(np.linalg.norm(here - there)) / width
        return float(
            np.exp(-gap / max(self.cfg.temporal_scale, 1e-6))
            * np.exp(-distance / max(self.cfg.spatial_scale, 1e-6))
        )

    def note_seen(self, identity_id: int, when: float, box: np.ndarray) -> None:
        """Record that an identity is visible here, now.

        Callers must do this for every frame an identified object is on screen,
        not merely when it is first resolved: the continuity evidence is only
        worth anything if it describes where the object actually was *last*, and
        a track that lives ten seconds would otherwise be remembered at the spot
        it occupied in its first second.
        """
        self._recent[identity_id] = (when, np.asarray(box, dtype=np.float64))
        # The stored interval must grow with the object, or the co-alive rule
        # below reads a stale end-time and lets two simultaneous objects merge.
        self.memory.touch(identity_id, when)

    def _remember_where(self, identity_id: int, when: float, box: np.ndarray | None) -> None:
        if box is not None:
            self.note_seen(identity_id, when, box)

    def _best_candidate(self, label: str, obs: np.ndarray, first_seen: float, last_seen: float,
                        exclude: int | None = None,
                        unavailable: frozenset[int] | set[int] | None = None) -> tuple[int | None, float]:
        """Highest-scoring same-label identity, or ``(None, -inf)`` if there is none.

        An identity whose lifetime overlaps this track's is skipped outright. One
        object cannot be in two places at once, so if both were on screen at the
        same moment they are provably different things, whatever their embeddings
        say. This is a hard fact rather than a threshold, and it is exactly the
        error that shows a viewer one person wearing another person's number:
        audited on this footage, *both* of the demo's re-identifications were
        between tracks that had been simultaneously alive.
        """
        best_id: int | None = None
        best_score = float("-inf")
        blocked = unavailable or ()
        for identity_id in self._shortlist(label, obs):
            if identity_id == exclude:
                continue
            # An identity that some other object is wearing right now is not
            # available, whatever the stored timestamps say. Comparing intervals
            # cannot settle this: an identity last seen at this very instant and
            # a track starting at this very instant read as consecutive under any
            # strict comparison, and as overlapping under any loose one, which
            # would then forbid every genuine return.
            if identity_id in blocked:
                continue
            identity = self.memory.get(identity_id)
            # Strict: a track ending exactly as another begins is sequential,
            # not simultaneous, and is a perfectly good re-identification.
            if identity is not None and identity.first_seen < last_seen and first_seen < identity.last_seen:
                continue
            exemplars = self.memory.exemplar_vectors(identity_id)
            if len(exemplars) == 0:
                continue
            score = self._score_identity(obs, exemplars)
            if score > best_score:
                best_id, best_score = identity_id, score
        return best_id, best_score

    def _shortlist(self, label: str, obs: np.ndarray) -> list[int]:
        """Candidate identity ids from the index, filtered to a matching label.

        ``k`` counts identities, not index rows: ``VisualMemory.search`` already
        over-fetches by the widest identity's exemplar count, so an identity
        cannot crowd itself out of the result. What this width does control is
        how far down the ranking the verifier is allowed to look, and a return
        does not always rank first — measured against 18 stored identities, a
        shortlist of 5 held the correct one 89% of the time, so one query in
        nine was decided without the right answer present at all.
        """
        if len(self.memory) == 0:
            return []
        k = self.candidates
        seen: dict[int, None] = {}
        for query in obs:
            for identity_id, _ in self.memory.search(query, k=k):
                seen.setdefault(identity_id, None)
        out = []
        for identity_id in seen:
            identity = self.memory.get(identity_id)
            if identity is not None and identity.label == label:
                out.append(identity_id)
        return out

    def _score_identity(self, obs: np.ndarray, exemplars: np.ndarray) -> float:
        """Verifier score for a whole track against one identity's exemplars.

        Max over exemplars, then a high quantile over the track's observations.

        The exemplar set is chosen for *diversity*, so most stored views
        legitimately disagree with any given crop — averaging over them would
        punish an identity for being well covered, whereas one convincing view
        match is exactly the evidence re-id needs.

        Across the track's own observations, a mean was measurably too strict:
        an object that turns partway through a track produces observations that
        genuinely match nothing stored, and they drag the average under the
        threshold. Ground-truth check on 27 tracks (first half remembered,
        second half resolved against it): mean bound 24/27, the 75th percentile
        bound 27/27. A plain max would also bind 27/27 but rests the whole
        decision on one frame, so the quantile keeps some robustness.
        """
        m, n = len(obs), len(exemplars)
        a = np.repeat(obs, n, axis=0)
        b = np.tile(exemplars, (m, 1))
        scores = np.asarray(self.verifier.score(a, b), dtype=np.float32).reshape(m, n)
        return float(np.percentile(scores.max(axis=1), 100.0 * self.observation_quantile))


def _stack_unit(embeddings: Sequence[np.ndarray], dim: int) -> np.ndarray:
    """Stack embeddings into a contiguous ``(N, dim)`` float32 array of unit vectors."""
    if len(embeddings) == 0:
        raise ValueError("resolve() needs at least one embedding")
    v = np.ascontiguousarray(
        np.stack([np.asarray(e).reshape(-1) for e in embeddings]), dtype=np.float32
    )
    if v.shape[1] != dim:
        raise ValueError(f"expected (N, {dim}), got {v.shape}")
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return np.ascontiguousarray(np.divide(v, norms, out=v.copy(), where=norms > 0))


@dataclass
class BindingEvents:
    """What one frame of binding did, for callers that count or draw."""

    created: int = 0
    rebound: int = 0
    reclaimed: int = 0
    taken: int = 0
    swapped: int = 0


class IdentityBinder:
    """Keeps live tracks bound to identities, frame by frame.

    Three scripts carried their own copy of this loop and the copies drifted:
    one fix landed in the demo but not the evaluation, another landed
    everywhere but was wrong in all three at once. The rules live here now.

    A track is identified as soon as it has enough appearance evidence, while
    it is still being watched — a viewer cannot be told who somebody is only
    after they have left. Identities worn by something on screen are off
    limits to anyone else, though a clearly stronger claim may take one; the
    loser is then re-identified. A binding made from two observations is
    revisited as the track accumulates more, so a record it missed at first can
    still be reclaimed.

    Two bookkeeping facts this class exists to get right. Only tracks that are
    *alive* make an identity unavailable — a track that has died must release
    its number, or every person who ever appeared is permanently "in use" and
    no one can ever be re-identified. And within one frame, an identity bound a
    moment ago by an earlier track is unavailable to later ones at the score it
    was won with, so the claim margin cannot be bypassed by ordering.
    """

    def __init__(self, reid: ReIdentifier, fps: float, fresh: int,
                 reconsider_every: int, min_evidence: int = 2,
                 swap_margin: float = 0.0, min_new_identity_confidence: float = 0.0,
                 convincing_confidence: float = 0.0) -> None:
        self.reid = reid
        self.fps = fps
        self.fresh = fresh
        self.reconsider_every = reconsider_every
        self.min_evidence = min_evidence
        self.swap_margin = swap_margin
        self.min_new_identity_confidence = min_new_identity_confidence
        # A detection at or above this is a real sighting; below it, a match
        # came from the tracker's low-confidence second pass and says nothing
        # about whether a person is there. The tracker's own high_conf.
        self.convincing_confidence = convincing_confidence
        self.bound: dict[int, Resolution] = {}
        # Tracks whose identity was taken while they were unseen. Such a track
        # is the same object's ghost, coasting on a stale prediction while the
        # real detection went to a new track; re-identifying it would give a
        # phantom box a fresh number. It stays dormant until actually seen.
        self._dormant: set[int] = set()
        self.first_frame: dict[int, int] = {}

    def identity_of(self, track_id: int) -> int | None:
        res = self.bound.get(track_id)
        return None if res is None else res.identity_id

    def step(self, active: Sequence, frame_idx: int) -> BindingEvents:
        """Bind, revisit and record whereabouts for every live track this frame."""
        events = BindingEvents()
        fps = self.fps
        for track in active:
            self.first_frame.setdefault(track.id, frame_idx)

        # What each visible object holds, and how well it fits that record NOW.
        # The score it was bound with is the wrong yardstick: the track that
        # created an identity holds it at -inf, so any newcomer clearing the
        # threshold could take a 13-hit person's number the moment he was
        # occluded. A challenger must beat the holder's actual fit.
        held_by_others: dict[int, float] = {}
        for t in active:
            res = self.bound.get(t.id)
            if res is None or t.time_since_update > self.fresh:
                continue
            fit = self.reid.score_against(res.identity_id, t.exemplars) if len(t.exemplars) else None
            held_by_others[res.identity_id] = res.score if fit is None else max(fit, res.score)
        in_use = set(held_by_others)

        if frame_idx % self.reconsider_every == 0:
            for track in active:
                held = self.bound.get(track.id)
                if held is None or not len(track.exemplars):
                    continue
                revised = self.reid.reconsider(
                    held.identity_id, track.label, track.exemplars,
                    self.first_frame[track.id] / fps, frame_idx / fps, track.hits,
                    box=track.box, unavailable=in_use - {held.identity_id},
                )
                if revised is not None:
                    for other, res in list(self.bound.items()):
                        if res.identity_id == held.identity_id:
                            self.bound[other] = revised
                    events.reclaimed += 1
            events.swapped += self._swap_crossed_numbers(active)

        # Identities bound earlier in THIS frame, at the score they were won
        # with. Rebuilt from live bindings only: a dead track's entry must not
        # count, which is why the whole of `bound` is never used here.
        taken_now: dict[int, float] = {}
        for track in active:
            if track.id in self.bound or len(track.exemplars) < self.min_evidence:
                continue
            if track.id in self._dormant:
                # A ghost is released only by a convincing sighting. ByteTrack's
                # second pass will happily feed a coasting box a 0.18 detection,
                # and that is not evidence of a person.
                if track.time_since_update > 0 or track.score < self.convincing_confidence:
                    continue
                self._dormant.discard(track.id)
            res = self.reid.resolve(
                track.label, track.exemplars, self.first_frame[track.id] / fps,
                frame_idx / fps, track.hits, box=track.box,
                unavailable=in_use | set(taken_now),
                held_by_others={**held_by_others, **taken_now},
            )
            if res.is_new and track.peak_score < self.min_new_identity_confidence:
                # Not convincing enough to be a new person. The record just
                # created is withdrawn; the track may still bind to an existing
                # identity on a later frame if its appearance says so.
                self.reid.memory.forget(res.identity_id)
                continue
            for other in active:
                if other.id != track.id and self.identity_of(other.id) == res.identity_id:
                    del self.bound[other.id]
                    events.taken += 1
                    # The loser is a ghost unless it was seen convincingly this
                    # very frame: a coasting box, or one held alive by a weak
                    # second-pass detection, is the same object's stale copy.
                    if other.time_since_update > 0 or other.score < self.convincing_confidence:
                        self._dormant.add(other.id)
            self.bound[track.id] = res
            taken_now[res.identity_id] = res.score
            if res.is_new:
                events.created += 1
            else:
                events.rebound += 1

        for track in active:
            res = self.bound.get(track.id)
            if res is not None and track.time_since_update == 0:
                self.reid.note_seen(res.identity_id, frame_idx / fps, track.box)
        return events

    def _swap_crossed_numbers(self, active: Sequence) -> int:
        """Give two live tracks each other's numbers back when both fit better.

        Takeover runs only when a track is first identified and the second
        look skips numbers in use, so two people who exchanged numbers during
        a crossing kept the wrong ones for as long as both stayed on screen.
        The condition is mutual: A must fit B's record better than its own AND
        B must fit A's better than its own, each by ``swap_margin``, and each
        must clear the binding threshold on the other's record. One confused
        frame on one side cannot flip two people.
        """
        if self.swap_margin <= 0.0:
            return 0
        live = [t for t in active if t.id in self.bound and len(t.exemplars)
                and t.time_since_update <= self.fresh]
        own: dict[int, float | None] = {
            t.id: self.reid.score_against(self.bound[t.id].identity_id, t.exemplars) for t in live
        }
        swaps = 0
        done: set[int] = set()
        for i, a in enumerate(live):
            if a.id in done or own[a.id] is None:
                continue
            for b in live[i + 1:]:
                if b.id in done or own[b.id] is None or a.label != b.label:
                    continue
                ida, idb = self.bound[a.id].identity_id, self.bound[b.id].identity_id
                a_on_b = self.reid.score_against(idb, a.exemplars)
                b_on_a = self.reid.score_against(ida, b.exemplars)
                if a_on_b is None or b_on_a is None:
                    continue
                bar = self.reid.threshold
                if (a_on_b >= bar and b_on_a >= bar
                        and a_on_b >= own[a.id] + self.swap_margin
                        and b_on_a >= own[b.id] + self.swap_margin):
                    self.bound[a.id] = Resolution(identity_id=idb, score=a_on_b, is_new=False)
                    self.bound[b.id] = Resolution(identity_id=ida, score=b_on_a, is_new=False)
                    done.update((a.id, b.id))
                    swaps += 1
                    break
        return swaps

    def forget(self, track_id: int) -> tuple[Resolution | None, int | None]:
        """Release a dead track's binding and return what it held.

        Returns ``(resolution, first_frame)`` so the caller can fold the track's
        final observations into the identity it wore.
        """
        self._dormant.discard(track_id)
        return self.bound.pop(track_id, None), self.first_frame.pop(track_id, None)

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
        best_id, best_score = self._best_candidate(label, obs, first_seen, last_seen)

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

    def _remember_where(self, identity_id: int, when: float, box: np.ndarray | None) -> None:
        if box is not None:
            self.note_seen(identity_id, when, box)

    def _best_candidate(self, label: str, obs: np.ndarray, first_seen: float, last_seen: float) -> tuple[int | None, float]:
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
        for identity_id in self._shortlist(label, obs):
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
        """Candidate identity ids from the index, filtered to a matching label."""
        if len(self.memory) == 0:
            return []
        seen: dict[int, None] = {}
        for query in obs:
            for identity_id, _ in self.memory.search(query, k=self.candidates):
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

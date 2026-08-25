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

from vision_memory.config import load_search_config
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
    ) -> None:
        self.memory = memory
        self.verifier = verifier
        # How many identities the index shortlists per query observation. The
        # index cannot filter by label, so a shortlist that is too tight can be
        # entirely consumed by wrong-label identities; searching once per
        # observation and taking the union widens it for free.
        self.candidates = candidates if candidates is not None else load_search_config().default_k

    def resolve(
        self,
        label: str,
        embeddings: Sequence[np.ndarray],
        first_seen: float,
        last_seen: float,
        appearances: int,
    ) -> Resolution:
        """Bind these observations into the best matching identity, or create one."""
        obs = _stack_unit(embeddings, self.memory.dim)
        best_id, best_score = self._best_candidate(label, obs)

        if best_id is not None and best_score >= self.verifier.threshold:
            identity_id = self.memory.remember(
                label, obs, first_seen, last_seen, appearances, identity_id=best_id
            )
            return Resolution(identity_id=identity_id, score=best_score, is_new=False)

        identity_id = self.memory.remember(label, obs, first_seen, last_seen, appearances)
        return Resolution(identity_id=identity_id, score=best_score, is_new=True)

    def _best_candidate(self, label: str, obs: np.ndarray) -> tuple[int | None, float]:
        """Highest-scoring same-label identity, or ``(None, -inf)`` if there is none."""
        best_id: int | None = None
        best_score = float("-inf")
        for identity_id in self._shortlist(label, obs):
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

        Max over exemplars, mean over observations. The exemplar set is chosen
        for *diversity*, so most stored views legitimately disagree with any
        given crop — averaging over them would punish an identity for being
        well covered, whereas one convincing view match is exactly the evidence
        re-id needs. Across the track's own observations the opposite holds:
        they are all the same object seen moments apart, so the whole track
        should agree, and a mean stops a single lucky frame from carrying the
        decision.
        """
        m, n = len(obs), len(exemplars)
        a = np.repeat(obs, n, axis=0)
        b = np.tile(exemplars, (m, 1))
        scores = np.asarray(self.verifier.score(a, b), dtype=np.float32).reshape(m, n)
        return float(scores.max(axis=1).mean())


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

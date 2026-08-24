"""Persistent visual memory: identity records in SQLite joined to a FAISS index.

An identity is one remembered object. It owns a *prototype* (the appearance-
weighted mean of every embedding ever seen for it, L2-normalized) kept in
SQLite only, plus up to K *exemplars* — a diversity-selected subset of the raw
embeddings, and the only vectors that live in the searchable index. Searching
against exemplars rather than prototypes keeps multi-view objects findable; the
prototype stays as the single stable summary vector for that identity.

The FAISS index is keyed by ``exemplars.vector_id``, so a vector always maps
back to exactly one identity row.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from vision_memory.config import MemoryConfig
from vision_memory.search import VectorIndex

_SCHEMA = """
CREATE TABLE IF NOT EXISTS identities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    label TEXT NOT NULL,
    first_seen REAL NOT NULL,
    last_seen REAL NOT NULL,
    appearances INTEGER NOT NULL,
    prototype BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS exemplars (
    vector_id INTEGER PRIMARY KEY AUTOINCREMENT,
    identity_id INTEGER NOT NULL REFERENCES identities(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_exemplars_identity ON exemplars(identity_id);
"""


@dataclass
class Identity:
    """One remembered object: metadata plus its prototype embedding."""

    id: int
    label: str
    first_seen: float
    last_seen: float
    appearances: int
    prototype: np.ndarray


def select_diverse(vectors: np.ndarray, k: int) -> np.ndarray:
    """Greedy max-min diversity selection; returns the chosen row indices.

    Seeds with the vector closest to the set mean (the most representative
    view), then repeatedly adds whichever remaining vector is least similar to
    everything already chosen. Cheap O(n^2) farthest-point traversal, which is
    fine because n is bounded by the exemplars of a single identity.
    """
    v = np.ascontiguousarray(vectors, dtype=np.float32)
    if v.ndim != 2:
        raise ValueError(f"expected (N, D), got {v.shape}")
    n = len(v)
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    if n <= k:
        return np.arange(n, dtype=np.int64)

    sims = v @ v.T
    start = int(np.argmax(v @ v.mean(axis=0)))
    chosen = [start]
    worst = sims[start].copy()  # max cosine from each vector to the chosen set
    while len(chosen) < k:
        worst[chosen] = np.inf
        nxt = int(np.argmin(worst))
        chosen.append(nxt)
        worst = np.maximum(worst, sims[nxt])
    return np.asarray(chosen, dtype=np.int64)


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v.astype(np.float32) if n == 0.0 else (v / n).astype(np.float32)


class VisualMemory:
    """Long-lived store of identities, their prototypes and their exemplars."""

    def __init__(self, cfg: MemoryConfig, dim: int) -> None:
        self.cfg = cfg
        self.dim = dim
        self._db_path = Path(cfg.db_path)
        self._index_path = Path(cfg.index_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self._db_path)
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.executescript(_SCHEMA)
        self._db.commit()
        if self._index_path.exists():
            self._index = VectorIndex.load(self._index_path)
            if self._index.dim != dim:
                raise ValueError(f"index at {self._index_path} has dim {self._index.dim}, expected {dim}")
            self._reconcile()
        else:
            self._index = VectorIndex(dim)

    def __enter__(self) -> "VisualMemory":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __len__(self) -> int:
        return int(self._db.execute("SELECT COUNT(*) FROM identities").fetchone()[0])

    def remember(
        self,
        label: str,
        embeddings: Sequence[np.ndarray],
        first_seen: float,
        last_seen: float,
        appearances: int,
        identity_id: int | None = None,
    ) -> int:
        """Store or merge observations of one object; returns its identity id.

        With ``identity_id=None`` a new identity is created. Otherwise the
        observations are folded into that identity: metadata is extended, the
        prototype is re-derived as an appearance-weighted mean, and the
        exemplar set is re-selected from old exemplars plus new embeddings.
        """
        obs = self._stack(embeddings)
        if identity_id is None:
            proto = _normalize(obs.mean(axis=0))
            cur = self._db.execute(
                "INSERT INTO identities (label, first_seen, last_seen, appearances, prototype)"
                " VALUES (?, ?, ?, ?, ?)",
                (label, float(first_seen), float(last_seen), int(appearances), proto.tobytes()),
            )
            identity_id = int(cur.lastrowid)
            pool = obs
        else:
            row = self._db.execute(
                "SELECT first_seen, last_seen, appearances, prototype FROM identities WHERE id = ?",
                (identity_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"no identity {identity_id}")
            old_ids = self._exemplar_ids(identity_id)
            old_vectors = np.stack([self._index.get(i) for i in old_ids]) if old_ids else None
            old_first, old_last, old_app, old_proto_blob = row
            old_proto = np.frombuffer(old_proto_blob, dtype=np.float32)
            # Weighted so a long-lived identity is not dragged by one short revisit.
            # Both terms are unit-norm first, or the new one is silently discounted
            # by however spread out the fresh observations happen to be.
            proto = _normalize(old_app * old_proto + appearances * _normalize(obs.mean(axis=0)))
            self._db.execute(
                "UPDATE identities SET label = ?, first_seen = ?, last_seen = ?,"
                " appearances = ?, prototype = ? WHERE id = ?",
                (
                    label,
                    min(float(first_seen), float(old_first)),
                    max(float(last_seen), float(old_last)),
                    int(old_app) + int(appearances),
                    proto.tobytes(),
                    identity_id,
                ),
            )
            pool = obs if old_vectors is None else np.concatenate([old_vectors, obs])
            if old_ids:
                self._index.remove(old_ids)
                self._db.executemany(
                    "DELETE FROM exemplars WHERE vector_id = ?", [(i,) for i in old_ids]
                )

        keep = select_diverse(pool, self.cfg.exemplars_per_identity)
        new_ids = []
        for _ in range(len(keep)):
            cur = self._db.execute("INSERT INTO exemplars (identity_id) VALUES (?)", (identity_id,))
            new_ids.append(int(cur.lastrowid))
        self._index.add(pool[keep], new_ids)
        return identity_id

    def search(self, embedding: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
        """Top-``k`` identities by best-matching exemplar, best cosine first."""
        if len(self._index) == 0:
            return []
        # Over-fetch: one identity can occupy several index slots. The widest
        # identity is read from the data, not from config, because identities
        # written under a larger K keep their rows after the setting changes.
        widest = self._db.execute(
            "SELECT COALESCE(MAX(cnt), 1) FROM (SELECT COUNT(*) AS cnt FROM exemplars GROUP BY identity_id)"
        ).fetchone()[0]
        want = min(len(self._index), max(1, k) * max(1, int(widest)))
        scores, vector_ids = self._index.search(embedding, want)
        hits = [(float(s), int(v)) for s, v in zip(scores[0], vector_ids[0]) if v >= 0]
        if not hits:
            return []
        placeholders = ",".join("?" * len(hits))
        owner = dict(self._db.execute(
            f"SELECT vector_id, identity_id FROM exemplars WHERE vector_id IN ({placeholders})",
            [v for _, v in hits],
        ).fetchall())
        best: dict[int, float] = {}
        for score, vid in hits:
            ident = owner.get(vid)
            if ident is None:
                continue
            if score > best.get(ident, -np.inf):
                best[ident] = float(score)
        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:k]

    def get(self, identity_id: int) -> Identity | None:
        """Return one identity, or None if it does not exist."""
        row = self._db.execute(
            "SELECT id, label, first_seen, last_seen, appearances, prototype FROM identities WHERE id = ?",
            (identity_id,),
        ).fetchone()
        return None if row is None else _to_identity(row, self.dim)

    def all_identities(self) -> list[Identity]:
        """Every stored identity, ordered by id."""
        rows = self._db.execute(
            "SELECT id, label, first_seen, last_seen, appearances, prototype FROM identities ORDER BY id"
        ).fetchall()
        return [_to_identity(r, self.dim) for r in rows]

    def exemplar_vectors(self, identity_id: int) -> np.ndarray:
        """The stored exemplar embeddings for one identity, shape ``(n, dim)``."""
        ids = self._exemplar_ids(identity_id)
        if not ids:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.stack([self._index.get(i) for i in ids])

    def save(self) -> None:
        """Flush the FAISS index, then commit SQLite.

        Order matters: the index is written first so that a crash in between
        leaves vectors with no exemplar row, which ``search`` and ``remember``
        already skip. The reverse order would leave exemplar rows pointing at
        vectors that were never persisted, which nothing can recover from.
        """
        self._index.save(self._index_path)
        self._db.commit()

    def _reconcile(self) -> None:
        """Drop exemplar rows whose vectors are missing from the loaded index.

        Guards against an index and a database that were not written together —
        a half-completed save, or a stale index file beside a fresh database
        (exemplar ids restart at 1, so those would otherwise collide).
        """
        rows = self._db.execute("SELECT vector_id FROM exemplars").fetchall()
        orphans = []
        for (vid,) in rows:
            try:
                self._index.get(int(vid))
            except Exception:
                orphans.append((int(vid),))
        if orphans:
            self._db.executemany("DELETE FROM exemplars WHERE vector_id = ?", orphans)
            self._db.commit()

    def close(self) -> None:
        """Persist everything and release the database handle."""
        self.save()
        self._db.close()

    def _exemplar_ids(self, identity_id: int) -> list[int]:
        rows = self._db.execute(
            "SELECT vector_id FROM exemplars WHERE identity_id = ? ORDER BY vector_id",
            (identity_id,),
        ).fetchall()
        return [int(r[0]) for r in rows]

    def _stack(self, embeddings: Sequence[np.ndarray]) -> np.ndarray:
        if len(embeddings) == 0:
            raise ValueError("remember() needs at least one embedding")
        v = np.ascontiguousarray(np.stack([np.asarray(e).reshape(-1) for e in embeddings]), dtype=np.float32)
        if v.shape[1] != self.dim:
            raise ValueError(f"expected (N, {self.dim}), got {v.shape}")
        # Everything downstream reads inner products as cosines, so unit norm is
        # enforced here rather than assumed of the caller.
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        return np.ascontiguousarray(np.divide(v, norms, out=v.copy(), where=norms > 0))


def _to_identity(row: tuple, dim: int | None = None) -> Identity:
    proto = np.frombuffer(row[5], dtype=np.float32).copy()  # copy: frombuffer is read-only
    if dim is not None and proto.shape[0] != dim:
        raise ValueError(f"identity {row[0]} has a {proto.shape[0]}-dim prototype, expected {dim}")
    return Identity(
        id=int(row[0]),
        label=str(row[1]),
        first_seen=float(row[2]),
        last_seen=float(row[3]),
        appearances=int(row[4]),
        prototype=proto,
    )

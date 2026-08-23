"""Exact k-NN over L2-normalized embeddings (FAISS flat inner-product index).

Pure vector store: no encoder, no metadata. Callers own the int64 ids
(later: SQLite row ids) and decide what a score means.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import faiss
import numpy as np


class VectorIndex:
    """Brute-force cosine search with caller-assigned ids.

    Scores are inner products; on unit-norm vectors that is cosine similarity
    in [-1, 1]. When fewer than ``k`` vectors are stored, missing slots come
    back as id ``-1`` with score ``-inf``.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))

    def __len__(self) -> int:
        return self._index.ntotal

    def add(self, vectors: np.ndarray, ids: Sequence[int]) -> None:
        """Insert rows of ``vectors`` under the given ids. Ids must be unique."""
        v = self._check(vectors)
        if len(ids) != len(v):
            raise ValueError(f"{len(v)} vectors but {len(ids)} ids")
        self._index.add_with_ids(v, np.asarray(ids, dtype=np.int64))

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Top-k per query. Returns ``(scores, ids)``, each shape ``(Q, k)``."""
        q = self._check(queries)
        if len(self) == 0:
            return (np.full((len(q), k), -np.inf, np.float32),
                    np.full((len(q), k), -1, np.int64))
        scores, ids = self._index.search(q, k)
        return scores, ids

    def remove(self, ids: Sequence[int]) -> int:
        """Delete by id. Returns number actually removed."""
        return int(self._index.remove_ids(np.asarray(ids, dtype=np.int64)))

    def get(self, id_: int) -> np.ndarray:
        """Return the stored vector for ``id_``."""
        return self._index.reconstruct(int(id_))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path))

    @classmethod
    def load(cls, path: Path) -> "VectorIndex":
        idx = faiss.read_index(str(path))
        obj = cls.__new__(cls)
        obj.dim = idx.d
        obj._index = idx
        return obj

    def _check(self, x: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"expected (N, {self.dim}), got {x.shape}")
        return x

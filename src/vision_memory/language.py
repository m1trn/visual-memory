"""Finding objects by description instead of by example.

Every other capability here compares a stored object against a new one. This
one compares a stored object against a *sentence*, which the rest of the system
cannot do: its vectors are deliberately INSTANCE vectors, trained so two people
in similar coats land apart. Text search needs the opposite - "a red backpack"
must match every red backpack - so a space that is good at telling individuals
apart is by construction bad at grouping them by kind. One space cannot serve
both.

So this is a second, parallel space. A CLIP-family model embeds images and text
into one shared vector space; the image side describes a stored identity, the
text side describes what the user typed, and the match is the same cosine
distance used everywhere else in this project. The vectors live in their own
index keyed by identity id, so nothing about the identity path changes.

The default model is chosen for speed on a CPU. A smaller architecture is not
automatically faster here: models designed for phone neural engines can run
slower under desktop PyTorch than a larger conventional transformer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class LanguageConfig:
    """Which CLIP-family model to load, and how confident a hit must be."""

    model_name: str = "ViT-B-32"
    pretrained: str = "laion2b_s34b_b79k"
    batch_size: int = 16
    # Cosine below this is not a match but the least-bad row in the index.
    # Text-image cosine lives on a much tighter scale than image-image: a good
    # hit sits near 0.28, not near 0.8.
    min_score: float = 0.22


class TextImageEmbedder:
    """One CLIP model, exposing an image side and a text side of one space.

    Both sides return L2-normalised rows, so similarity is a dot product, the
    same convention as every other vector in this project.
    """

    def __init__(self, cfg: LanguageConfig | None = None) -> None:
        import open_clip  # imported lazily: an optional capability, not a core one
        import torch

        self.cfg = cfg or LanguageConfig()
        self._torch = torch
        self.model, _, self._preprocess = open_clip.create_model_and_transforms(
            self.cfg.model_name, pretrained=self.cfg.pretrained)
        self.model.eval()
        self._tokenize = open_clip.get_tokenizer(self.cfg.model_name)
        self._dim = int(self.model.text_projection.shape[-1]) if hasattr(
            self.model, "text_projection") else int(self.encode_text("x").shape[-1])

    @property
    def dim(self) -> int:
        return self._dim

    def encode_images(self, crops_rgb: Sequence[np.ndarray]) -> np.ndarray:
        """Embed crops. Returns ``(N, dim)`` float32, each row unit-norm."""
        from PIL import Image

        if len(crops_rgb) == 0:
            return np.empty((0, self._dim), np.float32)
        out = []
        with self._torch.no_grad():
            for i in range(0, len(crops_rgb), self.cfg.batch_size):
                batch = self._torch.stack([
                    self._preprocess(Image.fromarray(np.ascontiguousarray(c)))
                    for c in crops_rgb[i : i + self.cfg.batch_size]])
                v = self.model.encode_image(batch)
                out.append((v / v.norm(dim=-1, keepdim=True)).numpy().astype(np.float32))
        return np.concatenate(out, axis=0)

    def encode_text(self, text: str | Sequence[str]) -> np.ndarray:
        """Embed one query, or several. Returns ``(dim,)`` or ``(N, dim)``."""
        one = isinstance(text, str)
        queries = [text] if one else list(text)
        with self._torch.no_grad():
            v = self.model.encode_text(self._tokenize(queries))
            v = (v / v.norm(dim=-1, keepdim=True)).numpy().astype(np.float32)
        return v[0] if one else v


class SemanticIndex:
    """Descriptions of remembered objects, searchable by sentence.

    Deliberately one vector per identity rather than per exemplar. Semantic
    search asks "which of the things I have seen is a red backpack", and the
    answer is an identity; keeping several near-identical CLIP vectors per
    identity would multiply the cost and the storage to sharpen a distinction
    the question never makes.
    """

    def __init__(self, dim: int) -> None:
        from vision_memory.search import VectorIndex

        self._index = VectorIndex(dim)
        self.dim = dim

    def __len__(self) -> int:
        return len(self._index)

    def describe(self, identity_id: int, vector: np.ndarray) -> None:
        """Record (or replace) what one identity looks like, semantically."""
        v = np.asarray(vector, np.float32).reshape(1, -1)
        self._index.remove([identity_id])
        self._index.add(v, [identity_id])

    def forget(self, identity_id: int) -> None:
        self._index.remove([identity_id])

    def find(self, text_vector: np.ndarray, k: int, min_score: float
             ) -> list[tuple[int, float]]:
        """Identities matching a text vector, best first, weak matches dropped.

        A vector index always returns its k nearest rows, so without a floor an
        empty scene still produces confident-looking answers. The floor is what
        makes "nothing here matches" expressible.
        """
        if len(self._index) == 0:
            return []
        scores, ids = self._index.search(
            np.asarray(text_vector, np.float32).reshape(1, -1), min(k, len(self._index)))
        return [(int(i), float(s)) for s, i in zip(scores[0], ids[0])
                if i >= 0 and s >= min_score]

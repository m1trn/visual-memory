"""Frozen vision encoder: image/crop -> L2-normalized embedding.

This is the single representation the whole system consumes. torch never
leaks past this module; callers get float32 numpy vectors.
"""

from __future__ import annotations

from typing import Sequence, Union

import warnings

import numpy as np
import torch
from PIL import Image

from vision_memory.config import EncoderConfig

ImageLike = Union[np.ndarray, Image.Image]

# DINOv2 was trained with ImageNet statistics; anything else silently degrades features.
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class Encoder:
    """Wraps a frozen DINOv2 backbone.

    Input images must be RGB: HWC uint8 numpy arrays or PIL images.
    OpenCV frames are BGR — convert once at the boundary with
    ``cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)``.
    """

    def __init__(self, cfg: EncoderConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        if cfg.backend == "dinov2":
            with warnings.catch_warnings():
                # DINOv2 probes for xFormers, an optional GPU attention library,
                # and warns per layer when it is absent. On CPU it is not wanted.
                warnings.filterwarnings("ignore", message="xFormers is not available")
                self.model = torch.hub.load("facebookresearch/dinov2", cfg.name, verbose=False)
        elif cfg.backend == "radio":
            # RADIO applies its own input conditioning, so images stay in [0, 1].
            self.model = torch.hub.load(
                "NVlabs/RADIO", "radio_model", version=cfg.name,
                progress=False, skip_validation=True, trust_repo=True,
            )
        else:
            raise ValueError(f"unknown encoder backend: {cfg.backend!r}")
        self.model.eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._dim = self._probe_dim()

    def _probe_dim(self) -> int:
        """Ask the model for its own embedding width with one dummy forward."""
        if self.cfg.backend == "dinov2":
            return int(self.model.embed_dim)
        with torch.inference_mode():
            probe = torch.zeros(1, 3, self.cfg.input_size, self.cfg.input_size, device=self.device)
            return int(self._forward(probe).shape[-1])

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """Backend-specific forward returning the global descriptor, shape (B, dim)."""
        if self.cfg.backend == "dinov2":
            return self.model(x)  # CLS token
        out = self.model(x)
        return out.summary if hasattr(out, "summary") else out[0]

    @property
    def dim(self) -> int:
        """Embedding dimensionality (384 for ViT-S)."""
        return self._dim

    def encode(self, image: ImageLike) -> np.ndarray:
        """Embed one image. Returns shape ``(dim,)`` float32, unit L2 norm."""
        return self.encode_batch([image])[0]

    def encode_batch(self, images: Sequence[ImageLike]) -> np.ndarray:
        """Embed many images. Returns shape ``(N, dim)`` float32, each row unit-norm.

        Chunks internally by ``cfg.batch_size`` to bound memory.
        """
        if len(images) == 0:
            return np.empty((0, self._dim), dtype=np.float32)
        out: list[np.ndarray] = []
        bs = self.cfg.batch_size
        with torch.inference_mode():
            for i in range(0, len(images), bs):
                x = self._preprocess(images[i : i + bs]).to(self.device)
                feats = self._forward(x)  # (B, dim)
                feats = torch.nn.functional.normalize(feats, dim=-1)
                out.append(feats.cpu().numpy().astype(np.float32, copy=False))
        return np.concatenate(out, axis=0)

    def _preprocess(self, images: Sequence[ImageLike]) -> torch.Tensor:
        """RGB images -> normalized float tensor (B, 3, S, S)."""
        size = self.cfg.input_size
        arrs = []
        for img in images:
            pil = img if isinstance(img, Image.Image) else Image.fromarray(np.asarray(img))
            pil = pil.convert("RGB").resize((size, size), Image.BICUBIC)
            arrs.append(np.asarray(pil, dtype=np.float32) / 255.0)
        x = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2)
        if self.cfg.backend == "radio":
            return x  # RADIO normalizes internally via its input conditioner
        return (x - _MEAN) / _STD

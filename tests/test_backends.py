"""Contract tests every encoder backend must satisfy.

The RADIO backend downloads ~400 MB of weights, so it is opt-in:
    VISION_MEMORY_TEST_BACKENDS=dinov2,radio python -m pytest tests/test_backends.py
Default runs the lightweight backend only.
"""

from __future__ import annotations

import dataclasses
import os

import numpy as np
import pytest
from PIL import Image

from vision_memory.config import load_encoder_config
from vision_memory.encoder import Encoder

# backend -> (torch.hub name, input size). Input size must be a multiple of the patch size.
BACKENDS = {
    "dinov2": ("dinov2_vits14", 224),
    "radio": ("c-radio_v3-b", 256),
}
_SELECTED = os.environ.get("VISION_MEMORY_TEST_BACKENDS", "dinov2").split(",")


@pytest.fixture(scope="module", params=[b for b in _SELECTED if b in BACKENDS])
def enc(request: pytest.FixtureRequest) -> Encoder:
    name, size = BACKENDS[request.param]
    cfg = dataclasses.replace(load_encoder_config(), backend=request.param, name=name, input_size=size)
    return Encoder(cfg)


def test_reports_a_positive_dim(enc: Encoder) -> None:
    assert enc.dim > 0


def test_encodes_to_unit_norm_vectors(enc: Encoder) -> None:
    img = np.random.randint(0, 255, (96, 64, 3), dtype=np.uint8)
    v = enc.encode(img)
    assert v.shape == (enc.dim,) and v.dtype == np.float32
    assert abs(np.linalg.norm(v) - 1.0) < 1e-5


def test_batch_matches_single_and_accepts_pil(enc: Encoder) -> None:
    a = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    b = Image.fromarray(np.random.randint(0, 255, (80, 40, 3), dtype=np.uint8))
    batch = enc.encode_batch([a, b])
    assert batch.shape == (2, enc.dim)
    assert np.allclose(batch[0], enc.encode(a), atol=1e-4)
    assert np.allclose(batch[1], enc.encode(b), atol=1e-4)


def test_similar_images_score_higher_than_unrelated(enc: Encoder) -> None:
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)
    near = np.clip(base.astype(np.int16) + rng.integers(-12, 12, base.shape), 0, 255).astype(np.uint8)
    far = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)
    v = enc.encode_batch([base, near, far])
    assert float(v[0] @ v[1]) > float(v[0] @ v[2])


def test_unknown_backend_rejected() -> None:
    cfg = dataclasses.replace(load_encoder_config(), backend="nope")
    with pytest.raises(ValueError):
        Encoder(cfg)

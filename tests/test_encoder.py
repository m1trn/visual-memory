import numpy as np
import pytest
from PIL import Image

from vision_memory.config import load_encoder_config
from vision_memory.encoder import Encoder


@pytest.fixture(scope="module")
def enc() -> Encoder:
    return Encoder(load_encoder_config())


def test_single_is_unit_norm(enc: Encoder) -> None:
    img = np.random.randint(0, 255, (120, 80, 3), dtype=np.uint8)
    v = enc.encode(img)
    assert v.shape == (enc.dim,) and v.dtype == np.float32
    assert abs(np.linalg.norm(v) - 1.0) < 1e-5


def test_batch_matches_single_and_accepts_pil(enc: Encoder) -> None:
    a = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    b = Image.fromarray(np.random.randint(0, 255, (100, 50, 3), dtype=np.uint8))
    batch = enc.encode_batch([a, b])
    assert batch.shape == (2, enc.dim)
    assert np.allclose(batch[0], enc.encode(a), atol=1e-5)
    assert np.allclose(batch[1], enc.encode(b), atol=1e-5)


def test_empty_batch(enc: Encoder) -> None:
    assert enc.encode_batch([]).shape == (0, enc.dim)

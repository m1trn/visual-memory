from __future__ import annotations

import numpy as np
import pytest

from vision_memory.appearance import colour_bands, fuse
from vision_memory.config import AppearanceConfig, load_appearance_config


def solid(colour: tuple[int, int, int], h: int = 40, w: int = 20) -> np.ndarray:
    patch = np.zeros((h, w, 3), dtype=np.uint8)
    patch[:, :] = colour
    return patch


def test_colour_bands_are_unit_norm_and_sized_by_band_count() -> None:
    one = colour_bands(solid((10, 200, 200)), 1)
    two = colour_bands(solid((10, 200, 200)), 2)
    assert two.shape[0] == 2 * one.shape[0]
    for v in (one, two):
        assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5


def test_bands_notice_arrangement_that_one_histogram_cannot() -> None:
    """Blue coat over dark trousers is not the same person as the reverse.

    A single histogram counts colours without caring where they were, so those
    two are indistinguishable to it. Splitting by band restores the arrangement.
    """
    blue, dark = (110, 200, 200), (10, 30, 40)
    a = np.vstack([solid(blue, h=20), solid(dark, h=20)])
    b = np.vstack([solid(dark, h=20), solid(blue, h=20)])
    one = float(colour_bands(a, 1) @ colour_bands(b, 1))
    two = float(colour_bands(a, 2) @ colour_bands(b, 2))
    assert one > 0.99  # identical to a single histogram
    assert two < 0.1   # plainly different once arrangement is kept


def test_fused_cosine_is_the_weighted_sum_of_its_parts() -> None:
    rng = np.random.default_rng(0)
    def unit(n: int) -> np.ndarray:
        v = rng.standard_normal(n).astype(np.float32)
        return v / np.linalg.norm(v)
    d1, d2, c1, c2 = unit(8), unit(8), unit(6), unit(6)
    w = 0.4
    got = float(fuse(d1, c1, w) @ fuse(d2, c2, w))
    want = (1 - w) * float(d1 @ d2) + w * float(c1 @ c2)
    assert got == pytest.approx(want, abs=1e-6)
    assert abs(float(np.linalg.norm(fuse(d1, c1, w))) - 1.0) < 1e-5


def test_config_is_wired() -> None:
    cfg = load_appearance_config()
    assert isinstance(cfg, AppearanceConfig)
    assert 0.0 <= cfg.colour_weight <= 1.0 and cfg.colour_bands >= 1


def test_reid_model_sees_the_whole_box_and_encoder_only_the_top() -> None:
    """A re-id network is trained on whole-body crops; a general encoder is not."""
    import dataclasses

    from vision_memory.appearance import build_embedder

    cfg = load_appearance_config()
    if cfg.model != "reid":
        pytest.skip("configured appearance model is not the re-id network")
    _, upper = build_embedder(cfg)
    assert upper == 1.0

    encoder_cfg = dataclasses.replace(cfg, model="encoder")
    assert encoder_cfg.deep_upper_fraction < 1.0

    with pytest.raises(ValueError):
        build_embedder(dataclasses.replace(cfg, model="nonsense"))


def test_reid_net_pads_short_batches_and_returns_unit_vectors() -> None:
    """The export has a fixed batch; a short call must not silently mis-align."""
    from vision_memory.appearance import ReidNet

    cfg = load_appearance_config()
    if cfg.model != "reid":
        pytest.skip("configured appearance model is not the re-id network")
    net = ReidNet(cfg.reid_model_path)
    for count in (1, 3, net._batch + 2 if net._batch else 5):
        crops = [np.random.randint(0, 255, (64, 32, 3), dtype=np.uint8) for _ in range(count)]
        out = net.encode_batch(crops)
        assert out.shape == (count, net.dim)
        assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-4)
    assert net.encode_batch([]).shape == (0, net.dim)

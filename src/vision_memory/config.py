"""Typed access to configs/default.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


@dataclass(frozen=True)
class EncoderConfig:
    backend: str
    name: str
    input_size: int
    device: str
    batch_size: int


@dataclass(frozen=True)
class DetectorConfig:
    name: str
    model_path: str
    input_size: int
    conf_threshold: float
    iou_threshold: float
    classes: tuple[int, ...] | None
    num_threads: int


@dataclass(frozen=True)
class TrackerConfig:
    max_age: int
    min_hits: int
    iou_threshold: float
    contested_iou: float
    max_centre_distance: float
    measurement_noise: float
    min_exemplar_confidence: float
    high_conf: float
    low_conf: float
    veto_views: int
    appearance_veto: float
    claim_margin: float
    appearance_weight: float
    embed_every_n: int
    max_exemplars: int


@dataclass(frozen=True)
class MemoryConfig:
    db_path: str
    index_path: str
    exemplars_per_identity: int
    reid_threshold: float


@dataclass(frozen=True)
class ReidConfig:
    verifier: str
    min_pair_gap: int
    max_pairs_per_track: int
    observation_quantile: float
    max_false_merge_rate: float
    claim_margin: float
    swap_margin: float
    min_new_identity_confidence: float
    merge_margin: float
    reconsider_every: int
    continuity_bonus: float
    spatial_scale: float
    temporal_scale: float
    test_fraction: float
    seed: int


@dataclass(frozen=True)
class AppearanceConfig:
    model: str
    reid_model_path: str
    specialist_labels: tuple[str, ...]
    colour_weight: float
    colour_bands: int
    deep_upper_fraction: float
    min_crop_px: int


@dataclass(frozen=True)
class VideoConfig:
    detect_every_n_frames: int
    min_crop_px: int
    crop_upper_fraction: float
    embed_for_association: bool


@dataclass(frozen=True)
class SearchConfig:
    default_k: int


@dataclass(frozen=True)
class AnomalyConfig:
    method: str
    k: int
    shrinkage: float
    n_estimators: int
    nu: float
    seed: int


def load_raw(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load the YAML config as a plain dict."""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_encoder_config(path: Path = DEFAULT_CONFIG_PATH) -> EncoderConfig:
    """Load only the `encoder` section as a typed dataclass."""
    return EncoderConfig(**load_raw(path)["encoder"])


def load_search_config(path: Path = DEFAULT_CONFIG_PATH) -> SearchConfig:
    """Load only the `search` section as a typed dataclass."""
    return SearchConfig(**load_raw(path)["search"])


def load_anomaly_config(path: Path = DEFAULT_CONFIG_PATH) -> AnomalyConfig:
    """Load only the `anomaly` section as a typed dataclass."""
    return AnomalyConfig(**load_raw(path)["anomaly"])


def load_detector_config(path: Path = DEFAULT_CONFIG_PATH) -> DetectorConfig:
    """Load only the `detector` section as a typed dataclass."""
    raw = dict(load_raw(path)["detector"])
    raw["classes"] = tuple(raw["classes"]) if raw["classes"] else None
    return DetectorConfig(**raw)


def load_tracker_config(path: Path = DEFAULT_CONFIG_PATH) -> TrackerConfig:
    """Load only the `tracker` section as a typed dataclass."""
    return TrackerConfig(**load_raw(path)["tracker"])


def load_memory_config(path: Path = DEFAULT_CONFIG_PATH) -> MemoryConfig:
    """Load only the `memory` section as a typed dataclass."""
    return MemoryConfig(**load_raw(path)["memory"])


def load_reid_config(path: Path = DEFAULT_CONFIG_PATH) -> ReidConfig:
    """Load only the `reid` section as a typed dataclass."""
    return ReidConfig(**load_raw(path)["reid"])


def load_video_config(path: Path = DEFAULT_CONFIG_PATH) -> VideoConfig:
    """Load only the `video` section as a typed dataclass."""
    return VideoConfig(**load_raw(path)["video"])


def load_appearance_config(path: Path = DEFAULT_CONFIG_PATH) -> AppearanceConfig:
    """Load only the `appearance` section as a typed dataclass."""
    raw = dict(load_raw(path)["appearance"])
    raw["specialist_labels"] = tuple(raw["specialist_labels"])
    return AppearanceConfig(**raw)

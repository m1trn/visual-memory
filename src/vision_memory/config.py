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
    max_centre_distance: float
    high_conf: float
    low_conf: float
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
class VideoConfig:
    detect_every_n_frames: int
    min_crop_px: int


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


def load_video_config(path: Path = DEFAULT_CONFIG_PATH) -> VideoConfig:
    """Load only the `video` section as a typed dataclass."""
    return VideoConfig(**load_raw(path)["video"])

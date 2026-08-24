"""Typed access to configs/default.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


@dataclass(frozen=True)
class EncoderConfig:
    name: str
    input_size: int
    device: str
    batch_size: int


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

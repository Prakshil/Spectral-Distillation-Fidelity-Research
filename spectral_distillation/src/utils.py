"""Shared utilities: seeding, config loading, logging, device selection."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import yaml


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_config(device: str = "cuda") -> str:
    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        logging.getLogger(__name__).warning(
            "CUDA requested but unavailable; falling back to 'cpu'."
        )
        return "cpu"
    return device


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_yaml_config(*paths: str | Path) -> dict:
    if not paths:
        raise ValueError("load_yaml_config requires at least one path")
    config: dict = {}
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            config = _deep_merge(config, yaml.safe_load(handle) or {})
    return config


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out
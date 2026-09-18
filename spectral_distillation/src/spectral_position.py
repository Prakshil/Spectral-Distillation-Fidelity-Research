"""Per-node spectral position and frequency-share routing signals."""

from __future__ import annotations

import numpy as np


def node_spectral_position(V: np.ndarray, evals: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    V = np.asarray(V, dtype=float)
    evals = np.asarray(evals, dtype=float)
    squared = V ** 2
    numerator = squared @ evals
    denominator = squared.sum(axis=1)
    return numerator / (denominator + eps)


def node_spectral_energy(V: np.ndarray, evals: np.ndarray) -> np.ndarray:
    V = np.asarray(V, dtype=float)
    evals = np.asarray(evals, dtype=float)
    return (V ** 2) @ evals


def high_frequency_share(V: np.ndarray, evals: np.ndarray, band_start: int = 2, eps: float = 1e-12) -> np.ndarray:
    V = np.asarray(V, dtype=float)
    evals = np.asarray(evals, dtype=float)
    total = node_spectral_energy(V, evals)
    high = (V[:, band_start:] ** 2) @ evals[band_start:]
    return high / (total + eps)


def spectral_ratio(position: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    position = np.asarray(position, dtype=float)
    return position / (position.max() + eps)
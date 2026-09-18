"""Spectral distortion diagnostics: the pre-training routing-quality metric."""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigvalsh

from spectral_distillation.src.spectral import min_eigengap


def compute_spectral_distortion(L_a: np.ndarray, L_sparse: np.ndarray, eps: float = 1e-8) -> float:
    e_a = np.sort(eigvalsh(np.asarray(L_a, dtype=float)))
    e_s = np.sort(eigvalsh(np.asarray(L_sparse, dtype=float)))
    relative_errors = np.abs(e_a - e_s) / (np.abs(e_a) + eps)
    relative_errors[0] = 0.0
    return float(relative_errors.max())


def compute_frobenius_norm(L_a: np.ndarray, L_sparse: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(L_a, dtype=float) - np.asarray(L_sparse, dtype=float), "fro"))


def compute_fragility(SD: float, min_gap: float, eps: float = 1e-8) -> float:
    return float(SD / (min_gap + eps))


def predict_retention(SD: float) -> float:
    return float(max(0.0, 1.0 - SD))


def davis_kahan_bound(frob_norm: float, min_gap: float, eps: float = 1e-8) -> float:
    return float(2.0 * frob_norm / (min_gap + eps))


def spectral_distortion_report(
    L_a: np.ndarray, L_sparse: np.ndarray, eps: float = 1e-8
) -> dict:
    L_a = np.asarray(L_a, dtype=float)
    L_sparse = np.asarray(L_sparse, dtype=float)
    e_a = np.sort(eigvalsh(L_a))
    e_s = np.sort(eigvalsh(L_sparse))
    relative_errors = np.abs(e_a - e_s) / (np.abs(e_a) + eps)
    relative_errors[0] = 0.0
    gaps = np.diff(e_a)
    min_gap = float(gaps.min()) if gaps.size else float("inf")
    frob = float(np.linalg.norm(L_a - L_sparse, "fro"))
    sd = float(relative_errors.max())
    return {
        "SD": sd,
        "min_eigengap": min_gap,
        "frobenius_norm": frob,
        "davis_kahan_bound": davis_kahan_bound(frob, min_gap, eps),
        "predicted_retention": predict_retention(sd),
        "fragility_score": compute_fragility(sd, min_gap, eps),
        "eigenvalues_dense": e_a.tolist(),
        "eigenvalues_sparse": e_s.tolist(),
        "relative_errors": relative_errors.tolist(),
    }
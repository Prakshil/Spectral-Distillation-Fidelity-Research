"""Empirical verification of Weyl, Davis-Kahan, and Cauchy interlacing bounds."""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh, eigvalsh

from spectral_distillation.src.spectral import compute_eigenvalues, min_eigengap


def check_weyl(L_a: np.ndarray, L_sparse: np.ndarray, eps: float = 1e-8) -> dict:
    L_a = np.asarray(L_a, dtype=float)
    L_sparse = np.asarray(L_sparse, dtype=float)
    e_a = compute_eigenvalues(L_a)
    e_s = compute_eigenvalues(L_sparse)
    shifts = np.abs(e_s - e_a)
    spectral_norm = float(np.linalg.norm(L_a - L_sparse, 2))
    frob_norm = float(np.linalg.norm(L_a - L_sparse, "fro"))
    violations = np.flatnonzero(shifts > spectral_norm + eps)
    return {
        "holds": bool(violations.size == 0),
        "max_shift": float(shifts.max()),
        "spectral_norm": spectral_norm,
        "frobenius_norm": frob_norm,
        "n_violations": int(violations.size),
    }


def check_davis_kahan(
    L_a: np.ndarray, L_sparse: np.ndarray, eps: float = 1e-8
) -> dict:
    L_a = np.asarray(L_a, dtype=float)
    L_sparse = np.asarray(L_sparse, dtype=float)
    e_a, va = eigh(L_a)
    _, vs = eigh(L_sparse)
    frob_norm = float(np.linalg.norm(L_a - L_sparse, "fro"))
    min_gap = min_eigengap(e_a)
    bound = frob_norm / (min_gap + eps)

    n = L_a.shape[0]
    sines = []
    for k in range(n):
        cos_angle = float(abs(va[:, k] @ vs[:, k]))
        sines.append(float(np.sqrt(max(0.0, 1.0 - cos_angle ** 2))))
    sines = np.asarray(sines)
    tol = max(eps, 1e-6)
    violations = np.flatnonzero(sines > bound + tol)
    return {
        "holds": bool(violations.size == 0),
        "bound_sin_theta": bound,
        "max_sin_theta": float(sines.max()) if sines.size else 0.0,
        "min_eigengap": min_gap,
        "frobenius_norm": frob_norm,
        "per_k_sin_theta": sines.tolist(),
        "n_violations": int(violations.size),
    }


def edge_removal_eigenvalues(W: np.ndarray, i: int, j: int) -> np.ndarray:
    W_removed = np.asarray(W, dtype=float).copy()
    W_removed[i, j] = 0.0
    W_removed[j, i] = 0.0
    from spectral_distillation.src.laplacian import compute_laplacian

    return compute_eigenvalues(compute_laplacian(W_removed))


def check_cauchy_interlacing(L: np.ndarray, i: int, j: int, eps: float = 1e-8) -> dict:
    L = np.asarray(L, dtype=float)
    n = L.shape[0]
    w = abs(L[i, j])
    e_i = np.zeros(n)
    e_j = np.zeros(n)
    e_i[i] = 1.0
    e_j[j] = 1.0
    L_removed = L - w * np.outer(e_i - e_j, e_i - e_j)
    e_full = compute_eigenvalues(L)
    e_removed = compute_eigenvalues(L_removed)
    return {
        "holds": bool(np.all(e_removed <= e_full + eps)),
        "componentwise_decrease": (e_full - e_removed).tolist(),
        "max_decrease": float((e_full - e_removed).max()),
        "edge_weight": float(w),
        "eigenvalues_full": e_full.tolist(),
        "eigenvalues_removed": e_removed.tolist(),
    }


def check_all_bounds(W: np.ndarray, tau: float = 0.5, eps: float = 1e-8) -> dict:
    from spectral_distillation.src.baselines import threshold_sparsify
    from spectral_distillation.src.laplacian import compute_laplacian

    L_full = compute_laplacian(W)
    W_sparse = threshold_sparsify(W, tau)
    L_sparse = compute_laplacian(W_sparse)
    return {
        "weyl": check_weyl(L_full, L_sparse, eps),
        "davis_kahan": check_davis_kahan(L_full, L_sparse, eps),
        "eigenvalues_dense": eigvalsh(L_full).tolist(),
        "eigenvalues_sparse": eigvalsh(L_sparse).tolist(),
    }
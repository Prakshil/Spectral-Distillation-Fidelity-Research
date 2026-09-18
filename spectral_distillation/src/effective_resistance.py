"""Effective resistance via exact pseudoinverse or top-k eigendecomposition."""

from __future__ import annotations

import numpy as np
from scipy.linalg import pinvh
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

from spectral_distillation.src.laplacian import compute_laplacian, edges_from_weights


def effective_resistance_exact(L: np.ndarray) -> np.ndarray:
    L = np.asarray(L, dtype=float)
    pinv = pinvh(L)
    d = np.diag(pinv)
    R = d[:, None] + d[None, :] - 2.0 * pinv
    return np.maximum(R, 0.0)


def effective_resistance_topk(L: np.ndarray, k: int = 100, tol: float = 1e-10) -> np.ndarray:
    L = np.asarray(L, dtype=float)
    n = L.shape[0]
    if k >= n - 1:
        return effective_resistance_exact(L)
    k = int(max(1, min(k, n - 1)))
    sigma = -1e-6
    evals, evecs = eigsh(csr_matrix(L), k=k + 1, which="LM", sigma=sigma)
    order = np.argsort(evals)
    evals, evecs = evals[order], evecs[:, order]
    keep = evals > tol
    evals, evecs = evals[keep], evecs[:, keep]
    scaled = evecs / np.sqrt(evals[None, :])
    pinv = scaled @ scaled.T
    d = np.diag(pinv)
    R = d[:, None] + d[None, :] - 2.0 * pinv
    return np.maximum(R, 0.0)


def compute_effective_resistance(
    L: np.ndarray, method: str | None = None, k: int = 100
) -> np.ndarray:
    n = np.asarray(L, dtype=float).shape[0]
    if method is None:
        method = "exact" if n <= 2000 else "topk"
    if method == "exact":
        return effective_resistance_exact(L)
    if method == "topk":
        return effective_resistance_topk(L, k=k)
    raise ValueError(f"unknown effective-resistance method: {method}")


def resistance_for_edges(R: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    return R[rows, cols]


def resistance_energy_identity(W: np.ndarray) -> float:
    rows, cols, weights = edges_from_weights(W)
    R = effective_resistance_exact(compute_laplacian(W))
    return float(np.sum(weights * R[rows, cols]))
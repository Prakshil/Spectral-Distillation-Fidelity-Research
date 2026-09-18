"""Spectral routines: eigenvalues, eigenvectors, eigengaps, connectivity facts."""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh, eigvalsh


def compute_eigenvalues(L: np.ndarray) -> np.ndarray:
    return eigvalsh(np.asarray(L, dtype=float))


def compute_eigenvectors(L: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    evals, evecs = eigh(np.asarray(L, dtype=float))
    return evals, evecs


def compute_eigengaps(evals: np.ndarray) -> np.ndarray:
    e = np.sort(np.asarray(evals, dtype=float))
    return np.diff(e)


def min_eigengap(evals: np.ndarray) -> float:
    gaps = compute_eigengaps(evals)
    if gaps.size == 0:
        return float("inf")
    return float(gaps.min())


def fiedler_value(evals: np.ndarray) -> float:
    e = np.sort(np.asarray(evals, dtype=float))
    if e.size < 2:
        return 0.0
    return float(max(e[1], 0.0))


def count_zero_eigenvalues(evals: np.ndarray, tol: float = 1e-8) -> int:
    return int(np.sum(np.abs(np.sort(np.asarray(evals, dtype=float))) < tol))


def eigenvector_orthonormality(V: np.ndarray, tol: float = 1e-6) -> bool:
    gram = V.T @ V
    return bool(np.allclose(gram, np.eye(V.shape[1]), atol=tol))
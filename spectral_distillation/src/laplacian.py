"""Graph Laplacian construction: adjacency, degrees, and quadratic forms."""

from __future__ import annotations

import numpy as np


def build_adjacency(A: np.ndarray, symmetrize: bool = True) -> np.ndarray:
    W = np.maximum(np.asarray(A, dtype=float), 0.0)
    if symmetrize:
        W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    return W


def compute_degree_matrix(W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    degrees = np.asarray(W, dtype=float).sum(axis=1)
    return np.diag(degrees), degrees


def compute_laplacian(W: np.ndarray) -> np.ndarray:
    D, _ = compute_degree_matrix(W)
    return D - np.asarray(W, dtype=float)


def quadratic_form(L: np.ndarray, f: np.ndarray) -> float:
    f = np.asarray(f, dtype=float)
    return float(f @ np.asarray(L, dtype=float) @ f)


def edges_from_weights(W: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    W = build_adjacency(W, symmetrize=False)
    rows, cols = np.triu_indices(W.shape[0], k=1)
    weights = W[rows, cols]
    keep = weights > 0
    return rows[keep], cols[keep], weights[keep]


def normalize_adjacency(W: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Symmetrically normalized adjacency (D^-1/2 (W + I) D^-1/2).

    Adds self-loops before normalizing so isolated nodes remain self-connected,
    matching the message-passing convention used by the GNN router.
    """
    W = build_adjacency(W)
    n = W.shape[0]
    A = W + np.eye(n)
    d_sqrt = np.sqrt(A.sum(axis=1))
    d_inv_sqrt = 1.0 / np.maximum(d_sqrt, eps)
    Dn = np.diag(d_inv_sqrt)
    return Dn @ A @ Dn


def is_laplacian(L: np.ndarray, tol: float = 1e-8) -> bool:
    L = np.asarray(L, dtype=float)
    symmetric = np.allclose(L, L.T, atol=tol)
    zero_rows = np.allclose(L.sum(axis=1), 0.0, atol=tol)
    psl = bool(np.all(np.linalg.eigvalsh(0.5 * (L + L.T)) >= -tol))
    return symmetric and zero_rows and psl
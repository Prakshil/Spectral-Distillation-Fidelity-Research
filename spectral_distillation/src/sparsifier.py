"""Spielman-Srivastava spectral sparsification with business-rule oracle filter."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from spectral_distillation.src.effective_resistance import compute_effective_resistance
from spectral_distillation.src.laplacian import (
    build_adjacency,
    compute_laplacian,
    edges_from_weights,
)

OracleScore = Callable[[int, int], float]


def spectral_sparsify(
    A: np.ndarray,
    n_edges_budget: int,
    oracle_score: OracleScore | None = None,
    delta: float = 0.5,
    epsilon: float = 0.1,
    k_eig: int = 100,
    sample: bool = False,
    seed: int | None = None,
    q_floor: float = 1e-8,
) -> np.ndarray:
    W = build_adjacency(A)
    rows, cols, weights = edges_from_weights(W)
    n = W.shape[0]
    n_edges = weights.size
    if n_edges_budget >= n_edges or n_edges == 0:
        return W
    if n_edges_budget < 1:
        return np.zeros_like(W)

    L = compute_laplacian(W)
    R = compute_effective_resistance(L, k=k_eig)
    r = R[rows, cols]

    logn = np.log(n) + 1e-12
    C = n_edges_budget / (n * logn)
    q = np.minimum(1.0, C * weights * r * logn / (epsilon ** 2 + 1e-12))
    q = np.maximum(q, 0.0)

    if oracle_score is not None:
        scores = np.asarray(
            [oracle_score(int(i), int(j)) for i, j in zip(rows.tolist(), cols.tolist())]
        )
        q = np.where(scores >= delta, q, 0.0)

    order = np.argsort(-q, kind="stable")
    keep = np.zeros(n_edges, dtype=bool)

    if sample:
        rng = np.random.default_rng(seed)
        keep = rng.random(n_edges) < q
        if int(keep.sum()) > n_edges_budget:
            keep = np.zeros(n_edges, dtype=bool)
            keep[order[:n_edges_budget]] = True
    else:
        if (q > 0).sum() <= n_edges_budget:
            keep = q > 0
        else:
            keep[order[:n_edges_budget]] = True

    W_sparse = np.zeros_like(W)
    rescaled = weights / np.maximum(q, q_floor)
    added = rescaled * keep
    W_sparse[rows, cols] = added
    W_sparse[cols, rows] = added
    return W_sparse
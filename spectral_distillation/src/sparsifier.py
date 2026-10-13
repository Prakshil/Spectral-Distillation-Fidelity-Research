"""Effective-resistance edge sampling with a business-rule oracle filter.

This is *inspired by* Spielman-Srivastava sparsification -- edges are scored by
``w_e * r_e`` (weight times effective resistance) -- but it is **not** an
SS-style spectral approximation. The SS guarantee needs a sampling constant
chosen from ``(n, eps)`` alone, independent of the target edge count. Here that
constant is instead set to ``n_edges_budget / (n log n)`` so the expected number
of kept edges lands near the requested budget; the ``log n`` cancels and the
scheme is best described as **budget-normalized resistance sampling**. The
returned weights are importance-rescaled (Horvitz-Thompson) by ``1/q`` so
resistance-weighted mass is preserved in expectation, but no ``1 +/- eps``
Laplacian bound is claimed or tested.
"""

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
    resistance_matrix: np.ndarray | None = None,
    resistance_edges: np.ndarray | None = None,
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
    if resistance_edges is not None:
        r = np.asarray(resistance_edges, dtype=float)
        if r.size != n_edges:
            raise ValueError(
                f"resistance_edges has {r.size} entries but the graph has {n_edges} edges"
            )
    elif resistance_matrix is not None:
        R = np.asarray(resistance_matrix, dtype=float)
        r = R[rows, cols]
    else:
        R = compute_effective_resistance(L, k=k_eig)
        r = R[rows, cols]

    # Budget-normalized sampling probability (see module docstring). Written in
    # SS form for readability, but C is budget-derived so the log n factors
    # cancel and this targets an edge count, not a 1 +/- eps spectral bound.
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
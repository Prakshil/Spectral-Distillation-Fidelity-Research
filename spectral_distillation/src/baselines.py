"""Baseline graph-distillation methods: thresholding, random, degree-based.

``random_sparsify`` and ``degree_sparsify`` use different weight conventions.
Random edge sampling is a randomized estimator, so it importance-rescales the
kept weights by ``n_edges / n_edges_budget`` to preserve edge-weight mass in
expectation. Degree-based selection is deterministic top-k, so it keeps the
original weights and simply drops the rest. Comparisons of homophily or
spectral distortion across the two therefore contrast a mass-preserving sampler
with a mass-dropping selector, not merely two ranking rules -- keep this in mind
when reading the method bands.
"""

from __future__ import annotations

import numpy as np

from spectral_distillation.src.laplacian import build_adjacency, edges_from_weights


def threshold_sparsify(A: np.ndarray, tau: float) -> np.ndarray:
    W = build_adjacency(A)
    return np.where(W > tau, W, 0.0)


def random_sparsify(
    A: np.ndarray, n_edges_budget: int, seed: int | None = None
) -> np.ndarray:
    W = build_adjacency(A)
    rows, cols, weights = edges_from_weights(W)
    n_edges = weights.size
    if n_edges_budget >= n_edges or n_edges == 0:
        return W
    rng = np.random.default_rng(seed)
    chosen = rng.choice(n_edges, size=int(n_edges_budget), replace=False)
    W_sparse = np.zeros_like(W)
    rescaled = weights * (n_edges / n_edges_budget)
    W_sparse[rows[chosen], cols[chosen]] = rescaled[chosen]
    W_sparse[cols[chosen], rows[chosen]] = rescaled[chosen]
    return W_sparse


def degree_sparsify(
    A: np.ndarray, n_edges_budget: int, seed: int | None = None
) -> np.ndarray:
    W = build_adjacency(A)
    rows, cols, weights = edges_from_weights(W)
    n_edges = weights.size
    if n_edges_budget >= n_edges or n_edges == 0:
        return W
    degrees = W.sum(axis=1)
    score = (degrees[rows] + 1.0) * (degrees[cols] + 1.0)
    order = np.argsort(-score, kind="stable")
    chosen = order[:n_edges_budget]
    W_sparse = np.zeros_like(W)
    W_sparse[rows[chosen], cols[chosen]] = weights[chosen]
    W_sparse[cols[chosen], rows[chosen]] = weights[chosen]
    return W_sparse
"""Tests for the baseline distillation methods."""

import numpy as np
import pytest

from spectral_distillation.src.baselines import degree_sparsify, random_sparsify, threshold_sparsify
from spectral_distillation.src.laplacian import compute_laplacian, edges_from_weights
from spectral_distillation.src.spectral import count_zero_eigenvalues, fiedler_value
from spectral_distillation.src.synthetic import stochastic_block_model

A_FULL = np.array(
    [
        [0.0, 0.8, 0.6, 0.3],
        [0.8, 0.0, 0.9, 0.2],
        [0.6, 0.9, 0.0, 0.1],
        [0.3, 0.2, 0.1, 0.0],
    ]
)


def _n_edges(W):
    return edges_from_weights(W)[0].size


def test_threshold_disconnects_weak_customer():
    W_sp = threshold_sparsify(A_FULL, 0.5)
    assert W_sp[3, :].sum() == 0.0
    e = np.sort(np.linalg.eigvalsh(compute_laplacian(W_sp)))
    assert count_zero_eigenvalues(e) == 2
    assert fiedler_value(e) == pytest.approx(0.0, abs=1e-8)


def test_threshold_high_tau_empties_graph():
    W_sp = threshold_sparsify(A_FULL, 0.95)
    assert _n_edges(W_sp) == 0


def test_threshold_low_tau_keeps_everything():
    W_sp = threshold_sparsify(A_FULL, 0.0)
    assert _n_edges(W_sp) == 6


def test_random_respects_budget_and_is_reproducible():
    W, _ = stochastic_block_model(40, 4, seed=0)
    budget = 50
    W_a = random_sparsify(W, budget, seed=7)
    W_b = random_sparsify(W, budget, seed=7)
    assert np.allclose(W_a, W_b)
    assert _n_edges(W_a) == budget


def test_random_total_weight_preserved_in_expectation():
    W, _ = stochastic_block_model(80, 4, seed=1)
    budget = 100
    W_sp = random_sparsify(W, budget, seed=0)
    total_full = W.sum() / 2
    total_sparse = W_sp.sum() / 2
    assert total_sparse == pytest.approx(total_full, rel=0.15)


def test_random_full_budget_returns_original():
    W, _ = stochastic_block_model(20, 3, seed=2)
    W_sp = random_sparsify(W, _n_edges(W) + 5, seed=0)
    assert np.allclose(W_sp, W)


def test_degree_sparsify_prefers_high_degree_edges():
    W, _ = stochastic_block_model(30, 3, seed=3)
    budget = 10
    W_sp = degree_sparsify(W, budget, seed=0)
    assert _n_edges(W_sp) == budget
    rows, cols, _ = edges_from_weights(W_sp)
    degrees = W.sum(axis=1)
    kept_scores = (degrees[rows] + 1) * (degrees[cols] + 1)
    rows_all, cols_all, _ = edges_from_weights(W)
    all_scores = (degrees[rows_all] + 1) * (degrees[cols_all] + 1)
    boundary = float(np.partition(all_scores, -budget)[-budget])
    assert kept_scores.min() >= boundary - 1e-9
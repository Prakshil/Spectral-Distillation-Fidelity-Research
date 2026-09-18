"""Tests for spectral sparsification: budgets, oracle filter, spectrum fidelity."""

import numpy as np
import pytest

from spectral_distillation.src.baselines import threshold_sparsify
from spectral_distillation.src.distortion import compute_spectral_distortion
from spectral_distillation.src.laplacian import compute_laplacian, edges_from_weights
from spectral_distillation.src.spectral import count_zero_eigenvalues, fiedler_value, min_eigengap
from spectral_distillation.src.sparsifier import spectral_sparsify
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
    rows, cols, _ = edges_from_weights(W)
    return rows.size


def test_budget_is_respected_deterministic():
    W, _ = stochastic_block_model(40, 4, seed=0)
    budget = 60
    W_sp = spectral_sparsify(W, budget, seed=0)
    assert _n_edges(W_sp) <= budget


def test_full_budget_returns_original():
    W, _ = stochastic_block_model(30, 3, seed=1)
    W_sp = spectral_sparsify(W, _n_edges(W) + 10)
    assert np.allclose(W_sp, W)


def test_zero_budget_returns_empty():
    W, _ = stochastic_block_model(20, 3, seed=2)
    W_sp = spectral_sparsify(W, 0)
    assert _n_edges(W_sp) == 0


def test_oracle_filter_excludes_low_score_edges():
    W, _ = stochastic_block_model(40, 4, seed=3)
    budget = 80

    def oracle(i, j):
        return 0.9 if (i + j) % 5 == 0 else 0.1

    W_sp = spectral_sparsify(W, budget, oracle_score=oracle, delta=0.5, seed=0)
    rows, cols, _ = edges_from_weights(W_sp)
    for i, j in zip(rows, cols):
        assert oracle(int(i), int(j)) >= 0.5


def test_oracle_filter_can_exclude_everything():
    W, _ = stochastic_block_model(20, 3, seed=4)
    W_sp = spectral_sparsify(W, 10, oracle_score=lambda i, j: 0.0, delta=0.5, seed=0)
    assert _n_edges(W_sp) == 0


def test_spectral_beats_threshold_fiedler_preservation_at_budget():
    budget = 4
    W_sp_spec = spectral_sparsify(A_FULL, budget, seed=0)
    L_full = compute_laplacian(A_FULL)
    L_spec = compute_laplacian(W_sp_spec)
    e_spec = np.sort(np.linalg.eigvalsh(L_spec))
    assert count_zero_eigenvalues(e_spec) == 1
    assert fiedler_value(e_spec) > 0.3


def test_spectral_low_sd_when_budget_allows():
    W_sp_spec = spectral_sparsify(A_FULL, 5, seed=0)
    L_full = compute_laplacian(A_FULL)
    L_sp_thr = compute_laplacian(threshold_sparsify(A_FULL, 0.5))
    sd_spec = compute_spectral_distortion(L_full, compute_laplacian(W_sp_spec))
    sd_thr = compute_spectral_distortion(L_full, L_sp_thr)
    assert sd_spec == pytest.approx(0.1975, abs=1e-3)
    assert sd_spec < sd_thr


def test_sd_monotonically_decreases_with_budget():
    W, _ = stochastic_block_model(50, 4, seed=5)
    L_full = compute_laplacian(W)
    sds = []
    for budget in (10, 30, 60, 120):
        W_sp = spectral_sparsify(W, budget, seed=0)
        sds.append(compute_spectral_distortion(L_full, compute_laplacian(W_sp)))
    diffs = np.diff(sds)
    assert np.all(diffs <= 1e-6)


def test_spectrum_preserved_within_epsilon_for_large_budget():
    W, _ = stochastic_block_model(60, 4, seed=6)
    n = W.shape[0]
    budget = int(15 * n * np.log(n))
    W_sp = spectral_sparsify(W, budget, epsilon=0.2, seed=0)
    assert _n_edges(W_sp) <= budget
    sd = compute_spectral_distortion(compute_laplacian(W), compute_laplacian(W_sp))
    assert sd < 0.2


def test_stochastic_mode_reproducible_and_bounded():
    W, _ = stochastic_block_model(40, 4, seed=7)
    budget = 60
    W_a = spectral_sparsify(W, budget, sample=True, seed=42)
    W_b = spectral_sparsify(W, budget, sample=True, seed=42)
    assert np.allclose(W_a, W_b)
    assert _n_edges(W_a) <= budget


def test_gap_preserved_for_connected_graph():
    W, _ = stochastic_block_model(60, 4, seed=8)
    L_full = compute_laplacian(W)
    e_full = np.sort(np.linalg.eigvalsh(L_full))
    budget = int(8 * W.shape[0])
    for seed in range(3):
        W_sp = spectral_sparsify(W, budget, seed=seed)
        e_sp = np.sort(np.linalg.eigvalsh(compute_laplacian(W_sp)))
        assert count_zero_eigenvalues(e_sp) == 1
        assert min_eigengap(e_sp) > 0.5 * min_eigengap(e_full) - 1e-6
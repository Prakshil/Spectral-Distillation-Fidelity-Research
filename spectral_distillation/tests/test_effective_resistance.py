"""Tests for effective resistance: identity, symmetry, and top-k approximation."""

import numpy as np
import pytest

from spectral_distillation.src.effective_resistance import (
    compute_effective_resistance,
    effective_resistance_exact,
    effective_resistance_topk,
    resistance_energy_identity,
)
from spectral_distillation.src.laplacian import compute_laplacian
from spectral_distillation.src.synthetic import stochastic_block_model

A_FULL = np.array(
    [
        [0.0, 0.8, 0.6, 0.3],
        [0.8, 0.0, 0.9, 0.2],
        [0.6, 0.9, 0.0, 0.1],
        [0.3, 0.2, 0.1, 0.0],
    ]
)


def test_energy_identity_sum_wr_equals_n_minus_1():
    assert resistance_energy_identity(A_FULL) == pytest.approx(3.0, abs=1e-6)


def test_energy_identity_on_random_graph():
    W, _ = stochastic_block_model(40, 4, seed=0)
    n = W.shape[0]
    assert resistance_energy_identity(W) == pytest.approx(n - 1, abs=1e-5)


def test_resistance_matrix_symmetric_and_nonnegative():
    L = compute_laplacian(A_FULL)
    R = effective_resistance_exact(L)
    assert np.allclose(R, R.T)
    assert np.all(R >= -1e-12)
    assert np.allclose(np.diag(R), 0.0)


def test_weak_customer_edges_have_high_effective_resistance():
    L = compute_laplacian(A_FULL)
    R = effective_resistance_exact(L)
    for i, j in [(0, 3), (1, 3), (2, 3)]:
        assert R[i, j] > 1.8
    assert R[0, 3] > R[1, 2]
    assert R[1, 3] > R[0, 1]


def test_topk_matches_exact_when_k_is_full_rank():
    W, _ = stochastic_block_model(60, 4, seed=1)
    L = compute_laplacian(W)
    R_exact = effective_resistance_exact(L)
    R_topk = effective_resistance_topk(L, k=60)
    assert np.allclose(R_exact, R_topk, atol=1e-6)


def test_topk_approximates_exact_on_truncated_rank():
    W, _ = stochastic_block_model(120, 4, seed=2)
    L = compute_laplacian(W)
    R_exact = effective_resistance_exact(L)
    rows, cols = np.triu_indices(W.shape[0], k=1)
    R_k30 = effective_resistance_topk(L, k=30)
    R_k60 = effective_resistance_topk(L, k=60)
    assert np.all(R_k30[rows, cols] <= R_exact[rows, cols] + 1e-8)
    assert np.all(R_k60[rows, cols] <= R_exact[rows, cols] + 1e-8)
    err30 = np.abs(R_k30[rows, cols] - R_exact[rows, cols]).mean()
    err60 = np.abs(R_k60[rows, cols] - R_exact[rows, cols]).mean()
    assert err60 < err30
    assert err60 < 0.1


def test_dispatch_picks_exact_for_small_and_topk_is_callable():
    L = compute_laplacian(A_FULL)
    R = compute_effective_resistance(L)
    assert np.allclose(R, effective_resistance_exact(L))
    R2 = compute_effective_resistance(L, method="topk", k=4)
    assert np.allclose(R, R2, atol=1e-5)
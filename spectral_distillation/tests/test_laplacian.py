"""Tests for Laplacian construction on the 4-node ERP running example."""

import numpy as np

from spectral_distillation.src.laplacian import (
    build_adjacency,
    compute_degree_matrix,
    compute_laplacian,
    is_laplacian,
    quadratic_form,
)

A_FULL = np.array(
    [
        [0.0, 0.8, 0.6, 0.3],
        [0.8, 0.0, 0.9, 0.2],
        [0.6, 0.9, 0.0, 0.1],
        [0.3, 0.2, 0.1, 0.0],
    ]
)


def test_build_adjacency_symmetrizes_and_zeroes_diagonal():
    W = build_adjacency(A_FULL, symmetrize=True)
    assert np.allclose(W, W.T)
    assert np.trace(W) == 0.0
    assert np.allclose(W, A_FULL)


def test_degree_matrix_matches_guide():
    D, degrees = compute_degree_matrix(build_adjacency(A_FULL))
    assert np.allclose(np.diag(D), [1.7, 1.9, 1.6, 0.6])
    assert np.allclose(degrees, [1.7, 1.9, 1.6, 0.6])


def test_laplacian_is_valid_psd_with_zero_row_sums():
    L = compute_laplacian(build_adjacency(A_FULL))
    assert is_laplacian(L)
    assert np.allclose(L.sum(axis=1), 0.0)
    assert np.allclose(L.sum(axis=0), 0.0)


def test_laplacian_off_diagonal_equals_negative_weight():
    L = compute_laplacian(build_adjacency(A_FULL))
    assert abs(L[0, 1] + 0.8) < 1e-12
    assert abs(L[2, 3] + 0.1) < 1e-12


def test_quadratic_form_zero_for_constant_signal():
    L = compute_laplacian(build_adjacency(A_FULL))
    f = np.ones(4)
    assert abs(quadratic_form(L, f)) < 1e-12


def test_quadratic_form_rough_signal_matches_guide_value():
    W_sparse = np.where(build_adjacency(A_FULL) > 0.5, build_adjacency(A_FULL), 0.0)
    L_sparse = compute_laplacian(W_sparse)
    f = np.array([1.0, 1.0, 5.0, 1.0])
    assert abs(quadratic_form(L_sparse, f) - 24.0) < 1e-6


def test_thresholded_graph_isolates_customer():
    W_sparse = np.where(build_adjacency(A_FULL) > 0.5, build_adjacency(A_FULL), 0.0)
    assert np.allclose(W_sparse[3, :], 0.0)
    L_sparse = compute_laplacian(W_sparse)
    evals = np.sort(np.linalg.eigvalsh(L_sparse))
    assert np.sum(np.abs(evals) < 1e-8) == 2
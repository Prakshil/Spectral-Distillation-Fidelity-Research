"""Tests for the spectral routines on the 4-node ERP example."""

import numpy as np
import pytest

from spectral_distillation.src.laplacian import build_adjacency, compute_laplacian
from spectral_distillation.src.spectral import (
    compute_eigenvalues,
    compute_eigenvectors,
    count_zero_eigenvalues,
    eigenvector_orthonormality,
    fiedler_value,
    min_eigengap,
)

A_FULL = np.array(
    [
        [0.0, 0.8, 0.6, 0.3],
        [0.8, 0.0, 0.9, 0.2],
        [0.6, 0.9, 0.0, 0.1],
        [0.3, 0.2, 0.1, 0.0],
    ]
)


def test_spectrum_matches_numerically_verified_values():
    L = compute_laplacian(build_adjacency(A_FULL))
    e = compute_eigenvalues(L)
    assert np.allclose(e, [0.0, 0.781832, 2.266461, 2.751707], atol=1e-5)


def test_sparse_thresholding_collapses_fiedler():
    W_full = build_adjacency(A_FULL)
    W_sparse = np.where(W_full > 0.5, W_full, 0.0)
    e_full = compute_eigenvalues(compute_laplacian(W_full))
    e_sparse = compute_eigenvalues(compute_laplacian(W_sparse))
    assert fiedler_value(e_full) > 0.4
    assert fiedler_value(e_sparse) == pytest.approx(0.0, abs=1e-8)
    assert count_zero_eigenvalues(e_sparse) == 2


def test_eigenvectors_orthonormal():
    L = compute_laplacian(build_adjacency(A_FULL))
    _, V = compute_eigenvectors(L)
    assert eigenvector_orthonormality(V)


def test_eigengaps_are_positive_and_minimum_matches():
    L = compute_laplacian(build_adjacency(A_FULL))
    e = compute_eigenvalues(L)
    gaps = np.diff(e)
    assert np.all(gaps > 0)
    assert min_eigengap(e) == pytest.approx(0.485247, abs=1e-4)


def test_isolated_customer_yields_two_zero_eigenvalues():
    W_sparse = np.zeros((4, 4))
    W_sparse[0, 1] = W_sparse[1, 0] = 0.8
    W_sparse[0, 2] = W_sparse[2, 0] = 0.6
    W_sparse[1, 2] = W_sparse[2, 1] = 0.9
    e = compute_eigenvalues(compute_laplacian(W_sparse))
    assert count_zero_eigenvalues(e) == 2
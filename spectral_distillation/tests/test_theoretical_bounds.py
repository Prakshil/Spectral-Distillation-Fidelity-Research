"""Empirical verification of Weyl, Davis-Kahan, and Cauchy interlacing bounds."""

import numpy as np
import pytest

from spectral_distillation.src.distortion import compute_frobenius_norm
from spectral_distillation.src.laplacian import build_adjacency, compute_laplacian
from spectral_distillation.src.spectral import compute_eigenvalues, min_eigengap
from spectral_distillation.src.theoretical_bounds import (
    check_all_bounds,
    check_cauchy_interlacing,
    check_davis_kahan,
    check_weyl,
)

A_FULL = np.array(
    [
        [0.0, 0.8, 0.6, 0.3],
        [0.8, 0.0, 0.9, 0.2],
        [0.6, 0.9, 0.0, 0.1],
        [0.3, 0.2, 0.1, 0.0],
    ]
)

W_FULL = build_adjacency(A_FULL)
L_FULL = compute_laplacian(W_FULL)


def test_weyl_holds_on_thresholded_example():
    W_sparse = np.where(W_FULL > 0.5, W_FULL, 0.0)
    L_sparse = compute_laplacian(W_sparse)
    result = check_weyl(L_FULL, L_sparse)
    assert result["holds"]
    assert result["max_shift"] <= result["frobenius_norm"] + 1e-9


def test_weyl_max_shift_matches_fiedler_collapse():
    W_sparse = np.where(W_FULL > 0.5, W_FULL, 0.0)
    L_sparse = compute_laplacian(W_sparse)
    result = check_weyl(L_FULL, L_sparse)
    e_full = compute_eigenvalues(L_FULL)
    e_sparse = compute_eigenvalues(L_sparse)
    assert result["max_shift"] == pytest.approx(abs(e_full[1] - e_sparse[1]), abs=1e-6)
    assert result["max_shift"] == pytest.approx(e_full[1], abs=1e-6)


def test_davis_kahan_holds_under_mild_perturbation():
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 0.001, size=(4, 4))
    noise = 0.5 * (noise + noise.T)
    L_perturbed = L_FULL + noise
    result = check_davis_kahan(L_FULL, L_perturbed)
    assert result["holds"]


def test_davis_kahan_bound_vacuous_for_fiedler_collapse():
    W_sparse = np.where(W_FULL > 0.5, W_FULL, 0.0)
    L_sparse = compute_laplacian(W_sparse)
    frob = compute_frobenius_norm(L_FULL, L_sparse)
    gap = min_eigengap(compute_eigenvalues(L_FULL))
    assert 2.0 * frob / gap > 1.0


def test_cauchy_interlacing_holds_for_every_edge_of_full_graph():
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    for i, j in edges:
        result = check_cauchy_interlacing(L_FULL, i, j)
        assert result["holds"]
        assert result["max_decrease"] <= 2.0 * result["edge_weight"] + 1e-8


def test_cauchy_eigenvalues_only_decrease():
    W_sparse = np.where(W_FULL > 0.5, W_FULL, 0.0)
    L_sparse = compute_laplacian(W_sparse)
    e_full = compute_eigenvalues(L_FULL)
    e_sparse = compute_eigenvalues(L_sparse)
    assert np.all(np.linalg.eigvalsh(L_sparse) <= np.linalg.eigvalsh(L_FULL) + 1e-8)
    assert e_sparse[1] <= e_full[1] + 1e-8


def test_check_all_bounds_runs():
    result = check_all_bounds(W_FULL, tau=0.5)
    assert result["weyl"]["holds"]
    assert "per_k_sin_theta" in result["davis_kahan"]


def test_davis_kahan_degeneracy_detected_on_full_components():
    n = 10
    W = np.zeros((n, n))
    for block in (range(5), range(5, 10)):
        for i in block:
            for j in block:
                if i != j:
                    W[i, j] = 0.2
    L = compute_laplacian(W)
    result = check_davis_kahan(L, L)
    assert result["holds"]
"""Tests for per-node spectral position signals."""

import numpy as np
import pytest

from spectral_distillation.src.laplacian import build_adjacency, compute_laplacian
from spectral_distillation.src.spectral import compute_eigenvectors
from spectral_distillation.src.spectral_position import (
    high_frequency_share,
    node_spectral_energy,
    node_spectral_position,
    spectral_ratio,
)

A_FULL = np.array(
    [
        [0.0, 0.8, 0.6, 0.3],
        [0.8, 0.0, 0.9, 0.2],
        [0.6, 0.9, 0.0, 0.1],
        [0.3, 0.2, 0.1, 0.0],
    ]
)


def test_energy_equals_degree_for_orthonormal_basis():
    L = compute_laplacian(build_adjacency(A_FULL))
    evals, V = compute_eigenvectors(L)
    energy = node_spectral_energy(V, evals)
    degrees = np.diag(L)
    assert np.allclose(energy, degrees, atol=1e-8)


def test_position_is_convex_combination_of_eigenvalues():
    L = compute_laplacian(build_adjacency(A_FULL))
    evals, V = compute_eigenvectors(L)
    position = node_spectral_position(V, evals)
    assert position.min() >= -1e-12
    assert position.max() <= evals.max() + 1e-12
    assert position.mean() == pytest.approx(evals.mean(), abs=1e-6)


def test_high_frequency_share_in_unit_interval():
    L = compute_laplacian(build_adjacency(A_FULL))
    evals, V = compute_eigenvectors(L)
    share = high_frequency_share(V, evals, band_start=2)
    assert np.all(share >= -1e-12)
    assert np.all(share <= 1.0 + 1e-12)


def test_spectral_ratio_normalizes_to_unit_max():
    x = np.array([0.0, 0.5, 2.0])
    ratio = spectral_ratio(x)
    assert np.allclose(ratio, [0.0, 0.25, 1.0])
    assert ratio.max() == pytest.approx(1.0)
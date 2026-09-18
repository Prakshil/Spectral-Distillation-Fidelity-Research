"""Tests for the spectral distortion diagnostic (golden 4-node values)."""

import numpy as np
import pytest

from spectral_distillation.src.baselines import threshold_sparsify
from spectral_distillation.src.distortion import (
    compute_fragility,
    compute_frobenius_norm,
    compute_spectral_distortion,
    davis_kahan_bound,
    predict_retention,
    spectral_distortion_report,
)
from spectral_distillation.src.laplacian import build_adjacency, compute_laplacian
from spectral_distillation.src.spectral import min_eigengap

A_FULL = np.array(
    [
        [0.0, 0.8, 0.6, 0.3],
        [0.8, 0.0, 0.9, 0.2],
        [0.6, 0.9, 0.0, 0.1],
        [0.3, 0.2, 0.1, 0.0],
    ]
)

W_FULL = build_adjacency(A_FULL)
W_SPARSE = threshold_sparsify(W_FULL, 0.5)
L_FULL = compute_laplacian(W_FULL)
L_SPARSE = compute_laplacian(W_SPARSE)


def test_spectral_distortion_of_thresholding_is_one():
    sd = compute_spectral_distortion(L_FULL, L_SPARSE)
    assert sd == pytest.approx(1.0, abs=1e-4)


def test_frobenius_norm_matches_guide():
    frob = compute_frobenius_norm(L_FULL, L_SPARSE)
    assert frob == pytest.approx(0.883176, abs=1e-4)
    assert frob ** 2 == pytest.approx(0.78, abs=1e-4)


def test_min_eigengap_and_fragility():
    gap = min_eigengap(np.sort(np.linalg.eigvalsh(L_FULL)))
    assert gap == pytest.approx(0.485247, abs=1e-4)
    fragility = compute_fragility(1.0, gap)
    assert fragility == pytest.approx(2.060808, abs=1e-2)


def test_retention_and_davis_kahan_bound():
    assert predict_retention(1.0) == pytest.approx(0.0)
    assert predict_retention(0.0) == pytest.approx(1.0)
    assert predict_retention(0.3) == pytest.approx(0.7)


def test_davis_kahan_bound_2_is_vacuous_under_fiedler_collapse():
    frob = compute_frobenius_norm(L_FULL, L_SPARSE)
    gap_2 = min_eigengap(np.sort(np.linalg.eigvalsh(L_FULL)))
    bound_2 = davis_kahan_bound(frob, gap_2)
    assert bound_2 > 1.0


def test_report_contains_all_fields():
    report = spectral_distortion_report(L_FULL, L_SPARSE)
    for key in (
        "SD",
        "min_eigengap",
        "frobenius_norm",
        "davis_kahan_bound",
        "predicted_retention",
        "fragility_score",
        "eigenvalues_dense",
        "eigenvalues_sparse",
        "relative_errors",
    ):
        assert key in report
    assert report["relative_errors"][0] == 0.0
    assert report["relative_errors"][1] == pytest.approx(1.0, abs=1e-3)
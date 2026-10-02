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
    rank_matched_distortion,
    spectral_distortion_profile,
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

# --- real-measurement support: cached reference + rank-matched metric --------


def _path_plus_isolated(n=140):
    """A connected path of n-40 nodes plus 40 isolated nodes (41 components)."""
    W = np.zeros((n, n))
    for i in range(n - 41):
        W[i, i + 1] = W[i + 1, i] = 1.0
    return W


def test_spectral_distortion_accepts_cached_reference():
    """A cached reference spectrum must give bit-identical SD."""
    W = _path_plus_isolated()
    L_full = compute_laplacian(W)
    L_sparse = compute_laplacian(W.copy())
    direct = compute_spectral_distortion(L_full, L_sparse)
    cached = compute_spectral_distortion(
        L_full, L_sparse, ref_eigenvalues=np.sort(np.linalg.eigvalsh(L_full))
    )
    assert direct == pytest.approx(cached, rel=1e-12)


def test_profile_matches_scalar_and_validates_cache():
    W = _path_plus_isolated()
    L_full = compute_laplacian(W)
    L_sparse = compute_laplacian(W.copy())
    ref = np.sort(np.linalg.eigvalsh(L_full))
    prof = spectral_distortion_profile(None, L_sparse, ref_eigenvalues=ref)
    assert prof["SD"] == pytest.approx(
        compute_spectral_distortion(L_full, L_sparse), rel=1e-12
    )
    # A cache from a different graph must be rejected, not silently mismatched.
    with pytest.raises(ValueError, match="different graph"):
        spectral_distortion_profile(None, L_sparse, ref_eigenvalues=np.zeros(7))
    with pytest.raises(ValueError):
        spectral_distortion_profile(None, L_sparse)


def test_legacy_indexwise_saturates_when_components_change():
    """Documents the defect the rank-matched metric exists to fix."""
    W = _path_plus_isolated()
    L_full = compute_laplacian(W)
    # Fragment into many components: lambda_max is set by the longest path
    # block in both graphs, but the number of zero eigenvalues explodes.
    W_frag = np.zeros_like(W)
    W_frag[:12, :12] = np.eye(12, k=1) + np.eye(12, k=-1)
    L_frag = compute_laplacian(W_frag)
    legacy = compute_spectral_distortion(L_full, L_frag)
    rm = rank_matched_distortion(
        None, L_frag, ref_eigenvalues=np.sort(np.linalg.eigvalsh(L_full))
    )
    assert legacy == pytest.approx(1.0, abs=1e-6)
    # The rank-matched metric sees only a small lambda_max change, not 100%.
    assert rm["lambda_max_rel_error"] < 0.10


def test_rank_matched_stays_finite_and_small_when_graph_fragments():
    W = _path_plus_isolated(n=140)
    W_frag = np.zeros_like(W)
    W_frag[:12, :12] = np.eye(12, k=1) + np.eye(12, k=-1)
    L_full = compute_laplacian(W)
    L_frag = compute_laplacian(W_frag)
    ref = np.sort(np.linalg.eigvalsh(L_full))
    rm = rank_matched_distortion(None, L_frag, ref_eigenvalues=ref)
    # Well defined and finite even though the legacy index-wise metric is
    # saturated at 1.0.
    assert np.isfinite(rm["lambda_max_rel_error"])
    assert rm["lambda_max_rel_error"] < 1.0
    assert rm["top1_rel_error"] == pytest.approx(rm["lambda_max_rel_error"])
    # top1 <= top8 <= top32 <= top128 (nested prefixes cannot shrink the max).
    assert (
        rm["top1_rel_error"]
        <= rm["top8_rel_error"]
        <= rm["top32_rel_error"]
        <= rm["top128_rel_error"]
    )


def test_rank_matched_is_zero_for_identical_graphs():
    W = _path_plus_isolated()
    L = compute_laplacian(W)
    ref = np.sort(np.linalg.eigvalsh(L))
    rm = rank_matched_distortion(None, L, ref_eigenvalues=ref)
    assert rm["lambda_max_rel_error"] == pytest.approx(0.0, abs=1e-9)
    assert rm["top32_rel_error"] == pytest.approx(0.0, abs=1e-9)


def test_rank_matched_preserves_ordering():
    """A single edge drop must increase lambda_max error monotonically."""
    W = np.zeros((40, 40))
    for i in range(39):
        W[i, i + 1] = W[i + 1, i] = 1.0
    ref = np.sort(np.linalg.eigvalsh(compute_laplacian(W)))
    errs = []
    for n_drop in (0, 2, 6, 12):
        Wd = W.copy()
        Wd[5 : 5 + n_drop, 5 : 5 + n_drop] = 0.0
        Wd[4, 5 + n_drop - 1] = Wd[4, 5 + n_drop - 1] = 0.0
        rm = rank_matched_distortion(None, compute_laplacian(Wd), ref_eigenvalues=ref)
        errs.append(rm["lambda_max_rel_error"])
    assert errs == sorted(errs), errs

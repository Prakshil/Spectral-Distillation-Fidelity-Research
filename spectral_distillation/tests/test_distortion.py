"""Tests for the spectral distortion diagnostic (golden 4-node values)."""

import numpy as np
import pytest

from spectral_distillation.src.baselines import threshold_sparsify
from spectral_distillation.src.distortion import (
    compute_fragility,
    compute_frobenius_norm,
    compute_spectral_distortion,
    connectivity_report,
    davis_kahan_bound,
    low_frequency_distortion,
    predict_retention,
    rank_matched_distortion,
    routing_survival,
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


def test_low_frequency_is_zero_for_identical_graphs():
    W = _path_plus_isolated()
    L = compute_laplacian(W)
    ref = np.sort(np.linalg.eigvalsh(L))
    lf = low_frequency_distortion(None, L, ref_eigenvalues=ref)
    assert lf["low_freq_k"] > 0
    assert lf["low_freq_max_error"] == pytest.approx(0.0, abs=1e-9)
    assert lf["low_freq_mean_error"] == pytest.approx(0.0, abs=1e-9)


def test_low_frequency_discriminates_where_legacy_saturates():
    """The defect this metric exists to fix, as a hard acceptance test.

    Index-matched pairing pins at exactly 1.0 for every mode whose reference
    eigenvalue is above ``eps`` and whose perturbed partner is 0, so the legacy
    scalar carried no information across the real retention ladder. Dropping the
    zero block on each side must restore a graded value.
    """
    W = _path_plus_isolated(n=140)
    L_full = compute_laplacian(W)
    ref = np.sort(np.linalg.eigvalsh(L_full))
    W_frag = np.zeros_like(W)
    W_frag[:12, :12] = np.eye(12, k=1) + np.eye(12, k=-1)
    L_frag = compute_laplacian(W_frag)

    legacy = compute_spectral_distortion(L_full, L_frag)
    lf = low_frequency_distortion(None, L_frag, ref_eigenvalues=ref)
    conn = connectivity_report(None, L_frag, ref_eigenvalues=ref)

    assert legacy == pytest.approx(1.0, abs=1e-6)
    # Graded: strictly inside (0, 1) rather than pinned at a rail.
    assert 0.0 < lf["low_freq_mean_error"] < 1.0
    # Fragmentation is accounted for separately instead of masquerading as error.
    assert conn["n_isolated_perturbed"] > 0
    assert conn["n_components_perturbed"] > conn["n_components_ref"]
    # The surviving modes are matched on both sides despite 129 zero eigenvalues.
    assert lf["n_matched_modes_ref"] > 0
    assert lf["n_matched_modes_perturbed"] > 0


def test_low_frequency_increases_monotonically_with_dropped_edges():
    W = np.zeros((60, 60))
    for i in range(59):
        W[i, i + 1] = W[i + 1, i] = 1.0
    ref = np.sort(np.linalg.eigvalsh(compute_laplacian(W)))
    errs = []
    for cut in (0, 4, 10, 18):
        Wd = W.copy()
        if cut:
            Wd[20 - cut : 20, 20 - cut : 20] = 0.0
            Wd[19, 20] = Wd[20, 19] = 0.0
        lf = low_frequency_distortion(None, compute_laplacian(Wd), ref_eigenvalues=ref)
        errs.append(lf["low_freq_mean_error"])
    assert errs == sorted(errs), errs


def test_low_frequency_rejects_mismatched_cached_reference():
    W = _path_plus_isolated()
    L = compute_laplacian(W)
    with pytest.raises(ValueError, match="eigenvalues"):
        low_frequency_distortion(None, L, ref_eigenvalues=np.zeros(7))


def test_low_frequency_handles_fully_isolated_perturbed_graph():
    W = _path_plus_isolated(n=40)
    L_full = compute_laplacian(W)
    ref = np.sort(np.linalg.eigvalsh(L_full))
    L_none = compute_laplacian(np.zeros((40, 40)))
    lf = low_frequency_distortion(None, L_none, ref_eigenvalues=ref)
    assert lf["low_freq_k"] == 0
    assert lf["low_freq_mean_error"] == pytest.approx(1.0)


def test_connectivity_report_counts_isolated_mass():
    W = _path_plus_isolated(n=50)
    L_full = compute_laplacian(W)
    ref = np.sort(np.linalg.eigvalsh(L_full))
    conn = connectivity_report(None, L_full, ref_eigenvalues=ref)
    assert conn["n_isolated_perturbed"] == 0
    assert conn["n_components_perturbed"] == conn["n_components_ref"]
    assert conn["largest_perturbed"] == pytest.approx(conn["largest_ref"])


def test_routing_survival_is_not_pinned_by_saturated_sd():
    """``predict_retention(SD)`` reads 0.0 whenever SD saturates at 1.0.

    That is why the four-node diagnostic predicted ~1.3e-8 retention. The
    replacement must stay graded and must not collapse when the legacy scalar
    saturates.
    """
    assert predict_retention(1.0) == pytest.approx(0.0)

    # Low spectral error, no isolated mass -> survival tracks the spectral error.
    assert routing_survival(0.10, 0, 1000) == pytest.approx(0.90, abs=1e-9)
    # Total disconnection must not be excused by a pristine spectrum: the two
    # penalties multiply, so 90% orphaned drives survival to ~0 even when the
    # smooth modes are intact.
    assert routing_survival(0.10, 900, 1000) == pytest.approx(0.09, abs=1e-9)
    # Total loss on both terms -> 0, never negative.
    assert routing_survival(1.0, 1000, 1000) == pytest.approx(0.0, abs=1e-9)
    # Monotone decreasing in spectral error.
    vals = [routing_survival(e, 0, 1000) for e in (0.1, 0.3, 0.5, 0.9)]
    assert vals == sorted(vals, reverse=True)
    assert all(0.0 <= v <= 1.0 for v in vals)
    # Monotone decreasing in isolated mass.
    iso = [routing_survival(0.2, i, 1000) for i in (0, 250, 500, 750)]
    assert iso == sorted(iso, reverse=True)


def test_low_frequency_error_is_normalised_by_graph_scale():
    """Per-mode normalisation explodes on the smoothest modes.

    The smooth modes have reference eigenvalues arbitrarily close to zero, so
    dividing by the mode itself turns a negligible absolute shift into an error
    of hundreds. Normalising by ``lambda_max`` bounds the value and makes graphs
    of different sizes directly comparable.
    """
    W = _path_plus_isolated(n=140)
    L_full = compute_laplacian(W)
    ref = np.sort(np.linalg.eigvalsh(L_full))
    W_frag = np.zeros_like(W)
    W_frag[:12, :12] = np.eye(12, k=1) + np.eye(12, k=-1)
    lf = low_frequency_distortion(None, compute_laplacian(W_frag), ref_eigenvalues=ref)
    assert lf["normalisation"] == pytest.approx(float(ref[-1]))
    assert 0.0 < lf["low_freq_mean_error"] < 1.0
    assert lf["low_freq_max_error"] < 1.0

"""Tests for the method-comparison robustness bands (Phase 6, workstream 1)."""

import numpy as np
import pytest

from spectral_distillation.experiments.compare_methods import (
    BAND_METRICS,
    METHODS,
    _method_metric,
    check_bands,
    evaluate,
)
from spectral_distillation.src.synthetic import make_erp_attention

N = 48
BUDGET = 2 * N


def _band(n, factor, eps, method, low_freq):
    return {
        "n": n,
        "budget_factor": factor,
        "epsilon": eps,
        "method": method,
        "low_freq_mean": low_freq,
    }


def test_evaluate_returns_all_methods_and_metrics():
    W = make_erp_attention(N, seed=0)["attention"]
    row = evaluate(W, BUDGET, seed=0, epsilon=0.1, mass_match=True)
    assert row["mass_match"] is True
    for method in METHODS:
        for metric in BAND_METRICS:
            assert np.isfinite(_method_metric(row, method, metric))
        assert 0.0 < row[f"{method}_mass_ratio"] <= 1.0
        assert row[f"{method}_edges"] > 0


def test_mass_match_changes_low_frequency_distortion():
    W = make_erp_attention(N, seed=0)["attention"]
    matched = evaluate(W, BUDGET, seed=0, epsilon=0.1, mass_match=True)
    raw = evaluate(W, BUDGET, seed=0, epsilon=0.1, mass_match=False)
    assert raw["spectral_mass_ratio"] < 1.0
    assert raw["spectral_lowfreq"] != matched["spectral_lowfreq"]


def test_mass_match_transform_equalizes_total_edge_weight():
    from spectral_distillation.src.sparsifier import spectral_sparsify

    W = make_erp_attention(N, seed=0)["attention"]
    tw_full = float(np.triu(W, 1).sum())
    W_sp = spectral_sparsify(W, BUDGET, epsilon=0.1)
    tw_sp = float(np.triu(W_sp, 1).sum())
    assert tw_sp < tw_full
    W_matched = W_sp * (tw_full / tw_sp)
    assert float(np.triu(W_matched, 1).sum()) == pytest.approx(tw_full, rel=1e-9)


def test_method_metric_rejects_unknown_metric():
    W = make_erp_attention(N, seed=0)["attention"]
    row = evaluate(W, BUDGET, seed=0, epsilon=0.1)
    with pytest.raises(ValueError):
        _method_metric(row, "spectral", "not_a_metric")


def test_check_bands_detects_universal_winner():
    bands = []
    for n in (32, 64):
        for f in (2.0, 4.0):
            bands.append(_band(n, f, 0.1, "spectral", 0.01))
            bands.append(_band(n, f, 0.1, "threshold", 0.30))
            bands.append(_band(n, f, 0.1, "random", 0.40))
            bands.append(_band(n, f, 0.1, "degree", 0.50))
    out = check_bands(bands, metric="low_freq")
    assert out["n_points"] == 4
    assert out["n_below_all_baselines"] == 4
    assert out["holds_everywhere"] is True


def test_check_bands_reports_failures():
    bands = [
        _band(32, 2.0, 0.1, "spectral", 0.50),
        _band(32, 2.0, 0.1, "threshold", 0.30),
        _band(32, 2.0, 0.1, "random", 0.40),
        _band(32, 2.0, 0.1, "degree", 0.10),
    ]
    out = check_bands(bands, metric="low_freq")
    assert out["n_points"] == 1
    assert out["n_below_all_baselines"] == 0
    assert out["holds_everywhere"] is False
    assert out["points"][0]["below_degree"] is False


def test_band_metrics_and_methods_are_consistent():
    assert set(METHODS) == {"threshold", "random", "degree", "spectral"}
    assert "low_freq" in BAND_METRICS

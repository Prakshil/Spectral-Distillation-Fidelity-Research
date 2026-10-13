"""Tests for the partial-correlation / bootstrap helpers used to separate the
spectral-distortion signal from the node-isolation confound in the SD-vs-routing
analysis (``analyze_sd_vs_routing.py``)."""

from __future__ import annotations

import numpy as np
from scipy import stats

from spectral_distillation.src.evaluation import (
    bootstrap_spearman_ci,
    partial_spearman,
)


def test_partial_spearman_removes_common_driver():
    # x and y are both driven by z, so their raw Spearman is high but the
    # partial correlation controlling for z should collapse toward zero.
    rng = np.random.default_rng(0)
    z = rng.normal(size=300)
    x = z + 0.02 * rng.normal(size=300)
    y = z + 0.02 * rng.normal(size=300)

    raw = stats.spearmanr(x, y).statistic
    r, p = partial_spearman(x, y, z)

    assert raw > 0.8
    assert abs(r) < 0.25
    assert p > 0.05


def test_partial_spearman_retains_direct_signal():
    # y = x + z with x independent of z: controlling for z must keep a strong
    # positive partial correlation.
    rng = np.random.default_rng(1)
    x = rng.normal(size=300)
    z = rng.normal(size=300)
    y = x + z

    r, p = partial_spearman(x, y, z)
    assert r > 0.3
    assert p < 0.01


def test_partial_spearman_matches_spearman_when_control_is_noise():
    rng = np.random.default_rng(2)
    x = rng.normal(size=300)
    y = x + 0.5 * rng.normal(size=300)
    z = rng.normal(size=300)

    raw = stats.spearmanr(x, y).statistic
    r, _ = partial_spearman(x, y, z)
    assert abs(r - raw) < 0.1


def test_partial_spearman_degenerate_inputs():
    assert np.isnan(partial_spearman([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])[0])
    # a constant control leaves the ordinary correlation intact
    r, _ = partial_spearman([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [5.0, 5.0, 5.0, 5.0])
    assert r == 1.0


def test_bootstrap_spearman_ci_signal_and_null():
    rng = np.random.default_rng(3)
    x = rng.normal(size=120)
    y = 2.0 * x + rng.normal(size=120)

    lo, hi, pt = bootstrap_spearman_ci(x, y, n_boot=500, seed=0)
    assert lo < pt < hi
    assert lo > 0.5

    x2 = rng.normal(size=120)
    y2 = rng.normal(size=120)
    lo2, hi2, _ = bootstrap_spearman_ci(x2, y2, n_boot=500, seed=0)
    assert lo2 < 0.0 < hi2

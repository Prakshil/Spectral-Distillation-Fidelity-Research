"""Tests for the top-k effective-resistance scalability study (Phase 6 WS2)."""

from types import SimpleNamespace

import numpy as np

from spectral_distillation.experiments.scalability_reeff import (
    _rank_correlation,
    _rel_errors,
    edge_resistance_from_scaled,
    run_graph,
    topk_scaled,
    vectorized_sbm,
)
from spectral_distillation.src.effective_resistance import effective_resistance_exact
from spectral_distillation.src.laplacian import compute_laplacian, edges_from_weights
from spectral_distillation.src.sparsifier import spectral_sparsify
from spectral_distillation.src.utils import get_logger


def test_vectorized_sbm_is_symmetric_and_block_sized():
    W, block = vectorized_sbm(60, n_blocks=3, p_in=0.4, p_out=0.2, seed=0, chunk=16)
    assert W.shape == (60, 60)
    assert np.allclose(W, W.T)
    assert np.allclose(np.diag(W), 0.0)
    assert sorted(np.bincount(block).tolist()) == [20, 20, 20]


def test_vectorized_sbm_p_out_zero_is_block_diagonal():
    W, block = vectorized_sbm(40, n_blocks=4, p_in=0.9, p_out=0.0, seed=1, chunk=8)
    rows, cols = np.nonzero(W)
    assert np.all(block[rows] == block[cols])


def test_edge_resistance_matches_exact_at_full_rank():
    W, _ = vectorized_sbm(50, n_blocks=2, p_in=0.6, p_out=0.05, seed=2)
    L = compute_laplacian(W)
    rows, cols, _ = edges_from_weights(W)
    scaled = topk_scaled(L, 49)
    r_k = edge_resistance_from_scaled(scaled, rows, cols)
    r_exact = effective_resistance_exact(L)[rows, cols]
    assert np.allclose(r_k, r_exact, atol=1e-6)


def test_edge_resistance_chunking_matches_single_pass():
    scaled = np.random.default_rng(0).normal(size=(200, 8))
    rows = np.random.default_rng(1).integers(0, 200, size=5000)
    cols = np.random.default_rng(2).integers(0, 200, size=5000)
    full = edge_resistance_from_scaled(scaled, rows, cols, chunk=5000)
    chunked = edge_resistance_from_scaled(scaled, rows, cols, chunk=128)
    assert np.allclose(full, chunked)


def test_rank_correlation_perfect_for_identical_and_monotone():
    ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.isclose(_rank_correlation(ref, ref), 1.0)
    assert np.isclose(_rank_correlation(ref ** 2, ref), 1.0)


def test_rel_errors_zero_for_identical_arrays():
    ref = np.array([0.1, 0.2, 0.3])
    out = _rel_errors(ref, ref)
    assert out["rel_frobenius"] == 0.0
    assert out["mean_rel"] == 0.0
    assert out["max_rel"] == 0.0


def test_rel_errors_detects_scaled_error():
    ref = np.array([1.0, 1.0, 1.0, 1.0])
    est = np.array([1.0, 1.0, 1.0, 2.0])
    out = _rel_errors(est, ref)
    assert out["rel_frobenius"] > 0
    assert out["max_rel"] == 1.0


def test_sparsifier_resistance_edges_matches_matrix_path():
    W, _ = vectorized_sbm(60, n_blocks=3, p_in=0.6, p_out=0.05, seed=3)
    L = compute_laplacian(W)
    rows, cols, _ = edges_from_weights(W)
    R = effective_resistance_exact(L)
    r_edges = R[rows, cols]
    budget = 4 * 60
    W_matrix = spectral_sparsify(W, budget, resistance_matrix=R, epsilon=0.1)
    W_edges = spectral_sparsify(W, budget, resistance_edges=r_edges, epsilon=0.1)
    assert np.allclose(W_matrix, W_edges)


def test_sparsifier_rejects_wrong_length_resistance_edges():
    W, _ = vectorized_sbm(40, n_blocks=2, p_in=0.6, p_out=0.05, seed=4)
    rows, _, _ = edges_from_weights(W)
    with np.testing.assert_raises(ValueError):
        spectral_sparsify(W, 20, resistance_edges=np.ones(rows.size - 1))


def test_run_graph_emits_expected_keys_and_positive_errors():
    args = SimpleNamespace(
        n_blocks=3,
        p_in=0.6,
        p_out=0.05,
        seed=0,
        budget_factor=4,
        exact_n_max=100,
        sd_n_max=100,
        epsilon=0.1,
        ks=[8, 16],
    )
    rows = run_graph(80, args, get_logger("test_scalability"))
    assert len(rows) == 2
    row = rows[0]
    for key in (
        "n",
        "k",
        "seconds_topk",
        "edge_rel_frobenius",
        "low_freq_mean_error",
        "low_freq_rel_vs_exact",
        "rel_to_kmax",
    ):
        assert key in row
    assert row["seconds_exact"] is not None
    assert row["edge_rel_frobenius"] > 0.0
    assert row["low_freq_rel_vs_exact"] >= 0.0


def test_run_graph_skips_exact_beyond_max():
    args = SimpleNamespace(
        n_blocks=3,
        p_in=0.6,
        p_out=0.05,
        seed=0,
        budget_factor=2,
        exact_n_max=10,
        sd_n_max=10,
        epsilon=0.1,
        ks=[8, 16],
    )
    rows = run_graph(80, args, get_logger("test_scalability_skip"))
    assert len(rows) == 2
    assert all(row["seconds_exact"] is None for row in rows)
    assert all("edge_rel_frobenius" not in row for row in rows)
    assert all(row["rel_to_kmax"] is not None for row in rows)

"""Tests for the real-world fraud graph loader (DGL Amazon) and its protocol wiring."""

import numpy as np
import pytest

from spectral_distillation.src.real_fraud import (
    DEFAULT_AMAZON_MAT,
    load_amazon_fraud,
)
from spectral_distillation.src.router_protocol import (
    label_free_assignment,
    oracle_bucket_assignment,
    random_assignment,
    uniform_assignment,
)

AMAZON_AVAILABLE = __import__("pathlib").Path(DEFAULT_AMAZON_MAT).exists()


def _skip_if_missing():
    if not AMAZON_AVAILABLE:
        pytest.skip("Amazon.mat not downloaded (data/benchmarks/amazon/raw)")


def test_loader_schema():
    _skip_if_missing()
    g = load_amazon_fraud()
    for key in ("W", "features", "y", "n_nodes", "d_features", "source",
                "positive_ratio", "n_edges"):
        assert key in g
    assert g["W"].shape == (g["n_nodes"], g["n_nodes"])
    assert g["features"].shape == (g["n_nodes"], g["d_features"])
    assert g["d_features"] == 25


def test_loader_labels_match_dgl_block():
    _skip_if_missing()
    g = load_amazon_fraud()
    assert set(np.unique(g["y"]).tolist()) == {0, 1}
    assert 0.09 < g["positive_ratio"] < 0.10
    assert int(g["y"].sum()) == 821


def test_adjacency_symmetric_diagonal_free():
    _skip_if_missing()
    W = load_amazon_fraud()["W"]
    assert np.allclose(W, W.T)
    assert np.count_nonzero(np.diag(W)) == 0
    assert np.count_nonzero(W) > 0


def test_single_connected_component():
    _skip_if_missing()
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    W = load_amazon_fraud()["W"]
    n_comp, _ = connected_components(sp.csr_matrix(W > 0), directed=False)
    assert n_comp == 1


def test_adaptive_oracle_populates_three_channels():
    _skip_if_missing()
    g = load_amazon_fraud()
    fixed = oracle_bucket_assignment(g["W"], g["y"], adaptive=False)
    adaptive = oracle_bucket_assignment(g["W"], g["y"], adaptive=True)
    fix_usage = np.bincount(np.argmax(fixed, axis=1), minlength=3)
    ada_usage = np.bincount(np.argmax(adaptive, axis=1), minlength=3)
    assert (ada_usage > 0).all()
    assert (fix_usage > 0).all()


def test_condition_shapes_and_usage():
    _skip_if_missing()
    g = load_amazon_fraud()
    n = g["n_nodes"]
    oracle = oracle_bucket_assignment(g["W"], g["y"], adaptive=True)
    uniform = uniform_assignment(n, 3)
    random = random_assignment(oracle, rng=0)
    # label-free does a full eigendecomposition; on the full graph this is
    # heavy, so validate it on a small induced subgraph instead.
    idx = np.arange(0, 1200)
    sub = {"W": g["W"][idx][:, idx], "features": g["features"][idx], "y": g["y"][idx]}
    lf, _ = label_free_assignment(
        sub["W"], sub["features"], hidden_dim=16, n_layers=2, epochs=5,
        seed=0, device="cpu",
    )
    assert oracle.shape == (n, 3)
    assert uniform.shape == (n, 3)
    assert random.shape == (n, 3)
    assert lf.shape == (1200, 3)
    assert np.allclose(random.sum(axis=1), 1.0, atol=1e-9)
    assert np.allclose(uniform.sum(axis=1)[:2], 1.0, atol=1e-9)


def test_loader_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_amazon_fraud("data/benchmarks/amazon/raw/definitely_not_here.mat")
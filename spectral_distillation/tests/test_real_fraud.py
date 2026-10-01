"""Tests for the real-world fraud graph loaders (DGL Amazon, Tolokers, YelpChi)."""

import numpy as np
import pytest

from spectral_distillation.src.real_fraud import (
    DEFAULT_AMAZON_MAT,
    DEFAULT_TOLOKERS_DIR,
    DEFAULT_YELP_MAT,
    load_amazon_fraud,
    load_real_fraud,
    load_tolokers,
    load_yelpchi,
)
from spectral_distillation.src.router_protocol import (
    label_free_assignment,
    oracle_bucket_assignment,
    random_assignment,
    uniform_assignment,
)

import pathlib

AMAZON_AVAILABLE = pathlib.Path(DEFAULT_AMAZON_MAT).exists()
TOLOKERS_AVAILABLE = (
    pathlib.Path(DEFAULT_TOLOKERS_DIR) / "nodes.parquet"
).exists() and (pathlib.Path(DEFAULT_TOLOKERS_DIR) / "edges.parquet").exists()
YELP_AVAILABLE = pathlib.Path(DEFAULT_YELP_MAT).exists()


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


def test_registry_dispatch_and_unknown_key():
    _skip_if_missing()
    assert load_real_fraud("amazon")["n_nodes"] == load_amazon_fraud()["n_nodes"]
    assert load_real_fraud("AMAZON")["n_nodes"] == load_amazon_fraud()["n_nodes"]
    with pytest.raises(KeyError):
        load_real_fraud("not_a_dataset")


def _skip_if_missing_tolokers():
    if not TOLOKERS_AVAILABLE:
        pytest.skip("Tolokers parquet not downloaded (data/benchmarks/tolokers/raw)")


def test_tolokers_schema_and_labels():
    _skip_if_missing_tolokers()
    g = load_tolokers()
    assert g["W"].shape == (g["n_nodes"], g["n_nodes"]) == (11758, 11758)
    assert g["d_features"] == 10
    assert g["features"].shape == (11758, 10)
    assert set(np.unique(g["y"]).tolist()) == {0, 1}
    assert int(g["y"].sum()) == 2566
    assert 0.21 < g["positive_ratio"] < 0.23
    assert g["n_edges"] == 519000


def test_tolokers_adjacency_contract():
    _skip_if_missing_tolokers()
    W = load_tolokers()["W"]
    assert np.allclose(W, W.T)
    assert np.count_nonzero(np.diag(W)) == 0
    assert np.all((W == 0) | (W == 1))
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    n_comp, _ = connected_components(sp.csr_matrix(W > 0), directed=False)
    assert n_comp == 1


def test_tolokers_is_sparser_than_amazon():
    """The two regimes differ in density and node-level homophily spread, which
    is what the multi-dataset amplification claim relies on."""
    _skip_if_missing()
    _skip_if_missing_tolokers()
    a, t = load_amazon_fraud(), load_tolokers()

    def node_hom_sd(g):
        W, y = g["W"], g["y"]
        deg = W.sum(1)
        h = np.divide(W @ y, deg, out=np.zeros_like(deg), where=deg > 0)
        return float(h.std())

    assert t["n_nodes"] > a["n_nodes"]
    assert (t["n_edges"] / t["n_nodes"]) < (a["n_edges"] / a["n_nodes"])
    assert node_hom_sd(t) > 3 * node_hom_sd(a)


def test_tolokers_missing_dir_raises():
    with pytest.raises(FileNotFoundError):
        load_tolokers("data/benchmarks/tolokers/nope")


def _skip_if_missing_yelp():
    if not YELP_AVAILABLE:
        pytest.skip("YelpChi.mat not downloaded (data/benchmarks/yelp/raw)")


def test_yelpchi_schema_and_labels():
    _skip_if_missing_yelp()
    g = load_yelpchi()
    assert g["W"].shape == (g["n_nodes"], g["n_nodes"])
    assert g["d_features"] == 32
    assert g["features"].shape == (g["n_nodes"], 32)
    assert set(np.unique(g["y"]).tolist()) == {0, 1}
    assert set(np.unique(g["y"])) == {0, 1}
    # published positive ratio is 14.53%; the node-capped LCC is close
    assert 0.13 < g["positive_ratio"] < 0.16


def test_yelpchi_adjacency_contract_and_connectivity():
    _skip_if_missing_yelp()
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    W = load_yelpchi()["W"]
    assert np.allclose(W, W.T)
    assert np.count_nonzero(np.diag(W)) == 0
    assert np.all((W == 0) | (W == 1))
    n_comp, _ = connected_components(sp.csr_matrix(W > 0), directed=False)
    assert n_comp == 1, "loader must return a single connected component"


def test_yelpchi_respects_max_nodes():
    _skip_if_missing_yelp()
    g = load_yelpchi(max_nodes=3000)
    assert g["n_nodes"] <= 3000
    # connectivity still holds after the induced-sample + LCC filter
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    n_comp, _ = connected_components(sp.csr_matrix(g["W"] > 0), directed=False)
    assert n_comp == 1


def test_yelpchi_sampling_is_deterministic():
    _skip_if_missing_yelp()
    a = load_yelpchi(max_nodes=3000, seed=7)
    b = load_yelpchi(max_nodes=3000, seed=7)
    c = load_yelpchi(max_nodes=3000, seed=8)
    assert np.array_equal(a["y"], b["y"])
    assert not np.array_equal(a["y"], c["y"])


def test_yelpchi_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_yelpchi("data/benchmarks/yelp/raw/not_here.mat")


def test_max_nodes_rejected_for_uncapped_datasets():
    """A silently ignored --max-nodes would make a capped run look uncapped."""
    _skip_if_missing()
    _skip_if_missing_tolokers()
    with pytest.raises(ValueError):
        load_real_fraud("amazon", max_nodes=5000)
    with pytest.raises(ValueError):
        load_real_fraud("tolokers", max_nodes=5000)


def test_max_nodes_dispatches_to_yelpchi():
    _skip_if_missing_yelp()
    g = load_real_fraud("yelpchi", max_nodes=3000)
    assert g["n_nodes"] <= 3000


def test_registry_dispatch_includes_yelpchi():
    _skip_if_missing_yelp()
    assert load_real_fraud("yelpchi")["n_nodes"] == load_yelpchi()["n_nodes"]
    assert load_real_fraud("YELPCHI")["n_nodes"] == load_yelpchi()["n_nodes"]


def test_three_graphs_span_three_density_regimes():
    """Amazon/Tolokers/YelpChi differ in density and node-level homophily spread.
    This spread is what the D2 proxy argument depends on, so pin it down."""
    _skip_if_missing()
    _skip_if_missing_tolokers()
    _skip_if_missing_yelp()

    def node_hom_sd(g):
        W, y = g["W"], g["y"]
        deg = W.sum(1)
        h = np.divide(W @ y, deg, out=np.zeros_like(deg), where=deg > 0)
        return float(h.std())

    a, t, yp = load_amazon_fraud(), load_tolokers(), load_yelpchi()
    densities = [g["n_edges"] / g["n_nodes"] for g in (a, t, yp)]
    assert densities[0] > densities[1] > densities[2], densities
    sds = [node_hom_sd(g) for g in (a, t, yp)]
    assert sds[1] > sds[0]
    # the three graphs must not collapse onto one another
    assert len({round(d, 1) for d in densities}) == 3
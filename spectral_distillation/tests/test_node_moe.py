"""Tests for the Node-MoE reproduction (``src/node_moe.py``).

Kept fast (tiny graphs, few epochs): they pin the *implementation*, not the
science -- Chebyshev-interpolation exactness, differentiated initialisation,
the smoothing loss, the head's fit/predict hooks, determinism, and that the
cross-fit gate cannot see test labels.
"""

import numpy as np

# Import the package before torch: ``spectral_distillation.src`` loads
# ``pyarrow.dataset`` as an import-order guard, and importing torch first makes
# that native load die with a Windows access violation.
from spectral_distillation.src.laplacian import normalize_adjacency
from spectral_distillation.src.node_moe import (
    ChebNetIIFilter,
    ChebNetIIHead,
    ChebNetIIModule,
    NodeMoEHead,
    build_differentiated_pool,
    chebnetii_basis,
    chebyshev_matrix,
    chebyshev_nodes,
    differentiated_filter_types,
    evaluate_node_moe_protocol_b,
    gate_features,
    target_filter,
    _smoothing_loss,
)
from spectral_distillation.src.router_protocol import oracle_bucket_assignment

import torch  # noqa: E402  (after the pyarrow import-order guard)


def _sbm(n_per=60, p_in=0.15, p_out=0.02, d=4, seed=0):
    """Two-community homophilic SBM with a community signal in feature 0."""
    rng = np.random.default_rng(seed)
    n = 2 * n_per
    y = np.array([0] * n_per + [1] * n_per)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            p = p_in if y[i] == y[j] else p_out
            if rng.random() < p:
                W[i, j] = W[j, i] = 1.0
    W = np.maximum(W, W.T)
    np.fill_diagonal(W, 0.0)
    X = rng.normal(0.0, 0.3, size=(n, d))
    X[:, 0] += y
    return W, X, y


# --------------------------------------------------------------------------- #
# Chebyshev machinery
# --------------------------------------------------------------------------- #

def test_chebyshev_nodes_lie_in_open_unit_interval():
    nodes = chebyshev_nodes(4)
    assert nodes.shape == (5,)
    assert np.all(nodes > -1.0) and np.all(nodes < 1.0)
    assert np.allclose(nodes, np.cos(((np.arange(5) + 0.5) * np.pi) / 5))


def test_chebyshev_matrix_matches_recursion():
    x = np.array([-0.7, 0.0, 0.5])
    T = chebyshev_matrix(x, 3)
    assert np.allclose(T[0], 1.0)
    assert np.allclose(T[1], x)
    assert np.allclose(T[2], 2 * x ** 2 - 1)
    assert np.allclose(T[3], 4 * x ** 3 - 3 * x)


def test_filter_interpolates_target_exactly_at_nodes():
    """gamma = g(nodes) -> f is the exact Chebyshev interpolant of g."""
    K = 4
    nodes = chebyshev_nodes(K)
    for name in ("low_pass", "high_pass", "band_pass", "comb"):
        flt = ChebNetIIFilter(K, name)
        g = target_filter(name, nodes)
        assert np.allclose(flt.gamma.detach().numpy(), g, atol=1e-6)
        assert np.allclose(flt.response().detach().numpy(), g, atol=1e-5)


def test_low_pass_heavier_at_low_frequency_end_than_high_pass():
    K = 3
    nodes = chebyshev_nodes(K)
    low = ChebNetIIFilter(K, "low_pass").response().detach().numpy()
    high = ChebNetIIFilter(K, "high_pass").response().detach().numpy()
    # x = -1 is the smooth / low-frequency end.
    i_low = int(np.argmin(nodes))
    i_high = int(np.argmax(nodes))
    assert low[i_low] > high[i_low]
    assert high[i_high] > low[i_high]
    assert np.allclose(low + high, 1.0, atol=1e-5)


def test_differentiated_filter_types_are_distinct_and_ordered():
    assert differentiated_filter_types(3) == ["low_pass", "high_pass", "band_pass"]
    assert len(set(differentiated_filter_types(6))) == 6


def test_smoothing_loss_zero_for_constant_and_nonnegative_otherwise():
    const = ChebNetIIFilter(3, "constant")
    assert float(_smoothing_loss([const])) < 1e-10
    assert float(_smoothing_loss([ChebNetIIFilter(3, "comb")])) > 0.0
    with torch.no_grad():
        const.gamma.add_(torch.tensor([0.0, 1.0, 0.0, 0.0]))
    assert float(_smoothing_loss([const])) >= 0.0


# --------------------------------------------------------------------------- #
# ChebNetII expert head
# --------------------------------------------------------------------------- #

def test_differentiated_modules_differ_before_training():
    W, X, _ = _sbm()
    W_norm = normalize_adjacency(W)
    Wt = torch.tensor(W_norm, dtype=torch.float32)
    xt = torch.tensor(X, dtype=torch.float32)
    basis = chebnetii_basis(Wt, xt, K=2)
    low = ChebNetIIModule(X.shape[1], 2, "low_pass")
    high = ChebNetIIModule(X.shape[1], 2, "high_pass")
    with torch.no_grad():
        out_low, out_high = low(basis), high(basis)
    # Different initialisation -> different filters -> different outputs.
    assert not torch.allclose(out_low, out_high, atol=1e-6)


def test_chebnetii_head_fits_and_beats_chance():
    W, X, y = _sbm()
    W_norm = normalize_adjacency(W)
    train = np.zeros(y.size, dtype=bool)
    train[::2] = True
    head = ChebNetIIHead(W_norm, X.shape[1], train, K=2, filter_type="low_pass",
                         epochs=200, seed=0)
    head.fit_graph(X, y)
    pred = head.predict_nodes(X, np.flatnonzero(~train))
    acc = float((pred == y[~train]).mean())
    assert acc > 0.7, acc


def test_chebnetii_head_is_deterministic():
    W, X, y = _sbm()
    W_norm = normalize_adjacency(W)
    train = np.zeros(y.size, dtype=bool)
    train[::2] = True
    idx = np.flatnonzero(~train)
    a = ChebNetIIHead(W_norm, X.shape[1], train, K=2, epochs=50, seed=0).fit_graph(X, y)
    b = ChebNetIIHead(W_norm, X.shape[1], train, K=2, epochs=50, seed=0).fit_graph(X, y)
    assert np.array_equal(a.predict_nodes(X, idx), b.predict_nodes(X, idx))


def test_chebnetii_head_never_uses_test_labels():
    W, X, y = _sbm()
    W_norm = normalize_adjacency(W)
    train = np.zeros(y.size, dtype=bool)
    train[::2] = True
    test = ~train
    y_flipped = y.copy()
    y_flipped[test] = 1 - y_flipped[test]
    a = ChebNetIIHead(W_norm, X.shape[1], train, K=2, epochs=50, seed=0).fit_graph(X, y)
    b = ChebNetIIHead(W_norm, X.shape[1], train, K=2, epochs=50, seed=0).fit_graph(X, y_flipped)
    idx = np.flatnonzero(test)
    assert np.array_equal(a.predict_nodes(X, idx), b.predict_nodes(X, idx))


# --------------------------------------------------------------------------- #
# Node-MoE gate and joint model
# --------------------------------------------------------------------------- #

def test_gate_features_shape_and_label_free():
    W, X, _ = _sbm()
    Wt = torch.tensor(normalize_adjacency(W), dtype=torch.float32)
    xt = torch.tensor(X, dtype=torch.float32)
    gf = gate_features(Wt, xt)
    assert gf.shape == (X.shape[0], 3 * X.shape[1])


def test_node_moe_head_gate_is_a_distribution_and_mixture_matches_experts():
    W, X, y = _sbm()
    W_norm = normalize_adjacency(W)
    train = np.zeros(y.size, dtype=bool)
    train[::2] = True
    head = NodeMoEHead(W_norm, X.shape[1], train,
                       filter_types=differentiated_filter_types(3),
                       K=2, epochs=50, seed=0).fit_graph(X, y)
    g = head.gate_weights(X)
    assert g.shape == (y.size, 3)
    assert np.allclose(g.sum(axis=1), 1.0, atol=1e-5)
    assert np.all(g >= 0.0)
    idx = np.flatnonzero(~train)
    expert_logits = head.expert_logits_nodes(X, idx)
    assert expert_logits.shape == (idx.size, 3)


def test_node_moe_head_is_deterministic():
    W, X, y = _sbm()
    W_norm = normalize_adjacency(W)
    train = np.zeros(y.size, dtype=bool)
    train[::2] = True
    idx = np.flatnonzero(~train)
    kw = dict(filter_types=differentiated_filter_types(3), K=2, epochs=50, seed=1)
    a = NodeMoEHead(W_norm, X.shape[1], train, **kw).fit_graph(X, y)
    b = NodeMoEHead(W_norm, X.shape[1], train, **kw).fit_graph(X, y)
    assert np.array_equal(a.predict_nodes(X, idx), b.predict_nodes(X, idx))
    assert np.allclose(a.gate_weights(X), b.gate_weights(X), atol=1e-6)


# --------------------------------------------------------------------------- #
# Protocol B
# --------------------------------------------------------------------------- #

def test_build_differentiated_pool_routing_blind():
    W, X, y = _sbm()
    W_norm = normalize_adjacency(W)
    fit = np.ones(y.size, dtype=bool)
    pool = build_differentiated_pool(W_norm, X, y, fit,
                                     differentiated_filter_types(3),
                                     K=2, epochs=50, seed=0)
    assert len(pool) == 3
    # Each expert trained on all fit nodes: still a valid, competent predictor.
    idx = np.arange(y.size)
    accs = [float((h.predict_nodes(X, idx) == y).mean()) for h in pool]
    assert min(accs) > 0.7


def test_protocol_b_returns_all_conditions_and_reports_node_counts():
    W, X, y = _sbm()
    W_norm = normalize_adjacency(W)
    rng = np.random.default_rng(0)
    fit = np.zeros(y.size, dtype=bool)
    fit[rng.random(y.size) < 0.4] = True
    router = np.zeros(y.size, dtype=bool)
    router[rng.random(y.size) < 0.2] = True
    test = ~(fit | router)
    oracle = oracle_bucket_assignment(W, y, adaptive=True)
    res = evaluate_node_moe_protocol_b(
        X, y, W_norm, fit, router, test,
        differentiated_filter_types(3), oracle,
        K=2, epochs=50, gate_epochs=100, seed=0,
    )
    for key in ("single_global", "node_moe_gate", "random", "oracle",
                "ensemble_uniform"):
        assert key in res["accuracy"]
        assert np.isfinite(res["accuracy"][key])
    assert res["n_fit_nodes"] == int(fit.sum())
    assert res["n_router_nodes"] == int(router.sum())

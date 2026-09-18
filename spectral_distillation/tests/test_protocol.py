"""Phase 3 tests: router protocol, planted control, mixture, evaluation stats.

Covers guide sections 6.2-6.7 building blocks: label/feature homophily,
oracle/random/uniform/label-free assignments, dilution-ladder invariants,
expert mixture robustness, paired statistics, Holm correction, and the
frozen-gate intervention. Kept fast (small graphs, few epochs) for CPU.
"""

import numpy as np
import pytest

from spectral_distillation.src.evaluation import (
    aggregate_seeds_within_split,
    effect_size_dz,
    frozen_intervention,
    holm_bonferroni,
    mean_ci95,
    paired_ttest,
    run_protocol,
    wilcoxon_paired,
)
from spectral_distillation.src.gnn_router import (
    build_router_gnn,
    degree_clustering_proxy,
    expert_mask_from_assignment,
    gnn_router_forward,
    route_nodes,
)
from spectral_distillation.src.laplacian import compute_laplacian, normalize_adjacency
from spectral_distillation.src.mixture import (
    fit_experts,
    mixture_accuracy,
    train_test_split,
)
from spectral_distillation.src.planted_control import (
    dilution_ladder,
    generate_pc_graph,
    single_m_expert_accuracy,
)
from spectral_distillation.src.router_protocol import (
    evaluate_decision_rules,
    feature_homophily,
    label_free_assignment,
    label_homophily,
    oracle_bucket_assignment,
    oracle_regime_assignment,
    random_assignment,
    structural_features,
    uniform_assignment,
)


# --------------------------------------------------------------------------- #
# homophily / prompts
# --------------------------------------------------------------------------- #

def test_label_homophily_is_proportion_of_same_label_neighbors():
    W = np.zeros((4, 4))
    W[0, 1] = W[1, 0] = 1.0
    W[1, 2] = W[2, 1] = 1.0
    W[2, 3] = W[3, 2] = 1.0
    W[3, 0] = W[0, 3] = 1.0
    y = np.array([0, 0, 1, 1])
    h = label_homophily(W, y)
    # node 0 sees {1,3}: one same-class (0-0), one diff -> 0.5
    assert h[0] == pytest.approx(0.5)
    assert h[2] == pytest.approx(0.5)
    assert 0.0 <= h.min() and h.max() <= 1.0


def test_label_homophily_isolated_node_is_nan():
    W = np.zeros((3, 3))
    W[0, 1] = W[1, 0] = 1.0
    y = np.array([0, 1, 2])
    h = label_homophily(W, y)
    assert np.isnan(h[2])
    assert h[0] == pytest.approx(0.0)
    assert h[1] == pytest.approx(0.0)


def test_feature_homophily_prefers_aligned_vectors():
    W = np.zeros((3, 3))
    W[0, 1] = W[1, 0] = 1.0
    X = np.array([[1.0, 0.0], [1.0, 0.1], [0.0, 1.0]])
    hx = feature_homophily(W, X)
    # nodes 0 and 1 have near-identical feature directions
    assert hx[0] > 0.9
    X2 = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    hx2 = feature_homophily(W, X2)
    assert hx2[0] < -0.9


def test_oracle_bucket_assignment_channels_and_soft_gates():
    W = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    # node 0: sole neighbor 1 with same label -> h=1 -> low-pass channel 0
    # node 3: sole neighbor 2 with same label -> h=1 -> low-pass channel 0
    y = np.array([0, 0, 0, 0])
    assign = oracle_bucket_assignment(W, y, h_low=0.4, h_high=0.6)
    assert assign.shape == (4, 3)
    assert np.allclose(assign[0], [1.0, 0.0, 0.0])
    soft = oracle_bucket_assignment(W, y, h_low=0.4, h_high=0.6, soft_gates=(0.8, 0.1, 0.1))
    assert np.allclose(np.sum(soft, axis=1), 1.0)
    assert np.allclose(soft[0], [0.8, 0.1, 0.1])


# --------------------------------------------------------------------------- #
# assignments
# --------------------------------------------------------------------------- #

def test_uniform_assignment_is_flat():
    a = uniform_assignment(10, 3)
    assert a.shape == (10, 3)
    assert np.allclose(a, 1 / 3)


def test_oracle_regime_assignment_one_hot():
    regimes = np.array([0, 1, 2, 1])
    a = oracle_regime_assignment(regimes, 3)
    assert np.allclose(a.sum(axis=1), 1.0)
    assert np.argmax(a, axis=1).tolist() == [0, 1, 2, 1]


def test_random_assignment_is_usage_matched_and_shuffled():
    oracle = oracle_regime_assignment(np.array([0, 0, 0, 1, 1, 2]), 3)
    rnd = random_assignment(oracle, rng=0)
    assert np.allclose(rnd.sum(axis=1), 1.0)
    assert rnd.shape == oracle.shape
    # per-node draws are shuffled, not the oracle ordering
    assert not np.array_equal(np.argmax(rnd, axis=1), np.argmax(oracle, axis=1))
    # usage is matched in expectation, not exactly, on a tiny sample
    assert rnd.sum() == pytest.approx(oracle.sum())


def test_random_assignment_deterministic_given_seed():
    oracle = oracle_regime_assignment(np.array([0, 0, 1, 1, 2, 2]), 3)
    a = random_assignment(oracle, rng=7)
    b = random_assignment(oracle, rng=7)
    assert np.array_equal(a, b)


def test_structural_features_shape_and_bounds(small_pc_graph):
    W = np.asarray(small_pc_graph["W"], dtype=float)
    features = np.asarray(small_pc_graph["features"], dtype=float)
    L = compute_laplacian(W)
    evals, V = np.linalg.eigh(L)
    feats = structural_features(W, features, evals, V)
    assert feats.shape == (W.shape[0], 4)
    assert np.all(np.isfinite(feats))


def test_label_free_assignment_returns_valid_soft_rows(small_pc_graph):
    W = np.asarray(small_pc_graph["W"], dtype=float)
    features = np.asarray(small_pc_graph["features"], dtype=float)
    assignment, router = label_free_assignment(
        W, features, hidden_dim=16, n_layers=2, n_experts=3, epochs=3, seed=0
    )
    assert assignment.shape == (W.shape[0], 3)
    assert np.allclose(assignment.sum(axis=1), 1.0)
    assert router is not None
    # label-free assignment should not collapse to a single expert
    usage = np.argmax(assignment, axis=1)
    assert np.unique(usage).size > 1


# --------------------------------------------------------------------------- #
# GNN router
# --------------------------------------------------------------------------- #

def test_route_nodes_deterministic_softmax():
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((5, 3))
    pi = np.exp(logits)
    pi = pi / pi.sum(axis=1, keepdims=True)
    soft = route_nodes(pi, np.eye(5))
    assert soft.shape == (5, 3)
    assert np.allclose(soft.sum(axis=1), 1.0, atol=1e-6)
    assert soft.min() >= 0.0
    # normalization is idempotent for an already-stochastic routing matrix
    assert np.allclose(soft, route_nodes(pi, np.eye(5)))


def test_gnn_router_forward_shape_and_probs():
    pytest.importorskip("torch")
    model = build_router_gnn(n_features=4, hidden_dim=16, n_layers=2, n_experts=3)
    W = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    W_norm = normalize_adjacency(W)
    x = np.random.default_rng(0).standard_normal((3, 4)).astype(np.float32)
    out = gnn_router_forward(model, x, W_norm, device="cpu")
    assert out.shape == (3, 3)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-6)
    assert out.min() >= 0.0


def test_expert_mask_from_assignment():
    a = np.zeros((6, 3))
    a[:, 0] = 1.0
    mask = expert_mask_from_assignment(a, 0)
    assert mask.dtype == np.bool_
    assert mask.sum() == 6
    mask1 = expert_mask_from_assignment(a, 1)
    assert mask1.sum() == 0


def test_degree_clustering_proxy_range():
    d = np.array([0.0, 2.0, 20.0])
    c = degree_clustering_proxy(d)
    assert np.all(c >= 0.0) and np.all(c <= 1.0)
    assert c[0] == pytest.approx(0.0)
    assert c[2] > c[1]


# --------------------------------------------------------------------------- #
# planted control / dilution
# --------------------------------------------------------------------------- #

def test_pc_graph_invariants(small_pc_graph):
    graph = small_pc_graph
    W = np.asarray(graph["W"], dtype=float)
    features = np.asarray(graph["features"], dtype=float)
    regimes = np.asarray(graph["regimes"])
    y = np.asarray(graph["y"])
    n = W.shape[0]
    assert W.shape == (n, n)
    assert np.allclose(W, W.T)
    assert np.trace(W) == pytest.approx(0.0)
    assert features.shape[0] == n
    assert regimes.shape == (n,)
    assert set(np.unique(regimes)).issubset({0, 1, 2})
    assert set(np.unique(y)) == {0, 1, 2, 3, 4}


def test_m_expert_does_not_saturate(small_pc_graph):
    graph = small_pc_graph
    acc = single_m_expert_accuracy(graph, n_train=0.5, seed=1)
    assert 0.3 < acc < 0.98  # above chance on 5 classes, but not saturated


def test_dilution_ladder_preserves_invariants():
    graph = generate_pc_graph(n_nodes=200, n_patches=4, d_features=8, seed=3)
    W = np.asarray(graph["W"], dtype=float)
    base_nnz = (W > 0).sum()
    rungs = dilution_ladder(graph, [0.0, 0.5, 1.0], seed=3)
    assert len(rungs) == 3
    for rung, d in zip(rungs, [0.0, 0.5, 1.0]):
        assert rung["dilution"] == pytest.approx(d)
        Wd = np.asarray(rung["W"], dtype=float)
        assert Wd.shape == W.shape
        assert np.allclose(Wd, Wd.T)
        assert np.trace(Wd) == pytest.approx(0.0)
        if d == 0.0:
            assert np.allclose(Wd, W)
        # converted edges are replaced, so edge count may not drop below backbone
        assert (Wd > 0).sum() >= base_nnz * 0.5
    # converted sets are nested
    m05 = rungs[1]["converted_mask"]
    m10 = rungs[2]["converted_mask"]
    assert m10.sum() >= m05.sum()


# --------------------------------------------------------------------------- #
# mixture
# --------------------------------------------------------------------------- #

def test_fit_experts_handles_small_routed_subsets():
    rng = np.random.default_rng(0)
    n, k = 60, 4
    X = rng.standard_normal((n, k))
    y = np.tile(np.arange(3), 20)
    assignment = np.zeros((n, 3))
    assignment[:, 1] = 1.0  # everyone to expert 1 -> experts 0,2 empty
    train, test = train_test_split(y, 0, 0)
    experts = fit_experts(X, y, assignment, train)
    assert experts[0] is None and experts[2] is None
    assert experts[1] is not None
    acc, _ = mixture_accuracy(X, y, assignment, experts, test)
    assert 0.0 <= acc <= 1.0


def test_mixture_correct_routing_beats_wrong_routing():
    # XOR-like data: no global linear model fits, but a separate expert per
    # x-sign quadrant does -> correct routing must beat uniform mixing.
    rng = np.random.default_rng(1)
    n = 200
    X = rng.uniform(-2.0, 2.0, size=(n, 2))
    y = ((X[:, 0] > 0) == (X[:, 1] > 0)).astype(int)
    left = X[:, 0] > 0
    assignment = np.zeros((n, 2))
    assignment[left, 0] = 1.0
    assignment[~left, 1] = 1.0
    # uniform mixing routes every node to one expert (argmax 0) -> global model
    wrong = np.full((n, 2), 0.5)

    train, test = train_test_split(y, 0, 0)
    experts = fit_experts(X, y, assignment, train)
    acc_right, _ = mixture_accuracy(X, y, assignment, experts, test)
    experts_w = fit_experts(X, y, wrong, train)
    acc_wrong, _ = mixture_accuracy(X, y, wrong, experts_w, test)
    assert acc_right > acc_wrong + 0.1


# --------------------------------------------------------------------------- #
# statistics / protocol / frozen intervention
# --------------------------------------------------------------------------- #

def test_wilcoxon_handles_constant_differences():
    a = np.array([0.6] * 8)
    b = np.array([0.5] * 8)
    assert wilcoxon_paired(a, b) < 0.05
    assert wilcoxon_paired(a, a) == pytest.approx(1.0)


def test_paired_ttest_and_effect_size():
    a = np.array([0.7, 0.6, 0.8, 0.7])
    b = np.array([0.6, 0.5, 0.6, 0.6])
    assert paired_ttest(a, b) < 0.05
    assert effect_size_dz(a, b) > 0.0
    assert effect_size_dz(a, a) == pytest.approx(0.0)


def test_mean_ci95_contains_mean():
    d = np.array([0.2, 0.1, 0.25, 0.15])
    m, lo, hi = mean_ci95(d, np.zeros_like(d))
    assert lo <= m <= hi


def test_holm_bonferroni_orders_thresholds():
    ps = [0.01, 0.04, 0.3]
    thr = holm_bonferroni(ps, alpha=0.05)
    # the smallest p gets the largest threshold
    assert thr[np.argmin(ps)] == pytest.approx(0.05 / 3)
    assert thr[np.argmax(ps)] == pytest.approx(0.05 / 1)


def test_aggregate_seeds_within_split():
    cells = {(s, seed): 0.5 + 0.1 * seed + 0.01 * s for s in range(3) for seed in range(2)}
    means = aggregate_seeds_within_split(cells, 3, 2)
    assert len(means) == 3
    assert abs(means[0] - np.mean([0.5, 0.6])) < 1e-9


def test_run_protocol_orders_pairs_and_applies_holm():
    def run_fn(condition, split, seed):
        base = {"oracle": 0.7, "uniform": 0.55, "random": 0.52, "label_free": 0.68}
        return base[condition] + 0.002 * split + 0.001 * seed

    res = run_protocol(run_fn, n_splits=10, n_seeds=2)
    by_name = {c.name: c for c in res.comparisons}
    assert by_name["oracle_vs_uniform"].significant
    assert by_name["oracle_vs_random"].significant
    assert by_name["label_free_vs_random"].significant
    for c in res.comparisons:
        assert c.alpha_corrected >= 0.05 / 3 - 1e-12


def test_evaluate_decision_rules_all_supported():
    comparisons = {
        "oracle_vs_uniform": {"mean_diff": 0.2, "p_wilcoxon": 0.01},
        "label_free_vs_random": {"mean_diff": 0.15, "p_wilcoxon": 0.01},
        "oracle_vs_random": {"mean_diff": 0.3, "p_wilcoxon": 0.01},
    }
    control = {"oracle_vs_uniform": {"mean_diff": 0.2, "p_wilcoxon": 0.01}}
    rules = evaluate_decision_rules(comparisons, control_comparisons=control, alpha=0.05)
    assert rules.supported
    assert "Supported" in rules.verdict


def test_evaluate_decision_rules_fails_without_signal():
    comparisons = {
        "oracle_vs_uniform": {"mean_diff": -0.1, "p_wilcoxon": 0.5},
        "label_free_vs_random": {"mean_diff": 0.0, "p_wilcoxon": 0.9},
        "oracle_vs_random": {"mean_diff": -0.05, "p_wilcoxon": 0.6},
    }
    rules = evaluate_decision_rules(comparisons)
    assert not rules.supported
    assert any(r is not True for r in (rules.D1, rules.D2, rules.D3))


def test_frozen_intervention_learned_within_permutation_range():
    rng = np.random.default_rng(0)
    n, K = 80, 3
    target = np.arange(n) % K
    # concentrated learned soft assignment toward each node's target expert
    learned = 0.1 / K + rng.random((n, K)) * 0.05
    learned[np.arange(n), target] = 0.85
    learned = learned / learned.sum(axis=1, keepdims=True)

    def run_expert_fn(assignment):
        assignment = np.asarray(assignment, dtype=float)
        return float(np.mean(assignment[np.arange(n), target]))

    oracle = np.zeros((n, K))
    oracle[np.arange(n), target] = 1.0
    out = frozen_intervention(
        learned,
        n,
        run_expert_fn,
        n_experts=K,
        oracle=oracle,
        gate_settings=["learned", "node_shuffled", "global_mean", "equal", "oracle_onehot"],
        n_permutations=10,
        seed=0,
    )
    assert out["oracle_onehot"] == pytest.approx(1.0)
    assert out["learned"] > out["node_shuffled_mean"] + out["node_shuffled_std"]
    assert out["learned"] > out["equal"]
    assert out["learned"] > out["global_mean"]


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="session")
def small_pc_graph():
    return generate_pc_graph(n_nodes=150, n_patches=3, d_features=8, seed=42)
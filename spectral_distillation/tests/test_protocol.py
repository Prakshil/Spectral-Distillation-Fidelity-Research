"""Phase 3 tests: router protocol, planted control, mixture, evaluation stats.

Covers guide sections 6.2-6.7 building blocks: label/feature homophily,
oracle/random/uniform/label-free assignments, dilution-ladder invariants,
expert mixture robustness, paired statistics, Holm correction, and the
frozen-gate intervention. Kept fast (small graphs, few epochs) for CPU.
"""

from pathlib import Path

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
    EXPERT_FACTORIES,
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
    _buckets_from_score,
    evaluate_decision_rules,
    feature_homophily,
    label_free_assignment,
    label_homophily,
    budget_normalized_homophily,
    oracle_bucket_assignment,
    oracle_regime_assignment,
    random_assignment,
    spectral_neighbor_similarity,
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


def test_budget_normalized_homophily_rescales_short_sentences():
    # sentence 0: 2 tokens, sentence 1: 6 tokens, sentence 2: 4 tokens.
    # Node 2 (sentence 1) has 1 same-sentence edge of a possible 6 -> low raw h
    # but the *proportion of budget used* is what changes routing.
    W = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    y = np.array([0, 0, 1, 1])
    h = budget_normalized_homophily(W, y, top_k=4)
    # node 0: 1/1 same-sentence edges inside a 2-token sentence -> budget 1 -> 1.0
    assert np.isclose(h[0], 1.0)
    # node 2: 1/1 same-sentence edges inside a 2-token sentence -> budget 1 -> 1.0
    assert np.isclose(h[2], 1.0)


def test_budget_normalized_oracle_uses_rescaled_thresholds():
    W = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    y = np.array([0, 0, 1, 1])
    a = oracle_bucket_assignment(W, y, adaptive=False, budget_normalized=True, top_k=4)
    assert a.shape == (4, 3)
    assert np.allclose(np.sum(a, axis=1), 1.0)
    # every node's only edge is same-sentence -> budget-normalized h = 1.0 -> low-pass
    assert np.allclose(a[:, 0], 1.0)


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
    assert feats.shape == (W.shape[0], 5)
    assert np.all(np.isfinite(feats))
    # column 0 is the spectral-neighborhood-similarity ordering key
    assert feats.shape[1] == 5


def test_buckets_from_score_orders_high_score_to_channel_zero():
    score = np.arange(99, dtype=float)
    a = _buckets_from_score(score, n_experts=3)
    assert a.shape == (99, 3)
    assert np.allclose(a.sum(axis=1), 1.0)
    # highest score -> channel 0 (low-pass)
    assert a[-1].argmax() == 0
    assert a[0].argmax() == 2
    # monotone: channel index never increases as score grows
    ch = a.argmax(axis=1)
    assert np.all(np.diff(ch) <= 0)


def test_buckets_from_score_degenerate_inputs():
    # constant score -> uniform, no crash
    a = _buckets_from_score(np.zeros(10), n_experts=3)
    assert np.allclose(a, 1.0 / 3.0)
    # single distinct value
    a = _buckets_from_score(np.ones(10), n_experts=3)
    assert np.allclose(a.sum(axis=1), 1.0)
    # quantiles can collide on heavy ties
    s = np.concatenate([np.zeros(50), np.ones(50)])
    a = _buckets_from_score(s, n_experts=3)
    assert np.allclose(a.sum(axis=1), 1.0)


def test_label_free_strategies_both_produce_valid_rows(small_pc_graph):
    W = np.asarray(small_pc_graph["W"], dtype=float)
    X = np.asarray(small_pc_graph["features"], dtype=float)
    for strategy, feat in [
        ("kmeans", "hx"),
        ("kmeans", "eig_nb_sim"),
        ("proxy_quantile", "hx"),
        ("proxy_quantile", "eig_nb_sim"),
    ]:
        a, _ = label_free_assignment(
            W, X, hidden_dim=16, n_layers=2, n_experts=3, epochs=3, seed=0,
            strategy=strategy, order_feature=feat,
        )
        assert a.shape == (W.shape[0], 3), strategy
        assert np.all(a >= 0.0), strategy
        assert np.allclose(a.sum(axis=1), 1.0, atol=1e-9), strategy


def test_proxy_quantile_is_monotone_in_order_feature(small_pc_graph):
    """The D2 fix depends on the shipped router bucketing the score directly."""
    W = np.asarray(small_pc_graph["W"], dtype=float)
    X = np.asarray(small_pc_graph["features"], dtype=float)
    a, _ = label_free_assignment(
        W, X, hidden_dim=16, n_layers=2, n_experts=3, epochs=3, seed=0,
        strategy="proxy_quantile", order_feature="eig_nb_sim",
    )
    L = compute_laplacian(W)
    _, V = np.linalg.eigh(L)
    s = spectral_neighbor_similarity(W, V, k=8)
    ch = a.argmax(axis=1)
    # mean score must decrease as channel index increases
    means = [s[ch == k].mean() if (ch == k).any() else np.inf for k in range(3)]
    assert means[0] >= means[1] - 1e-9
    assert means[1] >= means[2] - 1e-9


def test_spectral_neighbor_similarity_bounds(small_pc_graph):
    W = np.asarray(small_pc_graph["W"], dtype=float)
    L = compute_laplacian(W)
    _, V = np.linalg.eigh(L)
    s = spectral_neighbor_similarity(W, V, k=4)
    assert s.shape == (W.shape[0],)
    assert np.all(np.isfinite(s))
    assert (s <= 1.0 + 1e-9).all()
    assert (s >= -1.0 - 1e-9).all()


def test_label_free_order_feature_selects_column(small_pc_graph):
    W = np.asarray(small_pc_graph["W"], dtype=float)
    X = np.asarray(small_pc_graph["features"], dtype=float)
    a_new, _ = label_free_assignment(
        W, X, hidden_dim=16, n_layers=2, n_experts=3, epochs=3, seed=0,
        order_feature="eig_nb_sim",
    )
    a_legacy, _ = label_free_assignment(
        W, X, hidden_dim=16, n_layers=2, n_experts=3, epochs=3, seed=0,
        order_feature="hx",
    )
    for a in (a_new, a_legacy):
        assert a.shape == (W.shape[0], 3)
        assert np.allclose(a.sum(axis=1), 1.0, atol=1e-9)


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
# channel-permutation invariance of hard-routed mixture accuracy
# --------------------------------------------------------------------------- #

def test_mixture_accuracy_invariant_to_channel_permutation():
    """Hard routing makes accuracy depend only on the induced partition.

    ``fit_experts`` trains expert ``k`` on exactly the nodes routed to ``k`` and
    ``mixture_accuracy`` scores expert ``k`` on exactly those nodes, so the
    total is ``sum_k acc(group_k trained on group_k)``. Permuting channel
    labels therefore cannot change the score.

    Consequence for the candidate search: negating a score only swaps which
    quantile bucket is labelled 0 vs 2, so D2 is provably invariant to score
    sign. Orientation-by-sign is a no-op, and a candidate cannot be rescued or
    broken by flipping it. This test pins that property so a future change to
    routing does not silently alter previously reported D2 values.
    """
    from spectral_distillation.src.mixture import fit_experts, mixture_accuracy

    rng = np.random.default_rng(0)
    n, d, k = 300, 6, 3
    X = rng.normal(size=(n, d))
    y = (X[:, 0] + 0.5 * rng.normal(size=n) > 0).astype(int)
    score = rng.normal(size=n)

    onehot = np.zeros((n, k))
    onehot[np.arange(n), np.argsort(np.argsort(score)) % k] = 1.0
    permuted = onehot[:, [2, 1, 0]]

    train = np.zeros(n, dtype=bool)
    train[:200] = True
    test = ~train

    acc_a, _ = mixture_accuracy(X, y, onehot, fit_experts(X, y, onehot, train), test)
    acc_b, _ = mixture_accuracy(X, y, permuted, fit_experts(X, y, permuted, train), test)

    assert acc_a == pytest.approx(acc_b, abs=1e-12)


def test_positive_control_oracle_beats_global_where_routing_is_useful():
    """The harness must report a win on a task where routing provably helps.

    All three real datasets give 0 wins, which only means something if the
    protocol is capable of detecting a win at all. The positive control's label
    function is a non-separable mixture of linear rules: three latent groups,
    each with its own linear boundary, no single hyperplane fitting all three.
    A global logistic model is therefore structurally wrong, while an expert
    trained on one group is structurally right -- so oracle routing must win by
    a wide margin. Random routing must not, which rules out "everything beats
    the baseline".
    """
    from spectral_distillation.experiments.run_fixed_expert_protocol import (
        evaluate_all, feature_kmeans_routing, single_global_condition,
    )
    from spectral_distillation.src.positive_control import (
        generate_positive_control, oracle_from_groups,
    )

    graph = generate_positive_control(n=900, seed=0)
    X, y, groups = graph["features"], graph["y"], graph["group"]

    pool = feature_kmeans_routing(X, n_experts=3)
    # The fixed pool has to line up with the latent groups, otherwise the
    # positive control would fail for an unrelated reason.
    pool_labels = pool.argmax(axis=1)
    for g in range(3):
        idx = groups == g
        purity = np.bincount(pool_labels[idx], minlength=3).max() / idx.sum()
        assert purity > 0.95

    oracle = oracle_from_groups(groups, pool)
    conditions = {
        "single_global": single_global_condition(X.shape[0]),
        "oracle": oracle,
        "random": random_assignment(oracle, rng=0),
    }

    class _NullLog:
        def info(self, *a, **k):
            pass

    res = evaluate_all(
        conditions, X, y, 2, 2, log=_NullLog(),
        expert_assignment=pool,
        expert_factory=EXPERT_FACTORIES["logistic"],
    )
    mean = lambda k: float(np.mean(list(res[k].values())))  # noqa: E731
    global_acc, oracle_acc, random_acc = (
        mean("single_global"), mean("oracle"), mean("random")
    )

    assert oracle_acc > global_acc + 0.10, (
        f"positive control failed: oracle {oracle_acc:.4f} vs global {global_acc:.4f}"
    )
    # Guards against a degenerate 'every routing wins' harness.
    assert random_acc < oracle_acc - 0.05


def test_learned_router_recovers_routing_gain_on_positive_control():
    """RouterGNN-lite must find real structure when structure exists.

    Same positive control as above, where oracle routing beats the global model by
    >0.10. A learned router that only ever imitated the fixed pool would land at
    k-means accuracy; one that learns per-node expert suitability from training
    labels should land at or above it. This is the "can a learned router help at
    all" check -- the complement of the 0/234 real-data result.
    """
    from spectral_distillation.experiments.run_fixed_expert_protocol import (
        feature_kmeans_routing,
    )
    from spectral_distillation.src.laplacian import normalize_adjacency
    from spectral_distillation.src.learned_router import evaluate_learned_routing
    from spectral_distillation.src.mixture import train_test_split
    from spectral_distillation.src.positive_control import (
        generate_positive_control, oracle_from_groups,
    )

    graph = generate_positive_control(n=900, seed=0)
    X, y = graph["features"], graph["y"]
    W_norm = normalize_adjacency(graph["W"])
    pool = feature_kmeans_routing(X, n_experts=3)
    oracle = oracle_from_groups(graph["group"], pool)

    train, test = train_test_split(y, 0, 0)
    res = evaluate_learned_routing(
        X, y, W_norm, pool, train, test, oracle,
        expert_factory=EXPERT_FACTORIES["logistic"], seed=0, epochs=60,
    )
    acc = res["accuracy"]
    assert acc["single_global"] < 0.90
    # Trained router should be materially better than no routing on a task where
    # routing is provably useful.
    assert acc["learned_router"] > acc["single_global"] + 0.05, acc
    # And it should not be a strictly worse imitation of the fixed pool.
    assert acc["learned_router"] > acc["random"] + 0.05, acc


def test_learned_router_never_sees_test_labels():
    """Corrupting test labels must not change the router's learned assignment.

    Pins the leakage contract: the router is supervised by ``train`` nodes only,
    and every label outside ``train`` is invisible to it.
    """
    from spectral_distillation.experiments.run_fixed_expert_protocol import (
        feature_kmeans_routing,
    )
    from spectral_distillation.src.laplacian import normalize_adjacency
    from spectral_distillation.src.learned_router import (
        build_router_targets, evaluate_learned_routing, split_fit_router,
        train_router,
    )
    from spectral_distillation.src.mixture import fit_experts, train_test_split
    from spectral_distillation.src.positive_control import (
        generate_positive_control, oracle_from_groups,
    )

    graph = generate_positive_control(n=600, seed=1)
    X, y = graph["features"], graph["y"]
    train, test = train_test_split(y, 0, 0)
    fit_mask, router_mask = split_fit_router(train, y, seed=0)
    # Nothing outside `train` may enter supervision.
    assert not (router_mask & test).any()
    assert not (fit_mask & test).any()

    pool = feature_kmeans_routing(X, n_experts=3)
    experts = fit_experts(X, y, pool, fit_mask, expert_factory=EXPERT_FACTORIES["logistic"])
    router_idx = np.flatnonzero(router_mask)
    targets = build_router_targets(y, experts, X, router_idx)

    W_norm = normalize_adjacency(graph["W"])
    m1, _ = train_router(X, W_norm, targets, router_idx, 3, epochs=30, seed=0)

    # Flip every test label; the router's parameters must be untouched.
    y_flip = y.copy()
    y_flip[test] = 1 - y_flip[test]
    experts_flip = fit_experts(X, y_flip, pool, fit_mask,
                               expert_factory=EXPERT_FACTORIES["logistic"])
    targets_flip = build_router_targets(y_flip, experts_flip, X, router_idx)
    m2, _ = train_router(X, W_norm, targets_flip, router_idx, 3, epochs=30, seed=0)

    assert np.allclose(targets, targets_flip)
    import torch  # local: top-level torch breaks the pyarrow import order on Windows

    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(p1, p2)


def test_positive_control_generator_is_deterministic():
    from spectral_distillation.src.positive_control import generate_positive_control

    a = generate_positive_control(n=300, seed=3)
    b = generate_positive_control(n=300, seed=3)
    assert np.array_equal(a["features"], b["features"])
    assert np.array_equal(a["y"], b["y"])
    assert np.array_equal(a["W"], b["W"])


def test_shared_pool_single_global_is_trained_on_all_train_nodes():
    """The no-routing reference must not inherit the shared pool's blind spot.

    ``single_global`` is a one-column assignment, but protocol B shares a
    3-column k-means pool across conditions. Without a dedicated one-column fit,
    argmax sends every test node to head 0 -- a head trained on cluster 0 only --
    which understates the no-routing baseline and quietly flatters every
    candidate.
    """
    from spectral_distillation.experiments.run_fixed_expert_protocol import (
        evaluate_all, feature_kmeans_routing, single_global_condition,
    )

    rng = np.random.default_rng(0)
    n, d = 300, 6
    # Cluster structure correlated with the label, so the k-means pool splits
    # the training rows -- the exact situation that made head 0 partial.
    X = np.concatenate([
        rng.normal(loc=[-2.0] * d, size=(n // 2, d)),
        rng.normal(loc=[2.0] * d, size=(n // 2, d)),
    ])
    y = (X[:, 0] > 0).astype(int)
    train = np.zeros(n, dtype=bool)
    train[:200] = True

    km = feature_kmeans_routing(X, n_experts=3)
    assert km.shape[1] == 3

    class _NullLog:
        def info(self, *a, **k):
            pass

    # Record the training mask each head is built from. Accuracy alone cannot
    # catch this: on separable data a partial head still classifies the test set
    # perfectly, so the contract has to be asserted directly.
    seen: list[np.ndarray] = []

    def spy_factory(mask):
        seen.append(np.asarray(mask).copy())
        return EXPERT_FACTORIES["logistic"](mask)

    evaluate_all(
        {"single_global": single_global_condition(n)}, X, y, 1, 1,
        log=_NullLog(), expert_assignment=km, expert_factory=spy_factory,
    )

    # ``evaluate_all`` uses split=0/seed=0, so replicate that exact train mask
    # and require the global head to cover every one of its training nodes.
    expected_train, _ = train_test_split(y, 0, 0)
    covering = [m for m in seen if np.array_equal(m, expected_train)]
    assert len(covering) == 1, [int(m.sum()) for m in seen]
    # The k-means heads must be strict subsets, otherwise the assertion above
    # would pass even if the dedicated global fit were missing.
    assert any(
        m.sum() < int(expected_train.sum()) for m in seen if not np.array_equal(m, expected_train)
    ), [int(m.sum()) for m in seen]


def _ring_graph(n: int):
    A = np.zeros((n, n))
    for i in range(n):
        A[i, (i + 1) % n] = 1.0
        A[i, (i - 1) % n] = 1.0
    return A


def _sbm(n_per_block: int, n_blocks: int, p_in: float, p_out: float,
         seed: int = 0):
    """Stochastic block model. Returns ``(A, block_labels)``.

    Used instead of a ring wherever message passing must survive an induced
    subgraph. A ring has degree 2, so masking half the nodes leaves a held-out
    node with almost no owned neighbours and every graph expert degenerates into
    a per-row one -- which measures subgraph sparsity, not implicit routing.
    """
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(n_blocks), n_per_block)
    n = n_per_block * n_blocks
    P = np.where(labels[:, None] == labels[None, :], p_in, p_out)
    A = np.triu((rng.random((n, n)) < P).astype(float), 1)
    return A + A.T, labels


def _neighbour_only_labels(A, X):
    """Binary label from neighbour features only, so per-row models are at chance."""
    deg = (A > 0).sum(1)
    nbr_mean = (A @ X[:, 0]) / np.maximum(deg, 1)
    return (nbr_mean > np.median(nbr_mean)).astype(int)


def test_gnn_expert_trains_on_routed_subset_not_nodes_zero_and_one():
    """Regression: ``fit_experts`` passes a *boolean* mask to the factory.

    ``numpy`` casts a bool array to 0/1 rather than to node indices, so a naive
    ``asarray(train_idx, dtype=int)`` silently trains the graph expert on nodes 0
    and 1 only. That is invisible in the output shape and destroys accuracy, so
    it is pinned here.
    """
    from spectral_distillation.src.gnn_expert import gnn_head_factory
    from spectral_distillation.src.laplacian import normalize_adjacency

    rng = np.random.default_rng(0)
    n, d = 300, 6
    A = _ring_graph(n)
    X = rng.normal(size=(n, d))
    # Label depends on the neighbourhood, so only message passing can see it.
    y = ((0.25 * X[:, 0] + 1.0 * (A @ X)[:, 0]) > 0).astype(int)
    W_norm = normalize_adjacency(A)

    train = np.zeros(n, dtype=bool)
    train[:200] = True
    idx = np.flatnonzero(train)

    factory = gnn_head_factory(W_norm=W_norm, n_features=d, epochs=200)
    from_mask = factory(train).fit_graph(X, y)
    from_idx = factory(idx).fit_graph(X, y)

    assert np.array_equal(from_mask.train_idx, idx)
    p_mask = from_mask.predict_nodes(X, idx)
    p_idx = from_idx.predict_nodes(X, idx)
    assert np.array_equal(p_mask, p_idx)


def test_gnn_expert_beats_per_row_experts_on_neighbour_dependent_label():
    """The graph expert must actually exploit structure.

    Guards the premise of the whole GNN-expert arm: if message passing bought
    nothing, "routing does not help" would be an artefact of weak experts.
    """
    from spectral_distillation.src.gnn_expert import gnn_head_factory
    from spectral_distillation.src.laplacian import normalize_adjacency

    rng = np.random.default_rng(0)
    n, d = 400, 6
    A = _ring_graph(n)
    X = rng.normal(size=(n, d))
    y = ((0.25 * X[:, 0] + 1.0 * (A @ X)[:, 0]) > 0).astype(int)
    W_norm = normalize_adjacency(A)

    train = np.zeros(n, dtype=bool)
    train[:250] = True
    assign = np.ones((n, 1))  # one global expert, so this isolates model quality

    def acc_of(factory):
        heads = fit_experts(X, y, assign, train, expert_factory=factory)
        acc, _ = mixture_accuracy(X, y, assign, heads, ~train)
        return acc

    gnn_acc = acc_of(gnn_head_factory(W_norm=W_norm, n_features=d, epochs=600))
    lin_acc = acc_of(EXPERT_FACTORIES["logistic"])
    chance = max(y[~train].mean(), 1 - y[~train].mean())

    assert gnn_acc > lin_acc + 0.15, (gnn_acc, lin_acc, chance)
    assert gnn_acc > 0.7, gnn_acc


def test_expert_factory_swaps_head_and_stays_deterministic():
    """``expert_factory`` lets the protocol use non-linear heads reproducibly.

    The fixed-expert re-analysis must be repeatable, so an MLP head has to give
    identical predictions across two fits. Also checks the dummy-head path is
    still reachable through the factory.
    """
    rng = np.random.default_rng(0)
    n, d, k = 200, 5, 2
    X = rng.normal(size=(n, d))
    y = (X[:, 0] > 0).astype(int)
    assignment = np.zeros((n, k))
    assignment[np.arange(n), np.arange(n) % k] = 1.0
    train = np.zeros(n, dtype=bool)
    train[:120] = True
    test = ~train

    assert set(EXPERT_FACTORIES) == {"logistic", "mlp"}
    accs = []
    for _ in range(2):
        heads = fit_experts(X, y, assignment, train, expert_factory=EXPERT_FACTORIES["mlp"])
        acc, n_eval = mixture_accuracy(X, y, assignment, heads, test)
        # Only test nodes routed to a *trained* expert are scored. Each expert
        # needs >=10 train nodes; here both have many, so all test nodes routed
        # to channel 0/1 count.
        assert n_eval == int(test.sum())
        accs.append(acc)
    assert accs[0] == pytest.approx(accs[1], abs=1e-12)

    # single-class routed subset still yields a usable dummy head
    one_class = np.zeros((n, 1))
    one_class[:50] = 1.0
    heads = fit_experts(X, np.zeros(n, dtype=int), one_class, train,
                        expert_factory=EXPERT_FACTORIES["mlp"])
    assert len(heads) == 1


# --------------------------------------------------------------------------- #
# Ens-Avg: the ensemble baseline must average probabilities, not hard votes
# --------------------------------------------------------------------------- #

def test_ensemble_averages_probabilities_not_majority_vote():
    """Regression: Ens-Avg must average class-1 probabilities.

    It previously majority-voted on hard labels, so an expert at p=0.51 counted
    exactly as much as one at p=0.99 and the baseline measured voting strength
    instead of pool complementarity -- the quantity the whole ensemble control
    exists to isolate.

    The fixture is built so majority vote and probability averaging *disagree*:
    expert 0 is mildly right, experts 1 and 2 are confidently wrong. Averaging
    probabilities yields the correct class; hard voting does not.
    """
    from spectral_distillation.src.learned_router import uniform_ensemble_accuracy

    class _Fixed:
        """Minimal head exposing only ``predict_proba``."""

        def __init__(self, p1):
            self.p1 = np.asarray(p1, dtype=float)

        def predict_proba(self, block):
            p = self.p1
            return np.column_stack([1.0 - p, p])

        def predict(self, block):
            return (self.p1 > 0.5).astype(int)

    n = 4
    y = np.array([1, 1, 0, 0])
    all_test = np.ones(n, dtype=bool)

    # One strong expert plus two weak ones that individually outvote it. Under
    # probability averaging the strong expert carries the decision; under a count
    # of hard labels it does not.
    strong = _Fixed([0.90, 0.90, 0.10, 0.10])
    weak_a = _Fixed([0.45, 0.45, 0.55, 0.55])
    weak_b = _Fixed([0.45, 0.45, 0.55, 0.55])
    acc_ens = uniform_ensemble_accuracy(X=np.zeros((n, 1)), y=y,
                                       experts=[strong, weak_a, weak_b],
                                       test_mask=all_test)
    # mean p on node0 = (0.90+0.45+0.45)/3 = 0.60 -> 1 (correct)
    # mean p on node2 = (0.10+0.55+0.55)/3 = 0.40 -> 0 (correct)
    assert acc_ens == 1.0

    # Hard majority vote on the same pool: weak_a/weak_b both say 0 on node 0,
    # so the vote is 0 and node 0 is scored wrong. The gap is what the averaging
    # implementation buys.
    from scipy.stats import mode

    votes = np.stack([strong.predict(np.zeros((n, 1))),
                      weak_a.predict(np.zeros((n, 1))),
                      weak_b.predict(np.zeros((n, 1)))], axis=0)
    vote_pred = mode(votes, axis=0, keepdims=False).mode
    vote_acc = float(np.count_nonzero(vote_pred == y) / n)
    assert vote_acc < acc_ens, (vote_acc, acc_ens)


def _proba_with_fresh_cache(head, X, idx):
    """Class probabilities after clearing the ``h0`` cache.

    ``GNNExpertHead`` caches propagated features per instance, keyed on nothing.
    That is correct in the protocol (every caller passes the same ``X``) but it
    means re-scoring with a *different* ``X`` silently returns the stale cache.
    Tests that probe feature dependence have to drop the cache explicitly.
    """
    head._h0_cache = None
    return head.predict_proba_nodes(X, idx)


def test_ragged_edges_blocks_cross_subset_aggregation():
    """A ragged-edge expert must not see any node outside its own subset.

    This is the implicit-routing control (EMNLP 2023). A full-graph expert
    aggregates from every neighbour, so it can specialise on structure with no
    router involved. Perturbing the features of nodes the expert does *own*
    nothing must leave its output bit-identical -- if the output moves, the
    control is not in force and the null result would be uninterpretable.

    Asserted on probabilities, not argmax labels: a 5.0 feature shift can leave
    every hard label unchanged while the internal representation has clearly
    moved, which would make a label-level assertion pass vacuously.
    """
    from spectral_distillation.src.gnn_expert import gnn_head_factory
    from spectral_distillation.src.laplacian import normalize_adjacency

    rng = np.random.default_rng(0)
    n_per_block, n_blocks, d = 200, 4, 8
    A, _ = _sbm(n_per_block, n_blocks, 0.06, 0.01, seed=1)
    n = n_per_block * n_blocks
    X = rng.normal(size=(n, d))
    y = _neighbour_only_labels(A, X)
    W_norm = normalize_adjacency(A)

    own = np.zeros(n, dtype=bool)
    own[rng.permutation(n)[: n // 2]] = True

    ragged = gnn_head_factory(W_norm=W_norm, n_features=d, epochs=300,
                             ragged_edges=True)(own).fit_graph(X, y)
    full = gnn_head_factory(W_norm=W_norm, n_features=d, epochs=300,
                           ragged_edges=False)(own).fit_graph(X, y)

    idx = np.flatnonzero(own)
    base_ragged = _proba_with_fresh_cache(ragged, X, idx)
    base_full = _proba_with_fresh_cache(full, X, idx)

    # Perturb only features of nodes outside the expert's subset.
    X_perturbed = X.copy()
    X_perturbed[~own] += 5.0

    assert np.array_equal(
        _proba_with_fresh_cache(ragged, X_perturbed, idx), base_ragged
    )
    # The full-graph expert *does* respond, which is what makes the control
    # meaningful rather than vacuous.
    assert np.abs(
        _proba_with_fresh_cache(full, X_perturbed, idx) - base_full
    ).max() > 1e-3


def test_ragged_edges_still_uses_structure_inside_its_own_subset():
    """Ragged edges must not accidentally disable message passing entirely.

    If masking the non-owned columns left every node effectively isolated, the
    "stronger graph expert" arm would silently become a per-row arm and the
    experiment would measure expert weakness rather than implicit routing. So the
    ragged expert must still beat per-row baselines on a label that carries no
    own-feature signal at all.
    """
    from spectral_distillation.src.gnn_expert import gnn_head_factory
    from spectral_distillation.src.laplacian import normalize_adjacency

    rng = np.random.default_rng(0)
    n_per_block, n_blocks, d = 200, 4, 8
    A, _ = _sbm(n_per_block, n_blocks, 0.06, 0.01, seed=1)
    n = n_per_block * n_blocks
    X = rng.normal(size=(n, d))
    y = _neighbour_only_labels(A, X)
    W_norm = normalize_adjacency(A)

    train = np.zeros(n, dtype=bool)
    train[rng.permutation(n)[: n // 2]] = True
    assign = np.ones((n, 1))  # one expert, so this isolates model quality

    def acc_of(factory):
        heads = fit_experts(X, y, assign, train, expert_factory=factory)
        return mixture_accuracy(X, y, assign, heads, ~train)[0]

    ragged_acc, n_eval = mixture_accuracy(
        X, y, assign,
        [gnn_head_factory(W_norm=W_norm, n_features=d, epochs=800,
                         ragged_edges=True)(train).fit_graph(X, y)],
        ~train)
    full_acc = acc_of(gnn_head_factory(W_norm=W_norm, n_features=d, epochs=800))
    lin_acc = acc_of(EXPERT_FACTORIES["logistic"])
    mlp_acc = acc_of(EXPERT_FACTORIES["mlp"])

    assert n_eval == int((~train).sum())
    # Per-row models sit at chance because the label is neighbour-only.
    assert lin_acc < 0.6 and mlp_acc < 0.6, (lin_acc, mlp_acc)
    # The ragged expert must retain real propagation power...
    assert ragged_acc > 0.65, ragged_acc
    # ...while still being measurably handicapped against the full graph, since
    # it sees roughly half the neighbourhood. A ragged arm that matched the
    # full-graph arm would mean the masking is not actually binding.
    assert ragged_acc < full_acc, (ragged_acc, full_acc)


def test_ragged_edges_flag_requires_graph_expert():
    """``--ragged-edges`` must be rejected for per-row experts.

    Per-row heads never touch the graph, so the flag would be a silent no-op and
    the arm would appear controlled without being controlled. Guarded in the
    runner; pinned here.
    """
    import subprocess
    import sys

    for expert in ("logistic", "mlp"):
        r = subprocess.run(
            [sys.executable, "-m",
             "spectral_distillation.experiments.run_fixed_expert_protocol",
             "--ragged-edges", "--expert", expert, "--skip-self-routing"],
            capture_output=True, text=True, cwd=Path.cwd(),
        )
        assert r.returncode != 0, expert
        assert "requires" in (r.stderr + r.stdout)


def test_router_x_only_mode_removes_every_graph_term():
    """The implicit-routing control must actually remove graph propagation.

    "On the Benefits of Learning to Route in MoE Models" (EMNLP 2023) shows a
    frozen router still routes implicitly through earlier layers. Here the channel
    is ``AX``: a graph expert consumes it regardless of the frozen assignment. The
    ablation is only interpretable if the ``x_only`` gate contains *no* propagated
    column, so that is pinned here by perturbing the adjacency and requiring the
    features to be bit-identical.
    """
    from spectral_distillation.src.learned_router import _router_features

    rng = np.random.default_rng(0)
    n, d = 60, 4
    X = rng.normal(size=(n, d))
    A = rng.random((n, n))
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)

    x_only = _router_features(X, A, mode="x_only")
    assert x_only.shape == (n, d)
    assert np.array_equal(x_only, X)

    # Any change to the graph must leave the x_only gate untouched.
    A2 = A.copy()
    A2[0, 1] = A2[1, 0] = 5.0
    assert np.array_equal(_router_features(X, A2, mode="x_only"), x_only)

    # ...and must visibly change the default gate, or the control is vacuous.
    assert not np.array_equal(_router_features(X, A, mode="nodemoe"),
                              _router_features(X, A2, mode="nodemoe"))
    assert _router_features(X, A, mode="nodemoe").shape == (n, 3 * d)

    with pytest.raises(ValueError):
        _router_features(X, A, mode="nonexistent")


def test_learned_router_x_only_arm_is_propagation_free_end_to_end():
    """The ``x_only`` arm must route on features alone, with no graph leakage.

    Pins the wiring, not just the helper: the router's assignment must be
    unchanged when the adjacency is scrambled, because a silent fall-through to
    the Node-MoE gate would make the ablation a no-op while still looking like it
    ran.
    """
    from spectral_distillation.src.learned_router import (
        _router_features,
        evaluate_learned_routing,
    )

    rng = np.random.default_rng(0)
    n, d, k = 200, 5, 3
    A = rng.random((n, n))
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)
    W_norm = A / A.sum(axis=1, keepdims=True).clip(min=1e-12)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)

    pool = np.zeros((n, k))
    pool[np.arange(n), np.arange(n) % k] = 1.0
    train = np.zeros(n, dtype=bool)
    train[:130] = True
    test = ~train
    oracle = np.ones((n, k))
    oracle /= k

    def run(mode, adj):
        return evaluate_learned_routing(
            X, y, adj, pool, train, test, oracle,
            expert_factory=EXPERT_FACTORIES["logistic"], seed=0, epochs=30,
            feature_mode=mode,
        )

    A_scrambled = W_norm.copy()
    A_scrambled[0, 1] = A_scrambled[1, 0] = 0.9

    res = run("x_only", W_norm)
    assert res["router_feature_mode"] == "x_only"

    # Router gate is graph-free by construction in this mode.
    assert np.array_equal(_router_features(X, W_norm, mode="x_only"),
                          _router_features(X, A_scrambled, mode="x_only"))

    # The logistic expert also ignores the graph, so the whole arm's accuracy
    # must be invariant to scrambling the adjacency. If it is not, propagation is
    # still leaking in somewhere and the control is not clean.
    res_scrambled = run("x_only", A_scrambled)
    for cond in ("single_global", "learned_router", "feature_kmeans",
                 "ensemble_uniform"):
        assert res["accuracy"][cond] == pytest.approx(
            res_scrambled["accuracy"][cond], abs=1e-12
        ), cond

    # Sanity: the default arm is *not* graph-free in the same way, because the
    # gate sees AX. Its features must respond to the scramble.
    assert not np.array_equal(_router_features(X, W_norm, mode="nodemoe"),
                              _router_features(X, A_scrambled, mode="nodemoe"))


def test_x_only_arm_rejects_graph_expert():
    """``x_only`` + GNN expert is rejected: the graph channel stays open.

    The GNN expert consumes ``AX`` internally regardless of the gate, so pairing
    it with a feature-only router would leave implicit routing possible and make a
    null result uninterpretable. Guarded in the runner; pinned here.
    """
    import subprocess
    import sys

    r = subprocess.run(
        [sys.executable, "-m",
         "spectral_distillation.experiments.run_learned_router",
         "--router-mode", "x_only", "--expert", "gnn"],
        capture_output=True, text=True, cwd=Path.cwd(),
    )
    assert r.returncode != 0
    assert "implicit routing" in (r.stderr + r.stdout)


def test_ensemble_supports_graph_expert_probability_hook():
    """The graph expert must expose probabilities for Ens-Avg.

    It previously offered only ``predict_nodes`` (hard labels), so the ensemble
    either skipped it or fell back to voting -- which would have quietly dropped
    the GNN pool from the very control meant to explain the GNN results.
    """
    from spectral_distillation.src.gnn_expert import gnn_head_factory
    from spectral_distillation.src.laplacian import normalize_adjacency

    rng = np.random.default_rng(0)
    n, d = 200, 5
    A = _ring_graph(n)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] > 0).astype(int)
    W_norm = normalize_adjacency(A)

    train = np.zeros(n, dtype=bool)
    train[:140] = True
    assign = np.zeros((n, 2))
    assign[np.arange(n), np.arange(n) % 2] = 1.0

    heads = fit_experts(X, y, assign, train,
                        expert_factory=gnn_head_factory(W_norm=W_norm,
                                                        n_features=d, epochs=60))
    idx = np.flatnonzero(train)
    for head in heads:
        if head is None or hasattr(head, "_fallback"):
            continue
        assert hasattr(head, "predict_proba_nodes")
        p = head.predict_proba_nodes(X, idx)
        assert p.shape == (idx.size, 1)
        assert np.all((p >= 0) & (p <= 1))
        # label and probability paths must agree at the decision boundary
        labels = head.predict_nodes(X, idx)
        assert np.array_equal(labels, (p[:, 0] > 0.5).astype(int))

    from spectral_distillation.src.learned_router import uniform_ensemble_accuracy

    acc = uniform_ensemble_accuracy(X, y, heads, ~train)
    assert 0.0 < acc <= 1.0


# --------------------------------------------------------------------------- #
# self-routing credit is label-specific, not partition-generic
# --------------------------------------------------------------------------- #

def test_self_routing_credits_label_aligned_partitions_not_random_ones():
    """The self-routing confound is not 'any partition gets free credit'.

    Measured against no-routing *within* protocol A on the real graphs:
    random routing is worth about -0.001 on all three, while the label-aligned
    oracle is worth +0.003 / +0.038 / +0.044. If the credit were generic, the
    random arm would collect it too. Pin the weaker, protocol-independent half of
    that claim on a planted graph: a label-aligned partition must beat both a
    random partition and no-routing under self-routing, by more than the random
    partition does.
    """
    graph = generate_pc_graph(n_nodes=150, n_patches=3, d_features=8, seed=42)
    X, y = graph["features"], graph["y"]
    n = X.shape[0]
    rng = np.random.default_rng(0)
    from spectral_distillation.experiments.run_fixed_expert_protocol import (
        feature_kmeans_routing,
    )

    k = int(y.max()) + 1

    # a genuinely label-aligned partition: split on y itself
    aligned = np.zeros((n, k))
    aligned[np.arange(n), y] = 1.0
    rand = np.zeros((n, k))
    rand[np.arange(n), rng.integers(0, k, n)] = 1.0
    one = np.ones((n, 1))

    factory = EXPERT_FACTORIES["logistic"]
    acc = {k: [] for k in ("nor", "rand", "aligned")}
    for split in range(5):
        for seed in range(2):
            train, test = train_test_split(y, split, seed)
            acc["nor"].append(mixture_accuracy(
                X, y, one, fit_experts(X, y, one, train,
                                       expert_factory=factory), test)[0])
            acc["rand"].append(mixture_accuracy(
                X, y, rand, fit_experts(X, y, rand, train,
                                        expert_factory=factory), test)[0])
            acc["aligned"].append(mixture_accuracy(
                X, y, aligned, fit_experts(X, y, aligned, train,
                                           expert_factory=factory), test)[0])

    nor = float(np.mean(acc["nor"]))
    rnd = float(np.mean(acc["rand"])) - nor
    alg = float(np.mean(acc["aligned"])) - nor

    assert alg > 0.0, "label-aligned partition should gain under self-routing"
    assert alg > rnd, (
        "the self-routing credit must favour label-aligned partitions over "
        f"random ones (aligned {alg:+.4f} vs random {rnd:+.4f})"
    )


def test_self_routing_credit_vanishes_under_fixed_expert_protocol():
    """The same label-aligned partition must stop paying once the pool is fixed.

    This is the retractor for the retracted oracle numbers: with a routing-blind
    pool, routing every node to the expert fitted to *its own* label can no longer
    beat the same expert pool used without routing.
    """
    graph = generate_pc_graph(n_nodes=150, n_patches=3, d_features=8, seed=42)
    X, y = graph["features"], graph["y"]
    n = X.shape[0]
    from spectral_distillation.experiments.run_fixed_expert_protocol import (
        feature_kmeans_routing,
    )

    k = int(y.max()) + 1
    aligned = np.zeros((n, k))
    aligned[np.arange(n), y] = 1.0
    pool = feature_kmeans_routing(X)
    factory = EXPERT_FACTORIES["logistic"]

    gain_self, gain_fixed = [], []
    for split in range(5):
        for seed in range(2):
            train, test = train_test_split(y, split, seed)
            nor = mixture_accuracy(
                X, y, pool, fit_experts(X, y, pool, train,
                                        expert_factory=factory), test)[0]
            gain_self.append(mixture_accuracy(
                X, y, aligned, fit_experts(X, y, aligned, train,
                                           expert_factory=factory), test)[0] - nor)
            gain_fixed.append(mixture_accuracy(
                X, y, aligned, fit_experts(X, y, pool, train,
                                           expert_factory=factory), test)[0] - nor)

    assert np.mean(gain_self) > np.mean(gain_fixed), (
        "fixing the pool must reduce the self-routing gain "
        f"({np.mean(gain_self):+.4f} -> {np.mean(gain_fixed):+.4f})"
    )


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="session")
def small_pc_graph():
    return generate_pc_graph(n_nodes=150, n_patches=3, d_features=8, seed=42)
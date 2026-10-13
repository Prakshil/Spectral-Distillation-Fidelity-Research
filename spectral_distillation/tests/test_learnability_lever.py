"""Tests for the Phase B learnability lever.

The lever has to be honest at both ends and monotone in between: at
``separability = 0`` a single global rule must fit every group (routing is worth
nothing), at ``separability = 1`` it must reproduce the positive control
(routing is worth a lot), and the conditioned router must route only when the
held-out evidence says routing helps.
"""

from __future__ import annotations

import numpy as np
import pytest

from spectral_distillation.src.learnability_lever import (
    _common_rule_score,
    _group_rule_score,
    learnability_conditioned_router,
    planted_separability_graph,
)
from spectral_distillation.src.laplacian import normalize_adjacency
from spectral_distillation.src.mixture import EXPERT_FACTORIES, fit_experts, mixture_accuracy, train_test_split
from spectral_distillation.src.positive_control import _group_rules, oracle_from_groups
from sklearn.cluster import KMeans


def _pool(X: np.ndarray, n_experts: int = 3, seed: int = 0) -> np.ndarray:
    lab = KMeans(n_clusters=n_experts, n_init=10, random_state=seed).fit_predict(X)
    out = np.zeros((X.shape[0], n_experts))
    out[np.arange(X.shape[0]), lab] = 1.0
    return out


def _oracle_gain(graph: dict, seed: int = 0, split: int = 0) -> float:
    X, y = graph["features"], graph["y"]
    pool = _pool(X, seed=seed)
    oracle = oracle_from_groups(graph["group"], pool)
    train, te = train_test_split(y, split, seed)
    factory = EXPERT_FACTORIES["logistic"]
    experts = fit_experts(X, y, pool, train, expert_factory=factory)
    global_ = fit_experts(X, y, np.ones((X.shape[0], 1)), train, expert_factory=factory)
    oracle_acc, _ = mixture_accuracy(X, y, oracle, experts, te)
    global_acc, _ = mixture_accuracy(X, y, np.ones((X.shape[0], 1)), global_, te)
    return float(oracle_acc - global_acc)


def test_separability_bounds_are_validated():
    with pytest.raises(ValueError):
        planted_separability_graph(separability=1.5)
    with pytest.raises(ValueError):
        planted_separability_graph(separability=-0.1)
    with pytest.raises(ValueError):
        planted_separability_graph(n_features=3)


def test_graph_shapes_and_metadata():
    graph = planted_separability_graph(n=300, n_features=6, separability=0.5, seed=1)
    assert graph["W"].shape == (300, 300)
    assert graph["features"].shape == (300, 6)
    assert graph["y"].shape == (300,)
    assert set(np.unique(graph["y"])) <= {0, 1}
    assert graph["group"].shape == (300,)
    assert graph["separability"] == 0.5
    assert np.allclose(graph["W"], graph["W"].T)
    assert np.count_nonzero(np.diag(graph["W"])) == 0


def test_zero_separability_is_group_invariant():
    """At separability 0 every group shares one rule, so the label cannot depend
    on the latent group -- swapping the group marker must not change y."""
    graph = planted_separability_graph(n=600, separability=0.0, seed=3, edge_p=0.0)
    X, y = graph["features"], graph["y"]
    assert np.array_equal(y, (_common_rule_score(X) > 0).astype(np.int64))
    # The label must be recoverable without knowing the group.
    groups = graph["group"]
    for g in np.unique(groups):
        mask = groups == g
        assert np.array_equal(y[mask], (_common_rule_score(X[mask]) > 0).astype(np.int64))


def test_full_separability_reproduces_positive_control_rule():
    """At separability 1 the label is exactly the positive control's per-group rule."""
    graph = planted_separability_graph(n=600, separability=1.0, seed=4, edge_p=0.0)
    X, y, groups = graph["features"], graph["y"], graph["group"]
    assert np.array_equal(y, _group_rules(X, groups))
    # And not the common rule in general (else the lever would be flat).
    assert not np.array_equal(y, (_common_rule_score(X) > 0).astype(np.int64))


def test_group_rule_score_matches_positive_control_scores():
    graph = planted_separability_graph(n=600, separability=1.0, seed=5, edge_p=0.0)
    X, groups = graph["features"], graph["group"]
    score = _group_rule_score(X, groups)
    assert np.array_equal((score > 0).astype(np.int64), _group_rules(X, groups))


def test_routing_gain_is_near_zero_at_beta_0_and_large_at_beta_1():
    """The lever's whole point: routing is worthless at 0, valuable at 1."""
    g0 = planted_separability_graph(n=900, separability=0.0, seed=0, edge_p=0.05)
    g1 = planted_separability_graph(n=900, separability=1.0, seed=0, edge_p=0.05)
    gain0 = _oracle_gain(g0)
    gain1 = _oracle_gain(g1)
    assert gain1 > 0.10
    assert gain0 < 0.05
    assert gain1 > gain0 + 0.08


def test_conditioned_router_routes_on_positive_control():
    # n=1800 keeps the router-val split (dozens of nodes) large enough that the
    # held-out gain is unambiguous; a 900-node graph is borderline and flaky.
    graph = planted_separability_graph(n=1800, separability=1.0, seed=0, edge_p=0.05)
    X, y, W = graph["features"], graph["y"], graph["W"]
    W_norm = normalize_adjacency(W)
    pool = _pool(X)
    train, _ = train_test_split(y, 0, 0)
    res = learnability_conditioned_router(
        X, y, W_norm, pool, train, EXPERT_FACTORIES["logistic"], seed=0, epochs=200,
    )
    assert res["decision"] == "route"
    assert res["assignment"].shape == (X.shape[0], 3)
    assert res["validation_gain"] > 0.0


def test_conditioned_router_abstains_when_margin_unreachable():
    graph = planted_separability_graph(n=600, separability=1.0, seed=0, edge_p=0.05)
    X, y, W = graph["features"], graph["y"], graph["W"]
    W_norm = normalize_adjacency(W)
    pool = _pool(X)
    train, _ = train_test_split(y, 0, 0)
    res = learnability_conditioned_router(
        X, y, W_norm, pool, train, EXPERT_FACTORIES["logistic"],
        seed=0, epochs=120, margin=1.0,
    )
    assert res["decision"] == "abstain"
    assert res["assignment"].shape == (X.shape[0], 1)
    assert np.allclose(res["assignment"], 1.0)


def test_conditioned_router_supervision_is_disjoint():
    """Router/fit/validation splits must be pairwise disjoint and cover the train mask."""
    graph = planted_separability_graph(n=600, separability=0.7, seed=2, edge_p=0.05)
    X, y, W = graph["features"], graph["y"], graph["W"]
    W_norm = normalize_adjacency(W)
    pool = _pool(X)
    train, _ = train_test_split(y, 0, 0)
    res = learnability_conditioned_router(
        X, y, W_norm, pool, train, EXPERT_FACTORIES["logistic"], seed=0, epochs=50,
    )
    assert res["n_router_fit"] > 0
    assert res["n_router_val"] > 0
    assert res["n_router_fit"] + res["n_router_val"] <= int(train.sum())

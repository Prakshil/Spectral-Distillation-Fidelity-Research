"""Tests for the published-router reproduction (Phase A).

Covers the two routers named in the implementation guide's gap analysis:

* RouterGNN -- label-supervised message-passing router scored under Protocol B;
* AD-GNN -- per-node propagation depth via the signal-preservation factor.

Invariants pinned here mirror the RouterGNN-lite tests in ``test_protocol.py``:
supervision is cross-fit (``router_mask`` only), test labels are invisible, and
the routers recover real routing gain on the positive control.
"""

from __future__ import annotations

import numpy as np
import pytest

from spectral_distillation.src.learned_router import (
    build_router_targets,
    split_fit_router,
)
from spectral_distillation.src.mixture import EXPERT_FACTORIES, fit_experts
from spectral_distillation.src.published_routers import (
    ad_gnn_assignment,
    evaluate_published_routers,
    message_passing_router_assignment,
    signal_preservation_factor,
    train_message_passing_router,
)


def _homophilic_block_graph(n_per_group: int = 60, n_groups: int = 3) -> dict:
    """Block-diagonal graph where neighbours share features exactly."""
    rng = np.random.default_rng(0)
    n = n_per_group * n_groups
    X = rng.normal(size=(n, 4))
    groups = np.repeat(np.arange(n_groups), n_per_group)
    X[:, -1] = groups  # marker separates groups
    y = (groups % 2).astype(np.int64)
    W = np.zeros((n, n))
    for g in range(n_groups):
        idx = np.flatnonzero(groups == g)
        block = rng.random((idx.size, idx.size)) < 0.15
        np.fill_diagonal(block, False)
        W[np.ix_(idx, idx)] = block
    W = np.maximum(W, W.T)
    return {"W": W, "features": X, "y": y}


def test_signal_preservation_factor_matches_direct_computation():
    from spectral_distillation.src.laplacian import normalize_adjacency

    graph = _homophilic_block_graph(n_per_group=30)
    W_norm = normalize_adjacency(graph["W"])
    X = graph["features"]

    factor = signal_preservation_factor(X, W_norm, depths=(1, 2))

    Ax = W_norm @ X
    A2x = W_norm @ Ax
    x_norm = np.linalg.norm(X, axis=1)
    cos1 = np.einsum("ij,ij->i", X, Ax) / (x_norm * np.linalg.norm(Ax, axis=1))
    cos2 = np.einsum("ij,ij->i", X, A2x) / (x_norm * np.linalg.norm(A2x, axis=1))
    expected = np.minimum(np.clip(cos1, -1, 1), np.clip(cos2, -1, 1))
    assert np.allclose(factor, expected, atol=1e-10)


def test_signal_preservation_factor_is_deterministic_and_depth_sensitive():
    from spectral_distillation.src.laplacian import normalize_adjacency

    graph = _homophilic_block_graph(n_per_group=40)
    W_norm = normalize_adjacency(graph["W"])
    X = graph["features"]

    f_12 = signal_preservation_factor(X, W_norm, depths=(1, 2))
    f_1 = signal_preservation_factor(X, W_norm, depths=(1,))
    assert np.allclose(f_12, signal_preservation_factor(X, W_norm, depths=(1, 2)))
    # Surviving deeper smoothing is a harder criterion, never a lower score.
    assert np.all(f_12 <= f_1 + 1e-12)
    with pytest.raises(ValueError):
        signal_preservation_factor(X, W_norm, depths=(0,))


def test_ad_gnn_assignment_is_a_valid_deterministic_routing():
    from spectral_distillation.src.laplacian import normalize_adjacency

    graph = _homophilic_block_graph(n_per_group=40)
    X = graph["features"]
    W_norm = normalize_adjacency(graph["W"])

    a1 = ad_gnn_assignment(X, W_norm, n_experts=3)
    a2 = ad_gnn_assignment(X, W_norm, n_experts=3)
    assert a1.shape == (X.shape[0], 3)
    assert np.allclose(a1, a2)
    assert np.allclose(a1.sum(axis=1), 1.0)
    assert np.all(np.isin(a1, (0.0, 1.0)))
    # ad_gnn_assignment takes (X, W_norm) only: no label can enter the routing.


def test_ad_gnn_high_preservation_nodes_route_to_low_pass_channel():
    from spectral_distillation.src.laplacian import normalize_adjacency

    graph = _homophilic_block_graph(n_per_group=50, n_groups=3)
    X = graph["features"]
    W_norm = normalize_adjacency(graph["W"])

    factor = signal_preservation_factor(X, W_norm, depths=(1, 2))
    a = ad_gnn_assignment(X, W_norm, n_experts=3)
    high = factor > np.quantile(factor, 0.9)
    assert a[high].argmax(axis=1).mean() < 0.5  # mostly channel 0 (low-pass/deep)


def test_router_gnn_forward_depends_on_adjacency():
    """Message passing must actually consult the graph (architecture differs from MLP)."""
    import torch

    from spectral_distillation.src.laplacian import normalize_adjacency
    from spectral_distillation.src.learned_router import _router_features
    from spectral_distillation.src.mixture import train_test_split
    from spectral_distillation.src.published_routers import (
        train_message_passing_router,
    )
    from spectral_distillation.experiments.run_fixed_expert_protocol import (
        feature_kmeans_routing,
    )

    graph = _homophilic_block_graph(n_per_group=60)
    X, y = graph["features"], graph["y"]
    W_norm = normalize_adjacency(graph["W"])
    train, _ = train_test_split(y, 0, 0)
    fit_mask, router_mask = split_fit_router(train, y, seed=0)
    pool = feature_kmeans_routing(X, n_experts=3)
    experts = fit_experts(X, y, pool, fit_mask, expert_factory=EXPERT_FACTORIES["logistic"])
    router_idx = np.flatnonzero(router_mask)
    targets = build_router_targets(y, experts, X, router_idx)

    model, _ = train_message_passing_router(
        X, W_norm, targets, router_idx, 3, epochs=20, seed=0
    )
    a1 = message_passing_router_assignment(model, X, W_norm)

    perm = np.random.default_rng(2).permutation(X.shape[0])
    W_perm = W_norm[np.ix_(perm, perm)]
    a2 = message_passing_router_assignment(model, X, W_perm)

    assert a1.shape == (X.shape[0], 3)
    assert not np.array_equal(a1, a2), (
        "RouterGNN forward ignores the adjacency; message passing is not engaged."
    )


def test_message_passing_router_recovers_gain_on_positive_control():
    from spectral_distillation.experiments.run_fixed_expert_protocol import (
        feature_kmeans_routing,
    )
    from spectral_distillation.src.laplacian import normalize_adjacency
    from spectral_distillation.src.mixture import train_test_split
    from spectral_distillation.src.positive_control import (
        generate_positive_control,
        oracle_from_groups,
    )

    graph = generate_positive_control(n=900, seed=0)
    X, y = graph["features"], graph["y"]
    W_norm = normalize_adjacency(graph["W"])
    pool = feature_kmeans_routing(X, n_experts=3)
    oracle = oracle_from_groups(graph["group"], pool)
    train, test = train_test_split(y, 0, 0)

    res = evaluate_published_routers(
        X, y, W_norm, pool, train, test, oracle,
        expert_factory=EXPERT_FACTORIES["logistic"], seed=0, epochs=60,
    )
    acc = res["accuracy"]
    assert acc["single_global"] < 0.90
    # A label-supervised message-passing router must find the structure the
    # positive control guarantees is there.
    assert acc["router_gnn"] > acc["single_global"] + 0.05, acc
    assert acc["router_gnn"] > acc["random"] + 0.05, acc
    # The MLP gate (existing baseline) must report the same win in this harness.
    assert acc["mlp_gate"] > acc["single_global"] + 0.05, acc


def test_message_passing_router_never_sees_test_labels():
    import torch

    from spectral_distillation.experiments.run_fixed_expert_protocol import (
        feature_kmeans_routing,
    )
    from spectral_distillation.src.laplacian import normalize_adjacency
    from spectral_distillation.src.mixture import train_test_split
    from spectral_distillation.src.positive_control import generate_positive_control

    graph = generate_positive_control(n=600, seed=1)
    X, y = graph["features"], graph["y"]
    W_norm = normalize_adjacency(graph["W"])
    train, test = train_test_split(y, 0, 0)
    fit_mask, router_mask = split_fit_router(train, y, seed=0)
    assert not (router_mask & test).any()
    assert not (fit_mask & test).any()

    pool = feature_kmeans_routing(X, n_experts=3)
    experts = fit_experts(X, y, pool, fit_mask, expert_factory=EXPERT_FACTORIES["logistic"])
    router_idx = np.flatnonzero(router_mask)
    targets = build_router_targets(y, experts, X, router_idx)
    m1, _ = train_message_passing_router(
        X, W_norm, targets, router_idx, 3, epochs=30, seed=0
    )

    y_flip = y.copy()
    y_flip[test] = 1 - y_flip[test]
    experts_flip = fit_experts(X, y_flip, pool, fit_mask,
                               expert_factory=EXPERT_FACTORIES["logistic"])
    targets_flip = build_router_targets(y_flip, experts_flip, X, router_idx)
    m2, _ = train_message_passing_router(
        X, W_norm, targets_flip, router_idx, 3, epochs=30, seed=0
    )

    assert np.allclose(targets, targets_flip)
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(p1, p2)


def test_evaluate_published_routers_reports_all_conditions():
    from spectral_distillation.experiments.run_fixed_expert_protocol import (
        feature_kmeans_routing,
    )
    from spectral_distillation.src.laplacian import normalize_adjacency
    from spectral_distillation.src.mixture import train_test_split

    graph = _homophilic_block_graph(n_per_group=60)
    X, y = graph["features"], graph["y"]
    W_norm = normalize_adjacency(graph["W"])
    pool = feature_kmeans_routing(X, n_experts=3)
    from spectral_distillation.src.router_protocol import oracle_bucket_assignment

    oracle = oracle_bucket_assignment(graph["W"], y, adaptive=True)
    train, test = train_test_split(y, 0, 0)
    res = evaluate_published_routers(
        X, y, W_norm, pool, train, test, oracle,
        expert_factory=EXPERT_FACTORIES["logistic"], seed=0, epochs=20,
    )

    expected = {
        "single_global", "mlp_gate", "router_gnn", "ad_gnn",
        "feature_kmeans", "oracle", "random", "ensemble_uniform",
    }
    assert set(res["accuracy"]) == expected
    for name, v in res["accuracy"].items():
        assert 0.0 <= v <= 1.0, (name, v)
    for field in (
        "router_gnn_minus_no_routing",
        "ad_gnn_minus_no_routing",
        "mlp_gate_minus_no_routing",
        "ensemble_minus_no_routing",
    ):
        assert np.isfinite(res[field])
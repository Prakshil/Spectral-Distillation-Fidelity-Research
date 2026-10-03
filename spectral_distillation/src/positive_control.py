"""Synthetic positive control for the fixed-expert routing protocol.

Every real dataset in this repo reports the same thing: no routing beats a single
global model. That result is only informative if the harness *can* detect a win.
This module supplies a case where it must.

Design
------
Three latent groups. Within each group the label is a *linearly separable* rule
over its own feature pair. No single hyperplane satisfies all three rules
simultaneously, because the three rules point in different directions over the
same marginal feature distribution. So the label function is a non-separable
mixture of linear rules: too complex for one global linear model, trivial for
three local ones.

A marker feature is offset per group (5 / -5 / 0) so the groups are separated in
feature space. Two consequences, both deliberate:

* feature k-means recovers the latent groups, so the protocol's routing-blind
  expert pool is aligned with the true structure;
* the graph is block-diagonal over the same groups, so label homophily is high
  and the task is graph-native rather than a pure-features problem.

Expected outcome: with logistic experts, oracle routing and k-means routing both
land far above the global model, while random routing does not. That is the
positive control -- the protocol reports a large routing win when one exists.
"""

from __future__ import annotations

import numpy as np

N_GROUPS = 3


def _group_rules(X: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Per-group linear rules, each over a different pair of shared features."""
    scores = np.zeros(groups.shape[0], dtype=np.int64)
    scores[groups == 0] = X[groups == 0, 0] + X[groups == 0, 1]
    scores[groups == 1] = X[groups == 1, 1] - X[groups == 1, 2]
    scores[groups == 2] = X[groups == 2, 2] - X[groups == 2, 0]
    return (scores > 0).astype(np.int64)


def generate_positive_control(
    n: int = 1800,
    n_features: int = 4,
    seed: int = 0,
    edge_p: float = 0.02,
    n_groups: int = N_GROUPS,
) -> dict:
    """Build a graph dict shaped like ``load_real_fraud`` output.

    Returns the usual ``{"W", "features", "y", "d_features"}`` keys plus
    ``group`` (the latent partition) and ``marker_offsets``, which the experiment
    script uses to score the oracle routing without refitting anything.
    """
    if n_features < 4:
        raise ValueError("positive control needs at least 4 features (3 rules + 1 marker)")
    rng = np.random.default_rng(seed)
    sizes = np.full(n_groups, n // n_groups, dtype=np.int64)
    sizes[: n % n_groups] += 1
    groups = np.repeat(np.arange(n_groups), sizes)

    features = rng.normal(0.0, 1.0, size=(n, n_features))
    marker_offsets = np.array([5.0, -5.0, 0.0][:n_groups])
    features[:, -1] = marker_offsets[groups]
    y = _group_rules(features, groups)

    W = np.zeros((n, n), dtype=np.float64)
    for g in range(n_groups):
        idx = np.flatnonzero(groups == g)
        block = rng.random((idx.size, idx.size)) < edge_p
        np.fill_diagonal(block, False)
        W[np.ix_(idx, idx)] = block
    W = np.maximum(W, W.T)

    return {
        "W": W,
        "features": features,
        "y": y,
        "d_features": n_features,
        "group": groups,
        "marker_offsets": marker_offsets,
        "name": f"positive_control_n{n}_d{n_features}_seed{seed}",
    }


def oracle_from_groups(
    groups: np.ndarray,
    expert_assignment: np.ndarray,
) -> np.ndarray:
    """Turn the latent partition into one-hot routing over a fixed expert pool.

    The pool is routing-blind (feature k-means). Each latent group is mapped to
    the pool column that covers most of its nodes, so the returned matrix routes
    every node to the expert that actually trained on its group.
    """
    n_groups = int(groups.max()) + 1
    n_experts = expert_assignment.shape[1]
    pool_labels = expert_assignment.argmax(axis=1)
    mapping = np.empty(n_groups, dtype=np.int64)
    for g in range(n_groups):
        counts = np.bincount(pool_labels[groups == g], minlength=n_experts)
        mapping[g] = int(counts.argmax())
    out = np.zeros((expert_assignment.shape[0], n_experts), dtype=np.float64)
    out[np.arange(expert_assignment.shape[0]), mapping[groups]] = 1.0
    return out
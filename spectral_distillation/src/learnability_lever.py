"""Learnability lever: sweep routing gain from impossible to trivial.

Every real dataset in this repo reports the same null: no routing beats a single
global model (0 wins on the fair Protocol-B rerun; the union of winning runs is a
+-0.0005 coin-flip). :mod:`spectral_distillation.src.positive_control` shows the
harness *can* report a win -- but only at one extreme, an unroutable-vs-routable
binary. This module supplies the missing continuum.

The lever
---------
A single knob ``separability`` in ``[0, 1]`` interpolates the label function
between two regimes over the *same* marginal feature distribution and the *same*
graph:

* ``separability = 0`` -- every latent group shares one global rule, so a single
  linear hyperplane is optimal and routing is worth exactly nothing;
* ``separability = 1`` -- each group uses its own rule (the positive control), so
  no global hyperplane fits and routing is worth a lot.

Formally, with ``s_common(x) = x0 + x1`` and the per-group rules of
:func:`spectral_distillation.src.positive_control._group_rules`, the score is

    s_g(x) = (1 - separability) * s_common(x) + separability * s_rule(g, x),

and the label is ``s_g(x) > 0``. At the endpoints this reduces exactly to the two
known regimes, so the lever only interpolates; it does not invent structure.

Why this matters for the paper
------------------------------
The null is only a *routing* statement if routing was ever learnable on those
pools. The lever converts the null into a **learnability** statement: routing
becomes learnable above a threshold ``separability*``, and the real fraud /
heterophily pools sit below it. The companion
:func:`learnability_conditioned_router` is the constructive half -- a selective
router that measurably *abstains* (routes everything to the global head) whenever
the task is below the lever threshold, so "no win" becomes "correctly declined",
not "failed to find one".
"""

from __future__ import annotations

import numpy as np

N_GROUPS = 3


def _common_rule_score(X: np.ndarray) -> np.ndarray:
    """The single global rule shared by every group at ``separability = 0``."""
    return X[:, 0] + X[:, 1]


def _group_rule_score(X: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Per-group linear rules, each over a different pair of shared features.

    Reproduces :func:`spectral_distillation.src.positive_control._group_rules`
    *exactly*: same per-group feature pairs, same sign threshold. The scores are
    accumulated in float (not ``int64``) so the sign test is applied to the true
    value of the linear rule; truncating toward zero would flip the label of any
    node whose score falls in ``(0, 1)``. This makes ``separability = 1``
    bit-identical to the validated positive control.
    """
    scores = np.zeros(X.shape[0], dtype=float)
    scores[groups == 0] = X[groups == 0, 0] + X[groups == 0, 1]
    scores[groups == 1] = X[groups == 1, 1] - X[groups == 1, 2]
    scores[groups == 2] = X[groups == 2, 2] - X[groups == 2, 0]
    return scores


def planted_separability_graph(
    n: int = 3000,
    n_features: int = 8,
    seed: int = 0,
    edge_p: float = 0.02,
    cross_edge_p: float = 0.0,
    n_groups: int = N_GROUPS,
    separability: float = 1.0,
) -> dict:
    """A graph whose routing difficulty is set by one continuous knob.

    Returns the usual ``{"W", "features", "y", "d_features"}`` keys (shaped like
    ``load_real_fraud`` output) plus ``group`` (the latent partition),
    ``separability``, and ``marker_offsets``. At ``separability = 1`` it is the
    positive control (:mod:`spectral_distillation.src.positive_control`); at
    ``separability = 0`` every group obeys the same global rule and routing adds
    nothing.

    ``cross_edge_p`` is the homophily lever: probability of adding a random
    cross-group edge on top of the block-diagonal within-group graph. ``0.0`` is
    perfectly homophilous; raising it dilutes the graph structure independently
    of the label separability.
    """
    if not 0.0 <= separability <= 1.0:
        raise ValueError(f"separability must be in [0, 1], got {separability}")
    if n_features < 4:
        raise ValueError("planted graph needs at least 4 features (3 rules + 1 marker)")

    rng = np.random.default_rng(seed)
    sizes = np.full(n_groups, n // n_groups, dtype=np.int64)
    sizes[: n % n_groups] += 1
    groups = np.repeat(np.arange(n_groups), sizes)

    features = rng.normal(0.0, 1.0, size=(n, n_features))
    marker_offsets = np.array([5.0, -5.0, 0.0][:n_groups])
    features[:, -1] = marker_offsets[groups]

    score = (
        (1.0 - separability) * _common_rule_score(features)
        + separability * _group_rule_score(features, groups)
    )
    y = (score > 0).astype(np.int64)

    W = np.zeros((n, n), dtype=np.float64)
    for g in range(n_groups):
        idx = np.flatnonzero(groups == g)
        block = rng.random((idx.size, idx.size)) < edge_p
        np.fill_diagonal(block, False)
        W[np.ix_(idx, idx)] = block
    if cross_edge_p > 0.0:
        cross = rng.random((n, n)) < cross_edge_p
        np.fill_diagonal(cross, False)
        # Keep only cross-group edges so within-group density stays at edge_p.
        same_group = groups[:, None] == groups[None, :]
        cross = cross & ~same_group
        W = np.maximum(W, cross.astype(np.float64))
    W = np.maximum(W, W.T)

    return {
        "W": W,
        "features": features,
        "y": y,
        "d_features": n_features,
        "group": groups,
        "separability": float(separability),
        "marker_offsets": marker_offsets,
        "name": f"separability_lever_beta{separability:.2f}_n{n}_d{n_features}_seed{seed}",
    }


def learnability_conditioned_router(
    X: np.ndarray,
    y: np.ndarray,
    W_norm: np.ndarray,
    pool: np.ndarray,
    train_mask: np.ndarray,
    expert_factory,
    seed: int = 0,
    router_frac: float = 0.35,
    val_frac: float = 0.4,
    margin: float = 0.0,
    epochs: int = 200,
    device: str = "cpu",
) -> dict:
    """Selective router that routes only when routing is learnable.

    The deployable question is not "does routing win?" but "can we *tell* whether
    it will, using only held-out supervision of the same kind the MLP gate already
    gets?" This router answers it with an internal validation split:

    1. carve the training nodes into expert-fit and router-supervision halves
       (the usual cross-fit protocol);
    2. carve the router half again into ``router_fit`` / ``router_val``;
    3. fit the frozen pool experts and the global expert on the fit half;
    4. train the MLP gate on ``router_fit`` and score the *routed* mixture vs the
       *global* head on the disjoint ``router_val`` nodes;
    5. route only if that held-out gain exceeds ``margin``; otherwise abstain
       (route every node to the global head).

    Returns the chosen ``assignment`` (one column when abstaining), the
    ``decision`` string, the held-out ``validation_gain``, and the fitted experts
    so the caller can score the final assignment without refitting.
    """
    from spectral_distillation.src.learned_router import (
        build_router_targets,
        router_assignment,
        split_fit_router,
        train_router,
    )
    from spectral_distillation.src.mixture import fit_experts, mixture_accuracy

    fit_mask, router_mask = split_fit_router(
        train_mask, y, seed=seed, router_frac=router_frac
    )
    rfit, rval = split_fit_router(
        router_mask, y, seed=seed + 101, router_frac=val_frac
    )

    experts = fit_experts(X, y, pool, fit_mask, expert_factory=expert_factory)
    global_experts = fit_experts(
        X, y, np.ones((X.shape[0], 1)), fit_mask, expert_factory=expert_factory
    )

    n_experts = pool.shape[1]
    rfit_idx = np.flatnonzero(rfit)
    targets = build_router_targets(y, experts, X, rfit_idx)
    model, losses = train_router(
        X, W_norm, targets, rfit_idx, n_experts,
        epochs=epochs, seed=seed, device=device,
    )
    learned = router_assignment(model, X, W_norm, device=device)

    routed_val, _ = mixture_accuracy(X, y, learned, experts, rval)
    global_val, _ = mixture_accuracy(
        X, y, np.ones((X.shape[0], 1)), global_experts, rval
    )
    validation_gain = float(routed_val - global_val)

    if validation_gain > margin:
        assignment = learned
        decision = "route"
    else:
        assignment = np.ones((X.shape[0], 1))
        decision = "abstain"

    return {
        "assignment": assignment,
        "decision": decision,
        "validation_gain": validation_gain,
        "routed_val_acc": float(routed_val),
        "global_val_acc": float(global_val),
        "experts": experts,
        "global_experts": global_experts,
        "n_router_fit": int(rfit.sum()),
        "n_router_val": int(rval.sum()),
        "gate_loss_final": float(losses[-1]) if losses else float("nan"),
    }

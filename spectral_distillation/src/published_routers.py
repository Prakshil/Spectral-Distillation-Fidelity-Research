"""Published learned routers, scored under the honest fixed-expert Protocol B.

References
----------
The negative routing result in this repo is scored against hand-designed,
label-free scores (k-means, spectral, degree, random, oracle) plus a small MLP
gate. A reviewer is entitled to object that the *published* routers were never
run, so the negative is against strawmen. This module reproduces the two the
implementation guide (guide 11.5) names:

* **RouterGNN** -- a message-passing (GraphSAGE-style) router that mixes each
  node's gate features with its neighbours' before projecting to expert weights.
  The repo already carries the architecture
  (:class:`spectral_distillation.src.gnn_router.RouterGNN`) but only ever used it
  to *imitate* a label-free clustering; here it is trained from training labels
  and scored as a routing, exactly like the MLP gate (RouterGNN-lite) in
  :mod:`spectral_distillation.src.learned_router`. Holding the gate features fixed
  and varying only the router architecture isolates "does message passing in the
  router buy anything".

* **AD-GNN / Ada-Routing** (Hevapathige et al., 2026; guide 11.5: "per-node
  propagation depth via signal-preservation factor"). Each node is assigned the
  propagation depth whose aggregated representation best preserves its own
  feature signal; high-preservation (deep-smoothing-survives) nodes map to the
  low-pass channel, low-preservation nodes to the identity channel. Fully
  label-free and deterministic.

Both routers are scored on the *same* frozen routing-blind expert pool and the
*same* split as every baseline, with the same disjoint fit/router cross-fit
discipline as ``evaluate_learned_routing``: experts are fitted on ``fit_mask``,
router supervision targets are built on a disjoint ``router_mask``. No test
label is ever visible to a router.
"""

from __future__ import annotations

import numpy as np
import torch

from spectral_distillation.src.learned_router import (
    _router_features,
    build_router_targets,
    router_assignment,
    split_fit_router,
    train_router,
    uniform_ensemble_accuracy,
)
from spectral_distillation.src.mixture import fit_experts, mixture_accuracy


# --------------------------------------------------------------------------- #
# AD-GNN / Ada-Routing: per-node propagation depth via signal preservation.
# --------------------------------------------------------------------------- #


def signal_preservation_factor(
    X: np.ndarray,
    W_norm: np.ndarray,
    depths: tuple[int, ...] = (1, 2),
) -> np.ndarray:
    """Label-free per-node signal-preservation factor across propagation depths.

    For each depth ``d`` this is the cosine similarity between a node's own
    feature vector ``x_v`` and its ``d``-hop aggregated feature ``(A^d X)_v``:
    how much of the node's own signal *survives* smoothing by ``d`` rounds of
    neighbourhood averaging. High values mean the neighbourhood reinforces the
    node (homophilic regime, where deeper propagation helps); low/negative values
    mean smoothing destroys the signal (heterophilic regime, where the node should
    stay identity).

    The factor returned is the **minimum** over ``depths`` -- the signal that
    survives to the deepest hop requested -- which is the quantity AD-GNN uses to
    choose a per-node depth. Nodes with no feature energy (``||x_v|| = 0``) get
    ``0`` and therefore fall to the identity channel.

    Uses **no labels**, so it is a genuine label-free structural router.
    """
    X = np.asarray(X, dtype=float)
    A = np.asarray(W_norm, dtype=float)
    if X.size == 0:
        return np.zeros(X.shape[0])
    depths = tuple(sorted(set(int(d) for d in depths)))
    if not depths or min(depths) < 1:
        raise ValueError(f"depths must be positive integers, got {depths!r}")

    x_norm = np.linalg.norm(X, axis=1)
    Ax = X
    factor: np.ndarray | None = None
    for d in range(1, max(depths) + 1):
        Ax = A @ Ax
        if d not in depths:
            continue
        denom = x_norm * np.linalg.norm(Ax, axis=1)
        cos = np.divide(
            np.einsum("ij,ij->i", X, Ax),
            denom,
            out=np.zeros(X.shape[0], dtype=float),
            where=denom > 1e-12,
        )
        cos = np.clip(cos, -1.0, 1.0)
        factor = cos if factor is None else np.minimum(factor, cos)
    if factor is None:
        factor = np.zeros(X.shape[0], dtype=float)
    return factor


def ad_gnn_assignment(
    X: np.ndarray,
    W_norm: np.ndarray,
    n_experts: int = 3,
    depths: tuple[int, ...] = (1, 2),
) -> np.ndarray:
    """AD-GNN routing: quantile-bucket the signal-preservation factor.

    Reuses :func:`~spectral_distillation.src.router_protocol._buckets_from_score`
    so the assignment differs from the oracle's only in *which* score is bucketed:
    the highest-preservation nodes go to channel 0 (deep / low-pass), the lowest
    to channel ``n_experts - 1`` (identity / no propagation), the rest to the
    middle channel.
    """
    from spectral_distillation.src.router_protocol import _buckets_from_score

    score = signal_preservation_factor(X, W_norm, depths=depths)
    return _buckets_from_score(score, n_experts)


# --------------------------------------------------------------------------- #
# RouterGNN: label-supervised message-passing router.
# --------------------------------------------------------------------------- #


def train_message_passing_router(
    X: np.ndarray,
    W_norm: np.ndarray,
    targets: np.ndarray,
    router_idx: np.ndarray,
    n_experts: int,
    hidden_dim: int = 64,
    n_layers: int = 3,
    epochs: int = 200,
    lr: float = 1e-2,
    dropout: float = 0.1,
    seed: int = 0,
    device: str = "cpu",
    feature_mode: str = "nodemoe",
):
    """Fit :class:`~spectral_distillation.src.gnn_router.RouterGNN` on router nodes.

    Identical supervision to the MLP gate: soft cross-entropy against ``targets``
    (which is aligned to ``router_idx``), restricted to ``router_idx`` so no
    unlabeled node contributes gradient and no test label can leak in. The only
    change versus the MLP gate is the architecture -- each layer mixes the node's
    representation with its neighbours' before projecting.
    """
    from spectral_distillation.src.gnn_router import RouterGNN, _as_torch_W

    torch.manual_seed(seed)
    feats = _router_features(X, W_norm, mode=feature_mode)
    model = RouterGNN(
        n_features_in=feats.shape[1],
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_experts=n_experts,
        dropout=dropout,
    ).to(device)

    ht = torch.as_tensor(feats, dtype=torch.float32, device=device)
    Wt = _as_torch_W(W_norm).to(device)
    yt = torch.as_tensor(targets, dtype=torch.float32, device=device)
    router_t = torch.as_tensor(router_idx, dtype=torch.long, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses: list[float] = []
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        picked = model(ht, Wt)[router_t]
        loss = -(yt * torch.log(picked + 1e-8)).sum(dim=1).mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return model, losses


def message_passing_router_assignment(
    model,
    X: np.ndarray,
    W_norm: np.ndarray,
    device: str = "cpu",
    feature_mode: str = "nodemoe",
) -> np.ndarray:
    """Hard one-hot routing from a trained message-passing router's argmax."""
    from spectral_distillation.src.gnn_router import _as_torch_W

    model.eval()
    with torch.no_grad():
        feats = _router_features(X, W_norm, mode=feature_mode)
        ht = torch.as_tensor(feats, dtype=torch.float32, device=device)
        Wt = _as_torch_W(W_norm).to(device)
        pi = model(ht, Wt).cpu().numpy()
    out = np.zeros((X.shape[0], pi.shape[1]), dtype=float)
    out[np.arange(X.shape[0]), pi.argmax(axis=1)] = 1.0
    return out


# --------------------------------------------------------------------------- #
# Joint Protocol-B evaluation over the same frozen pool and split.
# --------------------------------------------------------------------------- #


def evaluate_published_routers(
    X: np.ndarray,
    y: np.ndarray,
    W_norm: np.ndarray,
    pool: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    oracle: np.ndarray,
    expert_factory,
    seed: int = 0,
    router_frac: float = 0.35,
    epochs: int = 200,
    hidden_dim: int = 64,
    n_layers: int = 3,
    device: str = "cpu",
    feature_mode: str = "nodemoe",
) -> dict:
    """Score RouterGNN and AD-GNN beside every baseline on one frozen pool/split.

    All conditions share one ``fit_experts`` call for the frozen pool and one
    separate one-column fit for the no-routing reference, so the pool is identical
    across conditions and any difference is attributable to routing alone. Returns
    a dict of absolute accuracies plus paired deltas against the no-routing and
    ensemble references.
    """
    from spectral_distillation.src.router_protocol import random_assignment

    n_experts = pool.shape[1]
    fit_mask, router_mask = split_fit_router(
        train_mask, y, seed=seed, router_frac=router_frac
    )
    experts = fit_experts(X, y, pool, fit_mask, expert_factory=expert_factory)
    global_experts = fit_experts(
        X, y, np.ones((X.shape[0], 1)), fit_mask, expert_factory=expert_factory
    )

    router_idx = np.flatnonzero(router_mask)
    targets = build_router_targets(y, experts, X, router_idx)

    # RouterGNN-lite (MLP gate) -- the existing learned baseline.
    mlp_model, _ = train_router(
        X, W_norm, targets, router_idx, n_experts, epochs=epochs,
        seed=seed, device=device, feature_mode=feature_mode,
    )
    mlp_assign = router_assignment(
        mlp_model, X, W_norm, device=device, feature_mode=feature_mode
    )

    # RouterGNN -- same supervision, message-passing architecture.
    gnn_model, _ = train_message_passing_router(
        X, W_norm, targets, router_idx, n_experts, hidden_dim=hidden_dim,
        n_layers=n_layers, epochs=epochs, seed=seed, device=device,
        feature_mode=feature_mode,
    )
    gnn_assign = message_passing_router_assignment(
        gnn_model, X, W_norm, device=device, feature_mode=feature_mode
    )

    # AD-GNN -- label-free per-node propagation depth.
    ada_assign = ad_gnn_assignment(X, W_norm, n_experts=n_experts)

    conditions = {
        "single_global": np.ones((X.shape[0], 1)),
        "mlp_gate": mlp_assign,
        "router_gnn": gnn_assign,
        "ad_gnn": ada_assign,
        "feature_kmeans": pool,
        "oracle": oracle,
        "random": random_assignment(oracle, rng=seed),
    }
    acc: dict[str, float] = {}
    for name, assignment in conditions.items():
        heads = global_experts if name == "single_global" else experts
        acc[name] = mixture_accuracy(X, y, assignment, heads, test_mask)[0]
    acc["ensemble_uniform"] = uniform_ensemble_accuracy(X, y, experts, test_mask)

    def _delta(name: str) -> float:
        return float(acc[name] - acc["single_global"])

    return {
        "accuracy": acc,
        "router_gnn_minus_no_routing": _delta("router_gnn"),
        "ad_gnn_minus_no_routing": _delta("ad_gnn"),
        "mlp_gate_minus_no_routing": _delta("mlp_gate"),
        "ensemble_minus_no_routing": float(
            acc["ensemble_uniform"] - acc["single_global"]
        ),
        "router_gnn_vs_kmeans_agreement": float(
            (gnn_assign.argmax(axis=1) == pool.argmax(axis=1)).mean()
        ),
        "ad_gnn_vs_kmeans_agreement": float(
            (ada_assign.argmax(axis=1) == pool.argmax(axis=1)).mean()
        ),
        "router_feature_mode": feature_mode,
        "n_fit_nodes": int(fit_mask.sum()),
        "n_router_nodes": int(router_mask.sum()),
    }

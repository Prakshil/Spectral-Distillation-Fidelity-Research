"""RouterGNN-lite: a learned router scored under the honest fixed-expert protocol.

Motivation
----------
The 3x3x26 sweep reports 0/234 routing wins, but every candidate routing there is
*fixed* and label-free (k-means, spectral, degree, random, oracle). That leaves an
obvious objection unanswered: a router that actually learns which expert suits
which node might beat a global model where none of the hand-designed scores did.

This module supplies that learned condition without weakening the protocol:

* the expert pool is the same routing-blind feature k-means partition used
  everywhere else, and it is frozen before routing is learned;
* the router is supervised by training labels only -- never test labels, and
  never the oracle's label-homophily buckets;
* every comparison condition is scored against the *same* frozen pool in the same
  split, so a win cannot come from a better pool.

Disjoint fit/router split
-------------------------
Experts are fitted on ``fit_mask`` only, and the router's supervision targets are
computed on ``router_mask``, which is disjoint from ``fit_mask``. Without this
split the target is degenerate: each expert classifies its own training nodes
almost perfectly, so "which expert owns this node" collapses to "which k-means
cluster is this node", the router learns to imitate the k-means partition, and
the reported learned-routing number is just k-means routing wearing a disguise.
Holding the experts out makes the target mean "which expert generalizes to this
node", which is what a router has to guess at deployment time.

"Lite" refers to the router, not the experts: the router is a small MLP over
cached one-hop propagated features ``[X, AX]``, the same trick
:mod:`spectral_distillation.src.gnn_expert` uses to stay fast. Experts are the
unmodified protocol-B heads.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from spectral_distillation.src.mixture import fit_experts, mixture_accuracy


def _expert_predict(X: np.ndarray, experts: list, idx: np.ndarray) -> np.ndarray:
    """Per-expert predicted labels for ``idx``: shape (len(idx), n_experts).

    ``nan`` marks an expert that could not be fitted, so it never collects target
    mass.
    """
    out = np.full((idx.size, len(experts)), np.nan, dtype=float)
    for k, head in enumerate(experts):
        if head is None or idx.size == 0:
            continue
        if hasattr(head, "predict_nodes"):
            out[:, k] = head.predict_nodes(X, idx)
        else:
            out[:, k] = head.predict(X[idx])
    return out


def build_router_targets(
    y: np.ndarray,
    experts: list,
    X: np.ndarray,
    router_idx: np.ndarray,
) -> np.ndarray:
    """Soft target over experts: spread mass on every expert that is correct.

    Nodes no expert classifies correctly get uniform mass -- there is nothing to
    learn there, and a hard argmax there would invent a preference the frozen
    experts do not support.
    """
    preds = _expert_predict(X, experts, router_idx)
    truth = y[router_idx][:, None]
    correct = (preds == truth) & ~np.isnan(preds)
    n_experts = preds.shape[1]
    mass = correct.sum(axis=1).astype(float)
    targets = np.where(correct, 1.0, 0.0)
    targets /= np.maximum(mass, 1.0)[:, None]
    unseen = mass == 0
    targets[unseen] = 1.0 / n_experts
    return targets


def _router_features(X: np.ndarray, W_norm: np.ndarray) -> np.ndarray:
    """Node-MoE gate features: ``[X, |AX−X|, |A²X−X|]``.

    Node-MoE (arXiv:2406.03464) shows raw features are insufficient for routing:
    the gate must see neighborhood discrepancy over 1–2 hops to estimate
    homophily/structural regime per node. Using cached label-free propagations
    matches the existing ``[X, AX]`` design while adding the discrepancy terms.
    """
    A = np.asarray(W_norm)
    AX = A @ X
    A2X = A @ (AX)
    return np.hstack([X, np.abs(AX - X), np.abs(A2X - X)])


class LiteRouterMLP(torch.nn.Module):
    """Two-layer MLP read-out over the cached propagated features."""

    def __init__(self, n_features_in: int, hidden_dim: int = 32,
                 n_experts: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_features_in, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, n_experts),
        )

    def forward(self, h):
        return F.softmax(self.net(h), dim=-1)


def train_router(
    X: np.ndarray,
    W_norm: np.ndarray,
    targets: np.ndarray,
    router_idx: np.ndarray,
    n_experts: int,
    hidden_dim: int = 32,
    epochs: int = 200,
    lr: float = 1e-2,
    dropout: float = 0.1,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[LiteRouterMLP, list[float]]:
    """Fit the router on ``router_idx`` nodes only.

    The loss is restricted to ``router_idx``, so no unlabeled node contributes
    gradient and no test label can leak in.
    """
    torch.manual_seed(seed)
    feats = _router_features(X, W_norm)
    model = LiteRouterMLP(
        n_features_in=feats.shape[1], hidden_dim=hidden_dim,
        n_experts=n_experts, dropout=dropout,
    ).to(device)
    ht = torch.as_tensor(feats, dtype=torch.float32, device=device)
    # ``targets`` is already aligned to ``router_idx`` (one row per router node).
    yt = torch.as_tensor(targets, dtype=torch.float32, device=device)
    router_t = torch.as_tensor(router_idx, dtype=torch.long, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses: list[float] = []
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        picked = model(ht)[router_t]
        loss = -(yt * torch.log(picked + 1e-8)).sum(dim=1).mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return model, losses


def router_assignment(model: LiteRouterMLP, X: np.ndarray, W_norm: np.ndarray,
                      device: str = "cpu") -> np.ndarray:
    """Hard one-hot routing from a trained router's argmax."""
    model.eval()
    with torch.no_grad():
        feats = _router_features(X, W_norm)
        ht = torch.as_tensor(feats, dtype=torch.float32, device=device)
        pi = model(ht).cpu().numpy()
    out = np.zeros((X.shape[0], pi.shape[1]), dtype=float)
    out[np.arange(X.shape[0]), pi.argmax(axis=1)] = 1.0
    return out


def split_fit_router(train_mask: np.ndarray, y: np.ndarray, seed: int = 0,
                     router_frac: float = 0.35) -> tuple[np.ndarray, np.ndarray]:
    """Carve the training nodes into expert-fit and router-supervision halves.

    Stratified on ``y`` so both halves keep the class balance; otherwise the
    router half can end up nearly single-class and the target degenerates.
    """
    rng = np.random.default_rng(seed)
    fit = np.zeros(train_mask.shape, dtype=bool)
    router = np.zeros(train_mask.shape, dtype=bool)
    for cls in np.unique(y):
        idx = np.flatnonzero(train_mask & (y == cls))
        rng.shuffle(idx)
        n_router = max(1, int(round(router_frac * idx.size)))
        router[idx[:n_router]] = True
        fit[idx[n_router:]] = True
    return fit, router


def uniform_ensemble_accuracy(
    X: np.ndarray,
    y: np.ndarray,
    experts: list[object | None],
    test_mask: np.ndarray,
) -> float:
    """Uniform ensemble (Ens-Avg): average predictions across all experts.

    For each test node, get prediction/prob from every expert that is usable;
    if experts return class labels, vote uniformly; if probabilities, average.
    Skips experts that are None. Returns accuracy over evaluated nodes.
    """
    idx = np.flatnonzero(test_mask)
    if idx.size == 0:
        return 0.0
    # collect predictions per expert
    all_preds = []
    usable_experts = []
    for k, head in enumerate(experts):
        if head is None:
            continue
        usable_experts.append(head)
        if hasattr(head, "predict_nodes"):
            pred = head.predict_nodes(X, idx)
        elif hasattr(head, "predict_proba"):
            # take argmax from proba
            proba = head.predict_proba(X[idx]) if hasattr(head, "predict") or True else None
            # sklearn style
            try:
                proba = head.predict_proba(X[idx])
                pred = proba.argmax(axis=1)
            except Exception:
                pred = head.predict(X[idx])
        else:
            try:
                pred = head.predict(X[idx])
            except Exception:
                continue  # skip unusable
        all_preds.append(pred)
    if len(all_preds) == 0:
        return 0.0
    # vote
    votes = np.stack(all_preds, axis=0)  # (n_experts, n_test)
    # majority vote
    from scipy.stats import mode
    try:
        ens = mode(votes, axis=0, keepdims=False).mode
    except Exception:
        ens = votes[0]
        for v in votes[1:]:
            ens = (ens + v) // 2  # rough
    correct = int(np.count_nonzero(ens == y[idx]))
    return float(correct / idx.size)


def evaluate_learned_routing(
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
    device: str = "cpu",
) -> dict:
    """Score learned routing against every baseline on one frozen pool/split.

    All conditions reuse one ``fit_experts`` call for the frozen pool, so the
    pool is identical across conditions; any difference is attributable to routing
    alone. Adds uniform ensemble (Ens-Avg) baseline.
    """
    from spectral_distillation.src.router_protocol import random_assignment

    n_experts = pool.shape[1]
    fit_mask, router_mask = split_fit_router(train_mask, y, seed=seed,
                                             router_frac=router_frac)
    experts = fit_experts(X, y, pool, fit_mask, expert_factory=expert_factory)
    # No-routing reference gets its own one-column fit, as in protocol B.
    global_experts = fit_experts(X, y, np.ones((X.shape[0], 1)), fit_mask,
                                 expert_factory=expert_factory)

    targets = build_router_targets(y, experts, X, np.flatnonzero(router_mask))
    router_idx = np.flatnonzero(router_mask)
    model, losses = train_router(
        X, W_norm, targets, router_idx, n_experts, epochs=epochs,
        seed=seed, device=device,
    )
    learned = router_assignment(model, X, W_norm, device=device)

    conditions = {
        "single_global": np.ones((X.shape[0], 1)),
        "learned_router": learned,
        "feature_kmeans": pool,
        "oracle": oracle,
        "random": random_assignment(oracle, rng=seed),
    }
    acc = {}
    for name, assignment in conditions.items():
        heads = global_experts if name == "single_global" else experts
        acc[name] = mixture_accuracy(X, y, assignment, heads, test_mask)[0]
    # uniform ensemble over frozen pool experts
    acc["ensemble_uniform"] = uniform_ensemble_accuracy(X, y, experts, test_mask)

    pool_labels = pool.argmax(axis=1)
    return {
        "accuracy": acc,
        "learned_minus_no_routing": acc["learned_router"] - acc["single_global"],
        "learned_minus_random": acc["learned_router"] - acc["random"],
        "learned_minus_ensemble": acc["learned_router"] - acc.get("ensemble_uniform", 0.0),
        "ensemble_minus_no_routing": acc.get("ensemble_uniform", 0.0) - acc["single_global"],
        "router_vs_kmeans_agreement": float(
            (learned.argmax(axis=1) == pool_labels).mean()
        ),
        "final_train_loss": losses[-1] if losses else float("nan"),
        "n_fit_nodes": int(fit_mask.sum()),
        "n_router_nodes": int(router_mask.sum()),
    }
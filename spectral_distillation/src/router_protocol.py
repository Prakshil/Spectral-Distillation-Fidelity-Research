"""The four router conditions and the D1-D4 decision rules.

Implements section 6 of the implementation guide: label-free / oracle /
random / uniform conditions sharing identical experts, training, and budget,
with only the assignment signal varying. Also computes the structural signal
fields used by the label-free router and the D1-D4 decision-rule evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spectral_distillation.src.laplacian import normalize_adjacency, build_adjacency

EXPERT_TYPES = ("low_pass", "high_pass", "identity")


def label_homophily(W: np.ndarray, y: np.ndarray) -> np.ndarray:
    """h(v) = |{u in N(v) : y_u == y_v}| / |N(v)| (label homophily, oracle only)."""
    W = np.asarray(build_adjacency(W), dtype=float)
    y = np.asarray(y)
    n = W.shape[0]
    same = np.zeros(n)
    neighbors = [np.flatnonzero(W[i] > 0) for i in range(n)]
    for i in range(n):
        nb = neighbors[i]
        if nb.size == 0:
            same[i] = np.nan
        else:
            same[i] = np.mean(y[nb] == y[i])
    return same


def feature_homophily(W: np.ndarray, X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """hx(v) = (1/|N(v)|) sum_{u in N(v)} cos(x_v, x_u)."""
    W = np.asarray(build_adjacency(W), dtype=float)
    X = np.asarray(X, dtype=float)
    n = W.shape[0]
    norm = np.linalg.norm(X, axis=1)
    X_norm = X / np.maximum(norm[:, None], eps)
    sims = X_norm @ X_norm.T
    out = np.zeros(n)
    for i in range(n):
        neighbors = np.flatnonzero(W[i] > 0)
        if neighbors.size:
            out[i] = float(np.mean(sims[i, neighbors]))
    return out


def boundary_score(W: np.ndarray, communities: np.ndarray) -> np.ndarray:
    """beta(v) = |{u in N(v) : c(u) != c(v)}| / |N(v)|."""
    W = np.asarray(build_adjacency(W), dtype=float)
    communities = np.asarray(communities)
    n = W.shape[0]
    degrees = W.sum(axis=1)
    cross = np.zeros(n)
    for i in range(n):
        nb = np.flatnonzero(W[i] > 0)
        if nb.size:
            cross[i] = np.sum(communities[nb] != communities[i])
    with np.errstate(invalid="ignore"):
        return cross / np.maximum(degrees, 1.0)


def oracle_bucket_assignment(
    W: np.ndarray,
    y: np.ndarray,
    h_low: float = 0.4,
    h_high: float = 0.6,
    soft_gates: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """Oracle assignment on real graphs from label homophily buckets.

    low-pass if h(v) >= h_high, high-pass if h(v) <= h_low, identity otherwise.
    Returns a (n, 3) soft assignment; ``soft_gates`` is the channel-mixing
    soft variant (0.8, 0.1, 0.1) applied to the winning channel only.
    """
    h = label_homophily(W, y)
    assignment = np.zeros((W.shape[0], 3))
    for i in range(W.shape[0]):
        if np.isnan(h[i]):
            assignment[i] = np.array(soft_gates) / np.sum(soft_gates) if soft_gates else np.array([1 / 3] * 3)
        elif h[i] >= h_high:
            assignment[i] = np.array(soft_gates) if soft_gates else np.array([1.0, 0.0, 0.0])
        elif h[i] <= h_low:
            assignment[i] = np.array(soft_gates)[::-1] if soft_gates else np.array([0.0, 0.0, 1.0])
        else:
            assignment[i] = soft_gates if soft_gates else np.array([0.0, 1.0, 0.0])
    return assignment


def oracle_regime_assignment(regime_ids: np.ndarray, n_regimes: int = 3) -> np.ndarray:
    """Oracle assignment on planted graphs: true regime id one-hot."""
    regime_ids = np.asarray(regime_ids, dtype=int)
    assignment = np.zeros((regime_ids.size, n_regimes))
    assignment[np.arange(regime_ids.size), np.clip(regime_ids, 0, n_regimes - 1)] = 1.0
    return assignment


def uniform_assignment(n: int, n_experts: int = 3) -> np.ndarray:
    """Uniform equal-mixing assignment (1/K, ..., 1/K), gates frozen."""
    return np.full((n, n_experts), 1.0 / n_experts)


def random_assignment(
    oracle_assignment: np.ndarray,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Usage-matched random assignment: same expert histogram, random per-node."""
    oracle_assignment = np.asarray(oracle_assignment, dtype=float)
    n, n_experts = oracle_assignment.shape
    if isinstance(rng, np.random.Generator):
        gen = rng
    else:
        gen = np.random.default_rng(rng if rng is not None else 0)
    usage = oracle_assignment.sum(axis=0)
    usage_probs = usage / np.maximum(usage.sum(), 1e-12)
    assignment = np.zeros_like(oracle_assignment)
    for i in range(n):
        assignment[i] = gen.multinomial(1, usage_probs)
    return assignment


def label_free_assignment(
    W: np.ndarray,
    X: np.ndarray,
    hidden_dim: int = 64,
    n_layers: int = 3,
    n_experts: int = 3,
    epochs: int = 200,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[np.ndarray, object]:
    """Learn the label-free structural assignment (no node labels used).

    Structural clustering over {hx, spectral_ratio, log_degree, clustering}
    is the primary label-free router: KMeans on the four fields, clusters
    reordered deterministically so the highest-feature-homophily group becomes
    the low-pass channel. A RouterGNN is additionally trained to imitate the
    clustering so the structural signal can be applied to unseen graphs.

    Returns (assignment, router) where ``assignment`` is the crisp structural
    clustering normalized to a soft distribution per row.
    """
    from sklearn.cluster import KMeans

    from spectral_distillation.src.gnn_router import train_condition
    from spectral_distillation.src.laplacian import compute_laplacian

    L = compute_laplacian(W)
    evals, V = np.linalg.eigh(L)
    feats = structural_features(W, X, evals, V)
    std = (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-8)

    km = KMeans(n_clusters=n_experts, n_init=10, random_state=seed).fit(std)
    labels = km.labels_
    hx = feats[:, 0]
    group_hx = [float(np.mean(hx[labels == k])) for k in range(n_experts)]
    order = np.argsort(group_hx)[::-1]  # highest feature homophily -> low-pass
    relabel = np.zeros_like(labels)
    for new_k, old_k in enumerate(order):
        relabel[labels == old_k] = new_k
    assignment = np.zeros((W.shape[0], n_experts))
    assignment[np.arange(W.shape[0]), relabel] = 1.0
    assignment = assignment / np.maximum(assignment.sum(axis=1, keepdims=True), 1e-12)

    W_norm = normalize_adjacency(W)
    router, _ = train_condition(
        x=feats,
        W=W,
        W_norm=W_norm,
        target_assignment=assignment,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_experts=n_experts,
        epochs=epochs,
        seed=seed,
        device=device,
    )
    return assignment, router


def structural_features(W: np.ndarray, X: np.ndarray, evals, V) -> np.ndarray:
    """The label-free {hx, spectral_ratio, log_degree, clustering} fields."""
    from spectral_distillation.src.gnn_router import degree_clustering_proxy
    from spectral_distillation.src.spectral_position import (
        node_spectral_position,
        spectral_ratio,
    )

    hx = feature_homophily(W, X)
    pos = spectral_ratio(node_spectral_position(V, evals))
    degrees = W.sum(axis=1)
    log_degree = np.log1p(degrees)
    clustering = degree_clustering_proxy(degrees)
    return np.stack([hx, pos, log_degree, clustering], axis=1)


@dataclass
class DecisionRules:
    """D1-D4 outcomes plus a mechanism verdict."""

    D1: bool | None = None  # signal exists: oracle > uniform
    D2: bool | None = None  # router finds it: label-free > random
    D3: bool | None = None  # not capacity: oracle > random @ same params
    D4: bool | None = None  # instrument works: D1 on positive control
    D1_p: float | None = None
    D2_p: float | None = None
    D3_p: float | None = None
    D4_p: float | None = None
    verdict: str = ""

    @property
    def supported(self) -> bool:
        return all(
            d is True for d in (self.D1, self.D2, self.D3, self.D4)
        )


def _resolve(comparisons: dict, alias: str, variants: tuple[str, ...]) -> tuple[dict | None, str]:
    for name in variants:
        row = comparisons.get(name)
        if row is not None:
            return {"mean_diff": float(row["mean_diff"]), "p": float(row["p_wilcoxon"])}, name
    return None, alias


def evaluate_decision_rules(
    comparisons: dict[str, dict],
    control_comparisons: dict[str, dict] | None = None,
    alpha: float = 0.05,
) -> DecisionRules:
    """Turn paired-comparison results into D1-D4 decision outcomes.

    ``comparisons`` keys follow ``{alt}_vs_{base}`` naming (e.g.,
    "oracle_vs_uniform", "label_free_vs_random", "oracle_vs_random").
    Each value: {"mean_diff": float, "p_wilcoxon": float}.
    ``control_comparisons`` optionally supplies the D4 mechanism-matched
    positive control (oracle vs uniform on PC-1c-R).
    """

    def wins(alias: str, variants: tuple[str, ...]):
        row, _found = _resolve(comparisons, alias, variants)
        if row is None:
            return None, None
        p = float(row["p"])
        return bool(float(row["mean_diff"]) > 0 and p < alpha), p

    d1, d1_p = wins("oracle_vs_uniform", ("oracle_vs_uniform", "oracle_uniform"))
    d2, d2_p = wins("label_free_vs_random", ("label_free_vs_random", "label_free_random"))
    d3, d3_p = wins("oracle_vs_random", ("oracle_vs_random", "oracle_random"))

    d4: bool | None = None
    d4_p: float | None = None
    if control_comparisons is not None:
        row, _found = _resolve(
            control_comparisons,
            "oracle_vs_uniform",
            ("oracle_vs_uniform", "oracle_uniform", "control_oracle_vs_uniform", "control_oracle_uniform"),
        )
        if row is not None:
            d4_p = float(row["p"])
            d4 = bool(float(row["mean_diff"]) > 0 and d4_p < alpha)

    rules = DecisionRules(
        D1=d1,
        D2=d2,
        D3=d3,
        D4=d4,
        D1_p=d1_p,
        D2_p=d2_p,
        D3_p=d3_p,
        D4_p=d4_p,
    )
    if rules.supported:
        rules.verdict = "Supported: signal exists, router finds it, not capacity, instrument valid."
    else:
        failures = [name for name, d in (("D1", d1), ("D2", d2), ("D3", d3), ("D4", d4)) if d is not True]
        rules.verdict = f"Not supported; failed rule(s): {', '.join(failures) or 'unknown'}."
    return rules
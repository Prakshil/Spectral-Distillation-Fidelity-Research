"""Ablation study (guide sections 4.2 and 6.7).

Two sweeps on the PC-1c-R control:

1. Oracle filter threshold: h_low / h_high bucket widths for the homophily
   oracle -- expected to show a plateau region where the signal is stable,
   and to degrade at extreme bucket edges.
2. Router feature ablation: which of the four label-free structural fields
   {hx, spectral_ratio, log_degree, clustering} contributes the most -- the
   gain of each single-feature and leave-one-out variant vs. random routing.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from spectral_distillation.src.mixture import fit_experts, mixture_accuracy, train_test_split
from spectral_distillation.src.planted_control import generate_pc_graph
from spectral_distillation.src.router_protocol import (
    feature_homophily,
    oracle_regime_assignment,
    random_assignment,
    uniform_assignment,
)
from spectral_distillation.src.utils import ensure_dir, get_logger, load_yaml_config, set_seed

FEATURE_FIELDS = ("hx", "spectral_ratio", "log_degree", "clustering")


def _assignment_from_field_subset(W: np.ndarray, X: np.ndarray, fields: tuple[str, ...], seed: int) -> np.ndarray:
    """Label-free assignment using only a subset of the structural fields."""
    from sklearn.cluster import KMeans

    from spectral_distillation.src.gnn_router import degree_clustering_proxy
    from spectral_distillation.src.laplacian import compute_laplacian
    from spectral_distillation.src.spectral_position import (
        node_spectral_position,
        spectral_ratio,
    )

    L = compute_laplacian(W)
    evals, V = np.linalg.eigh(L)
    pos = spectral_ratio(node_spectral_position(V, evals))
    degrees = W.sum(axis=1)
    all_feats = {
        "hx": feature_homophily(W, X),
        "spectral_ratio": pos,
        "log_degree": np.log1p(degrees),
        "clustering": degree_clustering_proxy(degrees),
    }
    use = [all_feats[f] for f in fields]
    feats = np.stack(use, axis=1)
    std = (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-8)
    km = KMeans(n_clusters=3, n_init=10, random_state=seed).fit(std)
    hx = all_feats["hx"]
    order = np.argsort([float(np.mean(hx[km.labels_ == k])) for k in range(3)])[::-1]
    relabel = np.zeros_like(km.labels_)
    for new_k, old_k in enumerate(order):
        relabel[km.labels_ == old_k] = new_k
    assignment = np.zeros((W.shape[0], 3))
    assignment[np.arange(W.shape[0]), relabel] = 1.0
    return assignment


def cell_accuracy(graph: dict, assignments: dict, condition: str, split: int, seed: int) -> float:
    X, y = graph["features"], graph["y"]
    train, test = train_test_split(y, split, seed)
    experts = fit_experts(X, y, assignments[condition], train)
    acc, _ = mixture_accuracy(X, y, assignments[condition], experts, test)
    return float(acc)


def _mean_gain(graph: dict, assignments: dict, alt: str, base: str, args) -> float:
    n = graph["W"].shape[0]
    with_alt = dict(assignments)
    if base == "uniform" and base not in with_alt:
        with_alt["uniform"] = uniform_assignment(n, 3)
    diffs = []
    for split in range(args.splits):
        for seed in range(args.seeds):
            a_alt = cell_accuracy(graph, with_alt, alt, split, seed)
            a_base = cell_accuracy(graph, with_alt, base, split, seed)
            diffs.append(a_alt - a_base)
    return float(np.mean(diffs))


def run_oracle_sweep(graph: dict, args, log) -> list[dict]:
    """Sweep h_low/h_high bucket width on the real-like ERP graph.

    Uses the ERP synthetic (guide narrative graph) where the homophily-bucket
    oracle is the natural oracle; features are the four label-free structural
    fields so the mixture is trained label-free and only the bucket thresholds
    vary.
    """
    from spectral_distillation.src.router_protocol import label_homophily

    W = np.asarray(graph["W"], dtype=float)
    y = np.asarray(graph["y"], dtype=int)
    n = W.shape[0]

    rows = []
    assignments = {}  # shared across the sweep
    uniform = uniform_assignment(n, 3)
    for delta in args.delta_sweep:
        h_low, h_high = 0.5 - delta / 2, 0.5 + delta / 2
        if h_low < 0 or h_high > 1:
            continue
        h = label_homophily(W, y)
        assign = np.zeros((n, 3))
        for i in range(n):
            if np.isnan(h[i]):
                assign[i, 0] = 1.0
            elif h[i] >= h_high:
                assign[i, 0] = 1.0
            elif h[i] <= h_low:
                assign[i, 2] = 1.0
            else:
                assign[i, 1] = 1.0
        assignments["oracle_bucket"] = assign
        gain = _mean_gain(graph, {"oracle_bucket": assign, "uniform": uniform}, "oracle_bucket", "uniform", args)
        rows.append(
            {
                "sweep": "oracle_delta",
                "delta": delta,
                "h_low": round(h_low, 3),
                "h_high": round(h_high, 3),
                "oracle_bucket_gain_vs_uniform": gain,
            }
        )
        log.info("delta=%.2f h_low=%.2f h_high=%.2f gain=%.4f", delta, h_low, h_high, gain)
    return rows


def _homophily_range_graph(n: int, seed: int = 0) -> dict:
    """Real-like SBM whose nodes span the homophily range h(v) in [0, 0.7].

    Three balanced blocks with prescribed homophily: block 0 homophilic
    (h~0.7, low-pass), block 1 mixed (h~0.5, identity), block 2 heterophilic
    (h~0.2, high-pass). Node features use class prototypes rotated per block
    (the planted-control mechanism) so correct routing genuinely helps.
    """
    from spectral_distillation.src.laplacian import build_adjacency

    rng = np.random.default_rng(seed)
    n0 = n // 3
    n1 = n // 3
    n2 = n - 2 * (n // 3)
    labels = np.concatenate([np.full(n0, 0), np.full(n1, 1), np.full(n2, 2)])
    sizes = [n0, n1, n2]
    # P[block_i, block_j] edge probability between blocks
    P = np.array(
        [
            [0.55, 0.08, 0.05],
            [0.08, 0.30, 0.12],
            [0.05, 0.12, 0.06],
        ]
    )
    W = np.zeros((n, n), dtype=float)
    offsets = np.array([0, n0, n0 + n1])
    lo, hi = 0.4, 0.9
    weak = (0.05, 0.3)
    for bi in range(3):
        for bj in range(bi, 3):
            p = P[bi, bj]
            a0, b0 = offsets[bi], offsets[bi] + sizes[bi]
            a1, b1 = offsets[bj], offsets[bj] + sizes[bj]
            for i in range(a0, b0):
                for j in range(max(i + 1, a1), b1):
                    if rng.random() < p:
                        w = rng.uniform(*((lo, hi) if p > 0.2 else weak))
                        W[i, j] = W[j, i] = w
    build_adjacency(W)
    # safety backbone (additive, preserves block homophily)
    perm = rng.permutation(n)
    for step in range(n - 1):
        i, j = perm[step], perm[step + 1]
        if W[i, j] == 0:
            W[i, j] = W[j, i] = 0.05
    d = 16
    bases = []
    for _ in range(3):
        A = rng.standard_normal((d, d))
        Q, _ = np.linalg.qr(A)
        bases.append(Q)
    n_protos = max(3, 5) if n >= 5 else 3
    protos = np.zeros((n_protos, d))
    for cid in range(n_protos):
        protos[cid, cid] = 1.0
    feats = np.zeros((n, d))
    for i in range(n):
        c = labels[i]
        feats[i] = protos[c] @ bases[c] + 0.4 * rng.standard_normal(d)
    return {"W": W, "y": labels, "features": feats}


def run_feature_sweep(graph: dict, args, log) -> list[dict]:
    W, X = graph["W"], graph["features"]
    base = {
        "oracle": oracle_regime_assignment(graph["regimes"], 3),
        "random": random_assignment(oracle_regime_assignment(graph["regimes"], 3), rng=args.random_seed),
    }
    rows = []
    single_fields = list(FEATURE_FIELDS)
    leave_one_out = [tuple(f for f in FEATURE_FIELDS if f != field) for field in FEATURE_FIELDS]
    variants = [(("all", "all"), FEATURE_FIELDS)] + [(("single", f), (f,)) for f in single_fields] + [
        (("loo", f), subset) for f, subset in zip(FEATURE_FIELDS, leave_one_out)
    ]

    assignments = dict(base)
    gains: dict[str, float] = {}
    for (kind, name), fields in variants:
        a = _assignment_from_field_subset(W, X, fields, seed=args.router_seed)
        label = f"{kind}:{name}"
        assignments[label] = a
        gains[label] = _mean_gain(graph, assignments, label, "random", args)
        log.info("feature variant %s gain=%.4f", label, gains[label])

    for label, gain in gains.items():
        kind, _, name = label.partition(":")
        rows.append(
            {
                "sweep": "feature_variant",
                "variant": label,
                "kind": kind,
                "field": name,
                "label_free_gain_vs_random": gain,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--n-patches", type=int, default=None)
    parser.add_argument("--splits", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--delta-sweep", type=float, nargs="+", default=None)
    parser.add_argument("--feature-sweep", action="store_true", default=False)
    parser.add_argument("--router-seed", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--out", default="logs")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    log = get_logger("ablation", config.get("logging", {}).get("level", "INFO"))
    pc = config.get("planted_control", {})
    filt = config.get("sparsification", {}).get("oracle_filter", {})

    args.n = args.n or pc.get("n_nodes", 10000)
    args.n_patches = args.n_patches or pc.get("n_patches", 50)
    args.splits = args.splits or 4
    args.seeds = args.seeds or 2
    if args.delta_sweep is None:
        args.delta_sweep = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
    set_seed(args.router_seed)

    t0 = time.perf_counter()
    if args.feature_sweep:
        graph = generate_pc_graph(
            n_nodes=args.n,
            n_patches=args.n_patches,
            regime_ratio=tuple(pc.get("regime_ratio", [0.4, 0.4, 0.2])),
            seed=args.router_seed,
        )
        log.info("built PC-1c-R graph n=%d (feature sweep)", args.n)
    else:
        graph = _homophily_range_graph(args.n, seed=args.router_seed)
        log.info("built real-like homophily-range graph n=%d (oracle sweep)", args.n)
    log.info("graph ready in %.1fs", time.perf_counter() - t0)

    rows: list[dict] = []
    if args.feature_sweep:
        rows += run_feature_sweep(graph, args, log)
    else:
        f = filt.get("delta", 0.5)
        if f not in args.delta_sweep:
            args.delta_sweep.insert(0, float(f))
        rows += run_oracle_sweep(graph, args, log)

    out_dir = ensure_dir(Path(args.out) / "ablation")
    csv_path = out_dir / "ablation.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    json_path = out_dir / "ablation.json"
    json_path.write_text(json.dumps({"rows": rows, "mode": "feature" if args.feature_sweep else "oracle_delta"}, indent=2), encoding="utf-8")
    log.info("wrote %s and %s", csv_path, json_path)

    print("\n=== Ablation ===")
    if rows:
        header = " | ".join(rows[0].keys())
        print(header)
        print("-" * len(header))
        for row in rows:
            print(" | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row.values()))


if __name__ == "__main__":
    main()
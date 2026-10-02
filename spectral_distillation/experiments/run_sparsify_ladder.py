"""Retention-ladder D1-D3 comparison on the real DGL Amazon fraud graph.

Tests the *distillation-sharpens-routing* hypothesis (the potential
"inverted-U"): on the dense, saturated Amazon co-review graph (8,639 nodes,
3.3M edges, edge label homophily 0.954) the label-free structural router
collapses to random (D2 fails, `run_real_fraud.py`). If spectral
(effective-resistance) sparsification removes *noise edges* first, the
surviving graph should carry the routing signal *sharper* -- so the label-free
router's advantage over random should *rise* at moderate retention before the
resolution makes it unusable at very low budgets. Budget-matched random /
degree prunings are the controls that should *not* display the same
sharpening.

Per ladder point (method, retention):
  - sparsify W (ER-sampling / uniform-random / top-degree) to an edge budget;
  - if retention < 1, report the spectral distortion (SD) vs the original;
  - rebuild all four condition assignments on the sparsified graph;
  - run the D1-D3 protocol cells and record the comparison statistics.

D4 is a graph-independent instrument check: its PC-1c-R positive control is
estimated once and reused for every point.

Usage:
    python experiments/run_sparsify_ladder.py --budgets 0.8,0.6,0.45,0.3,0.2,0.12,0.07
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from spectral_distillation.src.baselines import degree_sparsify, random_sparsify
from spectral_distillation.src.distortion import spectral_distortion_profile
from spectral_distillation.src.effective_resistance import effective_resistance_exact
from spectral_distillation.src.evaluation import (
    combine_comparisons,
    run_protocol,
)
from spectral_distillation.src.laplacian import compute_laplacian
from scipy.linalg import eigvalsh
from spectral_distillation.src.mixture import fit_experts, mixture_accuracy, train_test_split
from spectral_distillation.src.planted_control import generate_pc_graph
from spectral_distillation.src.real_fraud import load_real_fraud
from spectral_distillation.src.router_protocol import (
    evaluate_decision_rules,
    label_free_assignment,
    oracle_bucket_assignment,
    oracle_regime_assignment,
    random_assignment,
    uniform_assignment,
)
from spectral_distillation.src.sparsifier import spectral_sparsify
from spectral_distillation.src.utils import (
    device_config,
    ensure_dir,
    get_logger,
    load_yaml_config,
    set_seed,
)

METHODS = {"er": spectral_sparsify, "random": random_sparsify, "degree": degree_sparsify}

_PC_N = 2000


def _comparison_dict(c) -> dict:
    return {
        "mean_diff": c.mean_diff,
        "ci_low": c.ci_low,
        "ci_high": c.ci_high,
        "p_wilcoxon": c.p_wilcoxon,
        "p_paired_t": c.p_paired_t,
        "effect_size_dz": c.effect_size_dz,
        "significant": c.significant,
    }


def _homophily_mean(W: np.ndarray, y: np.ndarray) -> float:
    """Mean label homophily of a (non-destructive) graph."""
    from spectral_distillation.src.router_protocol import label_homophily

    h = label_homophily(W, y)
    finite = h[~np.isnan(h)]
    return float(finite.mean()) if finite.size else float("nan")


def _point_conditions(graph: dict, W: np.ndarray, args) -> dict[str, np.ndarray]:
    X, y = graph["features"], graph["y"]
    oracle = oracle_bucket_assignment(
        W, y, adaptive=args.adaptive_oracle,
        budget_normalized=args.budget_oracle, top_k=args.top_k,
    )
    uniform = uniform_assignment(W.shape[0], 3)
    random = random_assignment(oracle, rng=args.random_seed)
    label_free, _ = label_free_assignment(
        W, X,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        epochs=args.router_epochs,
        seed=args.router_seed,
        device=args.device,
        strategy=args.label_free_strategy,
        order_feature=args.order_feature,
        spectral_k=args.spectral_k,
    )
    return {"oracle": oracle, "label_free": label_free, "random": random, "uniform": uniform}


def _cell_accuracy(graph: dict, assignments: dict, condition: str, split: int, seed: int) -> float:
    X, y = graph["features"], graph["y"]
    train, test = train_test_split(y, split, seed)
    experts = fit_experts(X, y, assignments[condition], train)
    acc, _ = mixture_accuracy(X, y, assignments[condition], experts, test)
    return float(acc)


def _run_point(graph: dict, W: np.ndarray, method: str, retention: float, args, log) -> dict:
    n_orig = graph["n_edges"]
    budget = int(round(retention * n_orig))
    t0 = time.perf_counter()

    if retention >= 1.0 or budget >= n_orig:
        W_pt = graph["W"]
    elif method == "er":
        W_pt = spectral_sparsify(
            graph["W"], budget, oracle_score=None,
            sample=True, seed=args.random_seed,
            resistance_matrix=args._er_resistance,
        )
    else:
        sparsify = METHODS[method]
        W_pt = sparsify(graph["W"], budget, seed=args.random_seed)

    n_kept = int(np.count_nonzero(W_pt) // 2)
    log.info("[%s retention=%.2f] sparsified to %d/%d edges in %.1fs",
             method, retention, n_kept, n_orig, time.perf_counter() - t0)

    assignments = _point_conditions(graph, W_pt, args)
    for name, a in assignments.items():
        log.info("  %-10s usage=%s", name, np.bincount(np.argmax(a, axis=1), minlength=3).tolist())

    results = run_protocol(
        run_fn=lambda cond, split, seed: _cell_accuracy(graph, assignments, cond, split, seed),
        n_splits=args.splits,
        n_seeds=args.seeds,
        progress=not args.no_progress,
        progress_desc=f"cells [{method} r={retention:.2f}]",
    )
    comparisons = {name: _comparison_dict(c) for name, c in combine_comparisons(results).items()}

    sd = 0.0
    sd_profile = {}
    if retention < 1.0 and not args.skip_sd:
        sd_profile = spectral_distortion_profile(
            None, compute_laplacian(W_pt), ref_eigenvalues=args._ref_eigenvalues
        )
        sd = sd_profile["SD"]
        log.info("  SD vs original = %.4f (median %.2e, p99 %.2e, argmax mode %d/%d)",
                 sd, sd_profile["median_relative_error"],
                 sd_profile["p99_relative_error"], sd_profile["argmax_index"],
                 args._ref_eigenvalues.size - 1)

    decisions = evaluate_decision_rules(
        comparisons=comparisons, control_comparisons=args._pc_control
    )

    return {
        "method": method,
        "retention": float(retention),
        "n_edges_kept": n_kept,
        "n_edges_orig": n_orig,
        "fraction_kept": n_kept / n_orig,
        "sd_laplacian": float(sd),
        "sd_profile": sd_profile,
        "edge_label_homophily": _homophily_mean(W_pt, graph["y"]),
        "assignments_usage": {
            name: np.bincount(np.argmax(a, axis=1), minlength=3).tolist()
            for name, a in assignments.items()
        },
        "comparisons": comparisons,
        "findings": decisions.__dict__,
    }


def _positive_control(args, log) -> dict:
    t0 = time.perf_counter()
    graph = generate_pc_graph(n_nodes=_PC_N, n_patches=30, regime_ratio=(0.4, 0.4, 0.2), seed=args.router_seed)
    oracle = oracle_regime_assignment(graph["regimes"], 3)
    uniform = uniform_assignment(_PC_N, 3)
    random = random_assignment(oracle, rng=args.random_seed)
    results = run_protocol(
        run_fn=lambda cond, split, seed: _cell_accuracy(
            graph,
            {"oracle": oracle, "uniform": uniform, "random": random, "label_free": uniform},
            cond, split, seed,
        ),
        n_splits=min(args.splits, 6),
        n_seeds=min(args.seeds, 2),
        progress=not args.no_progress,
        progress_desc="D4 control cells",
    )
    combined = combine_comparisons(results)
    log.info("positive control (D4) ready in %.1fs", time.perf_counter() - t0)
    return {
        name: {"mean_diff": c.mean_diff, "p_wilcoxon": c.p_wilcoxon}
        for name, c in combined.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", default="amazon",
                        choices=["amazon", "tolokers", "yelpchi"],
                        help="real fraud graph to run on")
    parser.add_argument("--amazon-mat", default=None,
                        help="explicit path override for the amazon .mat")
    parser.add_argument("--data-path", default=None,
                        help="explicit path override for a non-amazon dataset dir")
    parser.add_argument("--budgets", default="0.8,0.6,0.45,0.3,0.2,0.12,0.07",
                        help="comma-separated retention fractions of original edges")
    parser.add_argument("--methods", default="er,random,degree",
                        help="comma-separated subset of er,random,degree")
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--splits", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--router-epochs", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--router-seed", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--adaptive-oracle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--budget-oracle", action="store_true")
    parser.add_argument("--label-free-strategy", default="kmeans",
                        choices=["kmeans", "proxy_quantile"],
                        help="label-free router bucketing strategy (see docs/d2_router_fix.md)")
    parser.add_argument("--order-feature", default="hx",
                        choices=["hx", "eig_nb_sim"],
                        help="per-node scalar the router orders nodes by")
    parser.add_argument("--spectral-k", type=int, default=8,
                        help="number of eigenmodes used by spectral scores")
    parser.add_argument("--out-suffix", default="",
                        help="suffix for the output directory, e.g. _eig_nb_sim_pq")
    parser.add_argument("--max-nodes", type=int, default=None,
                        help="yelpchi only: cap the sampled node count "
                             "(default loader cap 15000; use 0 for the full graph)")
    parser.add_argument("--skip-sd", action="store_true", help="skip spectral-distortion eigendecompositions")
    parser.add_argument("--resume", action="store_true",
                        help="keep (method, retention) points already present in ladder_results.jsonl "
                             "and only compute the missing ones")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--out", default="logs")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    log = get_logger("sparsify_ladder", config.get("logging", {}).get("level", "INFO"))
    proto = config.get("protocol", {})

    args.splits = args.splits or proto.get("n_splits", 10)
    args.seeds = args.seeds or proto.get("n_seeds", 3)
    args.router_epochs = args.router_epochs or 200
    args.hidden_dim = args.hidden_dim or 64
    args.n_layers = args.n_layers or 3
    args.device = device_config(args.device)

    budgets = [float(b) for b in args.budgets.split(",")]
    methods = [m.strip() for m in args.methods.split(",")]
    for m in methods:
        if m not in METHODS:
            raise SystemExit(f"unknown method: {m}")
    any_full = any(b >= 1.0 for b in budgets)
    if any_full and len(budgets) > 1:
        raise SystemExit("include full-retention budget only alone (it is invariant to method)")

    set_seed(args.router_seed)

    t0 = time.perf_counter()
    graph = load_real_fraud(args.dataset, args.data_path or args.amazon_mat,
                             max_nodes=args.max_nodes or None)
    log.info("loaded %s fraud graph n=%d d=%d edges=%d pos=%.4f in %.1fs",
             args.dataset,
             graph["n_nodes"], graph["d_features"], graph["n_edges"],
             graph["positive_ratio"], time.perf_counter() - t0)

    if "er" in methods:
        t0 = time.perf_counter()
        args._er_resistance = effective_resistance_exact(compute_laplacian(graph["W"]))
        log.info("exact effective resistance (n=%d sparse-ok? pinvh dense) ready in %.1fs",
                 graph["n_nodes"], time.perf_counter() - t0)
    else:
        args._er_resistance = None

    # The reference Laplacian is identical at every retention point, so its
    # spectrum is computed once up front and the O(n^2) buffer is released.
    # This keeps a 1.9 GB dense matrix from sitting next to the ER matrix for
    # the whole run and halves the per-point SD cost.
    args._ref_laplacian = None
    args._ref_eigenvalues = None
    if not args.skip_sd:
        t0 = time.perf_counter()
        args._ref_eigenvalues = np.sort(eigvalsh(compute_laplacian(graph["W"])))
        log.info("cached reference spectrum (%d eigenvalues) in %.1fs; SD will now "
                 "cost one eigvalsh per point instead of two",
                 args._ref_eigenvalues.size, time.perf_counter() - t0)

    args._pc_control = _positive_control(args, log)

    out_dir = ensure_dir(Path(args.out) / f"{args.dataset}_fraud_ladder{args.out_suffix}")
    out_path = out_dir / "ladder_results.jsonl"
    summary = {
        # Private ``_``-prefixed args are runtime caches, not configuration:
        # _er_resistance is a dense O(n^2) matrix and json.dumps(default=str)
        # would serialize it as a multi-megabyte ellipsis-truncated blob.
        "config": {k: v for k, v in args.__dict__.items() if not k.startswith("_")},
        "graph": {
            "n_nodes": graph["W"].shape[0],
            "d_features": graph["d_features"],
            "n_edges": graph["n_edges"],
            "positive_ratio": graph["positive_ratio"],
            "source": graph["source"],
        },
        "control": {"pc_n_nodes": _PC_N, **args._pc_control},
        "points": [],
    }

    # Resume support: a full 15-point ladder on a large graph can exceed a
    # single command timeout, and each point costs ~6 min. Reuse already-written
    # (method, retention) rows instead of recomputing them.
    write_lines = []
    done: set[tuple[str, float]] = set()
    if args.resume and out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            write_lines.append(line)
            done.add((str(row["method"]), float(row["retention"])))
            summary["points"].append(row)
        log.info(
            "resuming: %d/%d points already present in %s",
            len(done), len(budgets) * len(methods), out_path,
        )

    for b in budgets:
        run_methods = ["-"] if b >= 1.0 else methods
        for m in run_methods:
            if (m, float(b)) in done:
                print(f"{m:<7} r={b:<5.2f} skipped (already in artifact)")
                continue
            record = _run_point(graph, graph["W"], m, b, args, log)
            summary["points"].append(record)
            line = json.dumps(record, default=str)
            write_lines.append(line)
            # Persist after every point so an interrupted/timeout run keeps the
            # expensive resistance + protocol cells already completed.
            out_path.write_text("\n".join(write_lines) + "\n", encoding="utf-8")
            (out_dir / "ladder_summary.json").write_text(
                json.dumps(summary, indent=2, default=str), encoding="utf-8"
            )
            print(
                f"{m:<7} r={b:<5.2f} kept={record['n_edges_kept']:<9d} "
                f"D2={record['comparisons'].get('label_free_vs_random', {}).get('mean_diff', float('nan')):>+.4f}"
                f"  D1={record['comparisons'].get('oracle_vs_uniform', {}).get('mean_diff', float('nan')):>+.4f}"
                f"  SD={record['sd_laplacian']:.4f}"
            )

    log.info("wrote %s and ladder_summary.json", out_path)


if __name__ == "__main__":
    main()
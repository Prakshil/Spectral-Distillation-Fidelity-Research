"""Real fraud-graph D1-D4 protocol on public fraud graphs (multi-dataset).

Runs the same D1-D4 decision rules and frozen-gate interventions as the
synthetic PC-1c-R / ERP-fraud pipeline, but on a *real publicly-available
fraud graph*: the DGL Amazon review-correlation network (11,944 users, of whom
the 8,639 labeled ones -- 7,818 benign / 821 fraudulent -- form the graph used
here). The oracle reads the real benign/fraud labels via homophily buckets
(``oracle_bucket_assignment``); the label-free router must recover the same
routing from review-statistic features and graph structure alone, exactly as
in the synthetic control. A small PC-1c-R positive control re-estimates the
D4 instrument check within the same run.

Usage:
    python experiments/run_real_fraud.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from spectral_distillation.src.evaluation import (
    combine_comparisons,
    frozen_intervention,
    run_protocol,
)
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
from spectral_distillation.src.utils import (
    device_config,
    ensure_dir,
    get_logger,
    load_yaml_config,
    set_seed,
)

CONDITIONS = ("oracle", "label_free", "random", "uniform")

_PC_N = 2000


def build_conditions(graph: dict, args) -> dict[str, np.ndarray]:
    W, X, y = graph["W"], graph["features"], graph["y"]
    oracle = oracle_bucket_assignment(W, y, adaptive=args.adaptive_oracle,
                                      budget_normalized=args.budget_oracle, top_k=args.top_k)
    uniform = uniform_assignment(W.shape[0], 3)
    random = random_assignment(oracle, rng=args.random_seed)
    label_free, _ = label_free_assignment(
        W,
        X,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        epochs=args.router_epochs,
        seed=args.router_seed,
        device=args.device,
    )
    return {
        "oracle": oracle,
        "label_free": label_free,
        "random": random,
        "uniform": uniform,
    }


def cell_accuracy(graph: dict, assignments: dict, condition: str, split: int, seed: int) -> float:
    X, y = graph["features"], graph["y"]
    train, test = train_test_split(y, split, seed)
    experts = fit_experts(X, y, assignments[condition], train)
    acc, _ = mixture_accuracy(X, y, assignments[condition], experts, test)
    return float(acc)


def run_protocol_on_graph(graph: dict, args, log) -> dict:
    log.info("computing condition assignments...")
    t0 = time.perf_counter()
    assignments = build_conditions(graph, args)
    log.info("assignments ready in %.1fs", time.perf_counter() - t0)
    for name, a in assignments.items():
        log.info("  %-10s usage=%s", name, np.bincount(np.argmax(a, axis=1), minlength=3).tolist())

    results = run_protocol(
        run_fn=lambda cond, split, seed: cell_accuracy(graph, assignments, cond, split, seed),
        n_splits=args.splits,
        n_seeds=args.seeds,
        progress=not args.no_progress,
        progress_desc="main protocol cells",
    )
    comparisons = combine_comparisons(results)
    pairs = {
        name: {
            "mean_diff": c.mean_diff,
            "ci_low": c.ci_low,
            "ci_high": c.ci_high,
            "p_wilcoxon": c.p_wilcoxon,
            "p_paired_t": c.p_paired_t,
            "effect_size_dz": c.effect_size_dz,
            "significant": c.significant,
        }
        for name, c in comparisons.items()
    }

    control = _positive_control_pairs(args, log)
    decisions = evaluate_decision_rules(comparisons=pairs, control_comparisons=control)

    frozen = frozen_intervention(
        learned=assignments["label_free"],
        n_nodes=graph["W"].shape[0],
        oracle=assignments["oracle"],
        run_expert_fn=lambda assignment: _expert_accuracy_from_assignment(graph, assignment),
        n_permutations=args.frozen_perms,
        seed=args.router_seed,
    )

    record = {
        "config": args.__dict__,
        "graph": {
            "n_nodes": graph["W"].shape[0],
            "d_features": graph["d_features"],
            "n_edges": graph["n_edges"],
            "positive_ratio": graph["positive_ratio"],
            "n_labels": int(np.unique(graph["y"]).size),
            "source": graph["source"],
        },
        "assignments_usage": {
            name: np.bincount(np.argmax(a, axis=1), minlength=3).tolist()
            for name, a in assignments.items()
        },
        "findings": decisions.__dict__,
        "comparisons": pairs,
        "frozen": {k: v for k, v in frozen.items() if not isinstance(v, list)},
        "control": {"pc_n_nodes": _PC_N, **control},
    }
    return record


def _expert_accuracy_from_assignment(graph: dict, assignment: np.ndarray) -> float:
    X, y = graph["features"], graph["y"]
    rng = np.random.default_rng(0)
    n = y.size
    perm = rng.permutation(n)
    train = np.zeros(n, dtype=bool)
    train[perm[: n // 2]] = True
    experts = fit_experts(X, y, assignment, train)
    acc, _ = mixture_accuracy(X, y, assignment, experts, ~train)
    return float(acc)


def _positive_control_pairs(args, log) -> dict:
    """D4 instrument check on a small PC-1c-R graph (oracle > uniform)."""
    t0 = time.perf_counter()
    graph = generate_pc_graph(n_nodes=_PC_N, n_patches=30, regime_ratio=(0.4, 0.4, 0.2), seed=args.router_seed)
    oracle = oracle_regime_assignment(graph["regimes"], 3)
    uniform = uniform_assignment(_PC_N, 3)
    random = random_assignment(oracle, rng=args.random_seed)
    results = run_protocol(
        run_fn=lambda cond, split, seed: cell_accuracy(
            graph,
            {"oracle": oracle, "uniform": uniform, "random": random, "label_free": uniform},
            cond,
            split,
            seed,
        ),
        n_splits=min(args.splits, 6),
        n_seeds=min(args.seeds, 2),
        progress=not args.no_progress,
        progress_desc="D4 control cells",
    )
    combined = combine_comparisons(results)
    log.info("positive control (D4) ready in %.1fs", time.perf_counter() - t0)
    return {
        name: {
            "mean_diff": c.mean_diff,
            "p_wilcoxon": c.p_wilcoxon,
        }
        for name, c in combined.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", default="amazon",
                        choices=["amazon", "tolokers"],
                        help="real fraud graph to run on")
    parser.add_argument("--amazon-mat", default=None,
                        help="explicit path override for the amazon .mat")
    parser.add_argument("--data-path", default=None,
                        help="explicit path override for a non-amazon dataset dir")
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--splits", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--router-epochs", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--frozen-perms", type=int, default=20)
    parser.add_argument("--router-seed", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--adaptive-oracle", action=argparse.BooleanOptionalAction, default=True,
                        help="quantile-based homophily thresholds (default on: fixed 0.4/0.6 collapses the oracle on the skewed fraud labels)")
    parser.add_argument("--budget-oracle", action="store_true", help="normalize oracle homophily by same-label budget")
    parser.add_argument("--no-progress", action="store_true", help="disable tqdm progress bars")
    parser.add_argument("--out", default="logs")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    log = get_logger("real_fraud", config.get("logging", {}).get("level", "INFO"))
    proto = config.get("protocol", {})

    args.splits = args.splits or proto.get("n_splits", 10)
    args.seeds = args.seeds or proto.get("n_seeds", 3)
    args.router_epochs = args.router_epochs or 200
    args.hidden_dim = args.hidden_dim or 64
    args.n_layers = args.n_layers or 3
    args.device = device_config(args.device)

    set_seed(args.router_seed)

    t0 = time.perf_counter()
    graph = load_real_fraud(args.dataset, args.data_path or args.amazon_mat)
    log.info(
        "loaded %s fraud graph n=%d d=%d edges=%d pos=%.4f in %.1fs",
             args.dataset,
        graph["n_nodes"], graph["d_features"], graph["n_edges"],
        graph["positive_ratio"], time.perf_counter() - t0,
    )

    record = run_protocol_on_graph(graph, args, log)

    out_dir = ensure_dir(Path(args.out) / f"{args.dataset}_fraud_protocol")
    out_path = out_dir / "protocol_results.json"
    out_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    log.info("wrote %s", out_path)

    print(f"\n=== Real fraud-graph ({args.dataset}) router fidelity ===")
    print(f"{'comparison':<22}{'mean_diff':>10}{'p_wilcoxon':>12}{'d_z':>8}{'sig':>6}")
    for name, c in record["comparisons"].items():
        print(
            f"{name:<22}{c['mean_diff']:>10.4f}{c['p_wilcoxon']:>12.4g}{c['effect_size_dz']:>8.3f}"
            f"{'yes' if c['significant'] else 'no':>6}"
        )
    print("\nDecision rules:", record["findings"]["verdict"])
    print("Frozen gate (%s perms):" % args.frozen_perms)
    for k, v in record["frozen"].items():
        print(f"  {k:<18}{v:>10.4f}")
    print("\nPositive control (D4, PC n=%d):" % _PC_N)
    for name, c in record["control"].items():
        if name == "pc_n_nodes":
            continue
        print(f"  {name:<32}{c['mean_diff']:>10.4f}{c['p_wilcoxon']:>12.4g}")


if __name__ == "__main__":
    main()
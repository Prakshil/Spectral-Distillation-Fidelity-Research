"""Frozen-gate architecture and per-expert-capacity ablations (Phase 6 WS3).

The D4 instrument check in ``run_protocol.py`` freezes the routing gate and
shows ``oracle > node-shuffled > uniform/global-mean`` at a *single* expert
architecture (linear) and capacity. This runner broadens that coverage:

* expert architecture: linear, an MLP at several hidden widths, and a
  heterogeneous ``dendrite`` mixture with a different width per expert;
* gate nulls: the global node permutation plus a **within-regime** permutation
  (same regime sizes, node identity scrambled), which is the stricter null on
  graphs with known regimes.

For each (graph, architecture, capacity) it reports the frozen-gate ordering and
whether it holds, so a routing win cannot be attributed to one favorable width.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from spectral_distillation.src.evaluation import frozen_intervention
from spectral_distillation.src.mixture import (
    dendrite_expert_factories,
    expert_head,
    fit_experts,
    mixture_accuracy,
    mlp_head_width,
)
from spectral_distillation.src.planted_control import generate_pc_graph
from spectral_distillation.src.router_protocol import (
    label_free_assignment,
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

GATES = [
    "learned",
    "oracle_onehot",
    "equal",
    "global_mean",
    "node_shuffled",
    "within_regime_shuffled",
]


def architecture_configs(capacities: list[int], n_experts: int = 3) -> list[tuple[str, dict]]:
    """(name, factory kwargs) for the linear, per-width MLP, and dendrite experts.

    ``dendrite`` gets one factory per expert (widths cycled from ``capacities``
    when fewer capacities than experts are supplied).
    """
    configs: list[tuple[str, dict]] = [("logistic", {"expert_factory": expert_head})]
    for width in capacities:
        configs.append((f"mlp{width}", {"expert_factory": mlp_head_width(width, seed=0)}))
    widths = [int(capacities[i % len(capacities)]) for i in range(n_experts)]
    configs.append(("dendrite", {"expert_factories": dendrite_expert_factories(widths, seed=0)}))
    return configs


def make_run_expert(graph: dict, seed: int, expert_factory=None, expert_factories=None):
    """Build ``run_expert_fn(assignment) -> accuracy`` for one expert config.

    The train/test split is a deterministic random half-split keyed on ``seed``
    (identical across gate settings), so the only thing that varies between the
    learned gate and its nulls is the assignment itself.
    """

    X, y = graph["features"], graph["y"]
    n = y.size
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    train = np.zeros(n, dtype=bool)
    train[perm[: n // 2]] = True

    def run_expert_fn(assignment: np.ndarray) -> float:
        experts = fit_experts(
            X,
            y,
            assignment,
            train,
            expert_factory=expert_factory or expert_head,
            expert_factories=expert_factories,
        )
        acc, _ = mixture_accuracy(X, y, assignment, experts, ~train)
        return float(acc)

    return run_expert_fn


def gate_ordering(result: dict) -> dict:
    """Compare each gate against the learned assignment; does the ordering hold?

    ``holds`` requires the learned router to beat the three null gates (global
    node shuffle, global mean, uniform). The oracle comparison is reported as
    ``oracle_minus_learned`` but *not* folded into ``holds``: on a single
    half-split the label-free router can edge past the oracle by noise, so a
    small negative value here is expected and not a failure. The within-regime
    null is likewise reported as a gap: ``within_regime_gap ~ 0`` means the
    router is essentially a regime detector, while a clearly positive gap means
    it also carries within-regime signal.
    """
    learned = float(result["learned"])
    checks = {
        "learned_gt_node_shuffled": learned > float(result.get("node_shuffled_mean", -np.inf)),
        "learned_gt_global_mean": learned > float(result.get("global_mean", -np.inf)),
        "learned_gt_equal": learned > float(result.get("equal", -np.inf)),
    }
    checks = {k: bool(v) for k, v in checks.items()}
    checks["holds"] = all(checks.values())
    checks["oracle_ge_learned"] = bool(
        float(result.get("oracle_onehot", -np.inf)) >= learned
    )
    checks["oracle_minus_learned"] = float(result.get("oracle_onehot", learned)) - learned
    checks["within_regime_gap"] = learned - float(
        result.get("within_regime_shuffled_mean", learned)
    )
    return checks


def build_assignments(graph: dict, args) -> dict[str, np.ndarray]:
    W, X, regimes = graph["W"], graph["features"], graph["regimes"]
    oracle = oracle_regime_assignment(regimes, 3)
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
    return {"oracle": oracle, "label_free": label_free, "random": random, "uniform": uniform}


def run_all(graph: dict, assignments: dict, args, log) -> dict:
    capacities = list(args.capacities)
    results: dict[str, dict] = {}
    for name, kwargs in architecture_configs(capacities):
        run_expert_fn = make_run_expert(graph, args.split_seed, **kwargs)
        result = frozen_intervention(
            learned=assignments["label_free"],
            n_nodes=graph["W"].shape[0],
            oracle=assignments["oracle"],
            run_expert_fn=run_expert_fn,
            gate_settings=GATES,
            n_permutations=args.frozen_perms,
            seed=args.router_seed,
            strata=graph["regimes"],
        )
        result["ordering"] = gate_ordering(result)
        results[name] = result
        log.info(
            "%-10s learned=%.4f oracle=%.4f node_shuf=%.4f within_regime=%.4f holds=%s",
            name,
            result["learned"],
            result.get("oracle_onehot", float("nan")),
            result.get("node_shuffled_mean", float("nan")),
            result.get("within_regime_shuffled_mean", float("nan")),
            result["ordering"]["holds"],
        )
    return results


def _summary_table(results: dict, capacities: list[int]) -> dict:
    capacity_curve = {}
    for width in capacities:
        key = f"mlp{width}"
        if key in results:
            capacity_curve[str(width)] = {
                "learned": results[key]["learned"],
                "oracle_onehot": results[key].get("oracle_onehot"),
                "node_shuffled_mean": results[key].get("node_shuffled_mean"),
                "within_regime_shuffled_mean": results[key].get("within_regime_shuffled_mean"),
                "holds": results[key]["ordering"]["holds"],
            }
    return {
        "holds_all": all(r["ordering"]["holds"] for r in results.values()),
        "ordering": {name: r["ordering"] for name, r in results.items()},
        "capacity_curve": capacity_curve,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--graph", choices=["pc", "erp_fraud"], default="pc")
    parser.add_argument("--n", type=int, default=2000, help="graph size (CPU-scale by default)")
    parser.add_argument("--n-patches", type=int, default=50)
    parser.add_argument("--capacities", type=int, nargs="+", default=[8, 32, 64, 128])
    parser.add_argument("--router-epochs", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--router-seed", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--frozen-perms", type=int, default=10)
    parser.add_argument("--out", default="logs")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    log = get_logger("frozen_interventions", config.get("logging", {}).get("level", "INFO"))
    args.device = device_config(args.device)
    set_seed(args.router_seed)

    t0 = time.perf_counter()
    if args.graph == "erp_fraud":
        from spectral_distillation.src.synthetic import generate_erp_fraud_graph

        graph = generate_erp_fraud_graph(
            n_nodes=args.n, n_patches=args.n_patches, seed=args.router_seed
        )
    else:
        pc = config.get("planted_control", {})
        graph = generate_pc_graph(
            n_nodes=args.n,
            n_patches=args.n_patches,
            regime_ratio=tuple(pc.get("regime_ratio", [0.4, 0.4, 0.2])),
            seed=args.router_seed,
        )
    log.info("generated %s graph n=%d in %.1fs", args.graph, args.n, time.perf_counter() - t0)

    log.info("computing condition assignments...")
    assignments = build_assignments(graph, args)

    log.info("running frozen-gate ablations...")
    results = run_all(graph, assignments, args, log)

    record = {
        "config": {k: v for k, v in args.__dict__.items()},
        "graph": {
            "name": args.graph,
            "n_nodes": graph["W"].shape[0],
            "n_classes": graph["n_classes"],
            "regime_ratio": graph["regime_ratio"],
        },
        "assignments_usage": {
            name: np.bincount(np.argmax(a, axis=1), minlength=3).tolist()
            for name, a in assignments.items()
        },
        "results": results,
        "summary": _summary_table(results, list(args.capacities)),
    }

    out_dir = ensure_dir(Path(args.out) / "frozen_interventions")
    out_path = out_dir / f"{args.graph}.json"
    out_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    log.info("wrote %s", out_path)

    print(f"\n=== Frozen-intervention ablations ({args.graph}, n={args.n}) ===")
    print(f"{'architecture':<12}{'learned':>10}{'oracle':>10}{'node_shuf':>11}{'w/in_reg':>11}{'holds':>7}")
    for name, r in results.items():
        print(
            f"{name:<12}{r['learned']:>10.4f}{r.get('oracle_onehot', float('nan')):>10.4f}"
            f"{r.get('node_shuffled_mean', float('nan')):>11.4f}"
            f"{r.get('within_regime_shuffled_mean', float('nan')):>11.4f}"
            f"{'yes' if r['ordering']['holds'] else 'no':>7}"
        )
    print("\nOrdering holds for all architectures:", record["summary"]["holds_all"])


if __name__ == "__main__":
    main()

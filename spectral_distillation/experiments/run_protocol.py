"""Run the full D1-D4 router-fidelity protocol (guide sections 6.5-6.7).

Builds a PC-1c-R planted control graph (or, with ``--graph erp_fraud``, a
scalable synthetic ERP-fraud graph with the same protocol contract),
computes the four condition assignments (oracle, random, uniform,
label-free), trains identical expert heads per condition, and reports the
primary comparisons plus frozen-model gate interventions. Verdict: all four
decision rules must hold.
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
from spectral_distillation.src.planted_control import (
    generate_pc_graph,
    single_m_expert_accuracy,
)
from spectral_distillation.src.router_protocol import (
    evaluate_decision_rules,
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

CONDITIONS = ("oracle", "label_free", "random", "uniform")


def build_conditions(graph: dict, args) -> dict[str, np.ndarray]:
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


def _pairs_dict(results) -> dict:
    """Plain-dict comparison table from a :class:`ProtocolResults`."""
    return {
        name: {
            "mean_diff": c.mean_diff,
            "ci_low": c.ci_low,
            "ci_high": c.ci_high,
            "p_wilcoxon": c.p_wilcoxon,
            "p_paired_t": c.p_paired_t,
            "effect_size_dz": c.effect_size_dz,
            "significant": c.significant,
        }
        for name, c in combine_comparisons(results).items()
    }


def assert_no_saturation(graph: dict, log, lo: float = 0.3, hi: float = 0.98) -> float | None:
    """Assert the planted graph is informative: a single-M expert is not perfect.

    The control is only a valid instrument if routing *can* matter, i.e. no arm
    saturates. If a feature-only expert already scores near the ceiling, the
    routing signal is unmeasurable and every D-rule is vacuous. Raises if the
    invariant is violated on a planted graph that carries regimes.
    """
    if "regimes" not in graph:
        return None
    acc = single_m_expert_accuracy(graph, seed=0)
    if not (lo < acc < hi):
        raise AssertionError(
            f"no-saturation invariant violated: single-M expert accuracy {acc:.3f} "
            f"not in ({lo}, {hi}); the control cannot support a routing claim."
        )
    log.info("no-saturation invariant holds: single-M expert acc=%.3f in (%.2f, %.2f)",
             acc, lo, hi)
    return acc


def _summarize(results, args, graph, assignments, control_comparisons=None) -> dict:
    pairs = _pairs_dict(results)
    # D4 must be an *independent* mechanism-matched positive control: oracle vs
    # uniform on a fresh PC-1c-R realisation, not the main run's own D1 numbers.
    # Passing ``pairs`` as its own control (the old behaviour) makes D4 == D1 by
    # construction, so the "instrument works" rule can never fail independently
    # and carries no information.
    decisions = evaluate_decision_rules(
        comparisons=pairs,
        control_comparisons=control_comparisons,
    )
    frozen = frozen_intervention(
        learned=assignments["label_free"],
        n_nodes=graph["W"].shape[0],
        oracle=assignments["oracle"],
        run_expert_fn=lambda assignment: _expert_accuracy_from_assignment(graph, assignment),
        n_permutations=args.frozen_perms,
        seed=args.router_seed,
    )
    return {
        "config": args.__dict__,
        "graph": {
            "n_nodes": graph["W"].shape[0],
            "n_classes": graph["n_classes"],
            "regime_ratio": graph["regime_ratio"],
        },
        "assignments_usage": {
            name: np.bincount(np.argmax(a, axis=1), minlength=3).tolist()
            for name, a in assignments.items()
        },
        "findings": decisions.__dict__,
        "comparisons": pairs,
        "control_comparisons": control_comparisons,
        "frozen": {k: v for k, v in frozen.items() if not isinstance(v, list)},
    }


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


def build_graph(args, seed: int, pc: dict) -> dict:
    """Generate the protocol graph (PC-1c-R or ERP-fraud) at ``seed``."""
    if args.graph == "erp_fraud":
        from spectral_distillation.src.synthetic import generate_erp_fraud_graph

        return generate_erp_fraud_graph(
            n_nodes=args.n,
            n_patches=args.n_patches,
            seed=seed,
        )
    return generate_pc_graph(
        n_nodes=args.n,
        n_patches=args.n_patches,
        regime_ratio=tuple(pc.get("regime_ratio", [0.4, 0.4, 0.2])),
        seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--graph", choices=["pc", "erp_fraud"], default="pc")
    parser.add_argument("--n", type=int, default=None, help="graph size (default uses config, 10000)")
    parser.add_argument("--n-patches", type=int, default=None)
    parser.add_argument("--splits", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--router-epochs", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--router-seed", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--control-seed", type=int, default=None,
                        help="seed of the independent PC-1c-R positive control "
                             "(default: router-seed + 1000)")
    parser.add_argument("--control-splits", type=int, default=None,
                        help="splits for the independent D4 control arm "
                             "(default: same as --splits)")
    parser.add_argument("--control-seeds", type=int, default=None,
                        help="seeds for the independent D4 control arm "
                             "(default: same as --seeds)")
    parser.add_argument("--frozen-perms", type=int, default=20)
    parser.add_argument("--out", default="logs")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    log = get_logger("run_protocol", config.get("logging", {}).get("level", "INFO"))
    pc = config.get("planted_control", {})
    proto = config.get("router_protocol", {}).get("protocol", {}) if config.get("router_protocol") else config.get("protocol", {})

    args.n = args.n or pc.get("n_nodes", 10000)
    args.n_patches = args.n_patches or pc.get("n_patches", 50)
    args.splits = args.splits or proto.get("n_splits", 10)
    args.seeds = args.seeds or proto.get("n_seeds", 3)
    args.router_epochs = args.router_epochs or 200
    args.hidden_dim = args.hidden_dim or 64
    args.n_layers = args.n_layers or 3
    args.device = device_config(args.device)

    set_seed(args.router_seed)
    if args.control_seed is None:
        args.control_seed = args.router_seed + 1000
    if args.control_splits is None:
        args.control_splits = args.splits
    if args.control_seeds is None:
        args.control_seeds = args.seeds

    t0 = time.perf_counter()
    graph = build_graph(args, args.router_seed, pc)
    log.info("generated %s graph n=%d (%d patches) in %.1fs",
             args.graph, args.n, args.n_patches, time.perf_counter() - t0)
    assert_no_saturation(graph, log)

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
    )

    for c in results.comparisons:
        log.info(
            "%s: mean_diff=%.4f p_wilcoxon=%.4g d_z=%.3f significant=%s",
            c.name,
            c.mean_diff,
            c.p_wilcoxon,
            c.effect_size_dz,
            c.significant,
        )

    log.info("running independent D4 control (seed=%d, %d splits x %d seeds)...",
             args.control_seed, args.control_splits, args.control_seeds)
    t0 = time.perf_counter()
    control_graph = build_graph(args, args.control_seed, pc)
    assert_no_saturation(control_graph, log)
    # D4 only needs oracle vs uniform, so skip the (expensive) label-free
    # router training on the control graph.
    c_oracle = oracle_regime_assignment(control_graph["regimes"], 3)
    c_uniform = uniform_assignment(control_graph["W"].shape[0], 3)
    control_assignments = {
        "oracle": c_oracle,
        "uniform": c_uniform,
        "random": random_assignment(c_oracle, rng=args.random_seed),
        "label_free": c_uniform,
    }
    control_results = run_protocol(
        run_fn=lambda cond, split, seed: cell_accuracy(
            control_graph, control_assignments, cond, split, seed
        ),
        n_splits=args.control_splits,
        n_seeds=args.control_seeds,
    )
    control_comparisons = _pairs_dict(control_results)
    log.info("control arm ready in %.1fs (oracle_vs_uniform p=%.4g)",
             time.perf_counter() - t0,
             control_comparisons.get("oracle_vs_uniform", {}).get("p_wilcoxon", float("nan")))

    record = _summarize(results, args, graph, assignments, control_comparisons)
    out_dir = ensure_dir(Path(args.out) / ("erp_fraud_protocol" if args.graph == "erp_fraud" else "protocol"))
    out_path = out_dir / "protocol_results.json"
    out_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    log.info("wrote %s", out_path)

    print("\n=== Router fidelity protocol ===")
    print(f"{'comparison':<22}{'mean_diff':>10}{'p_wilcoxon':>12}{'d_z':>8}{'sig':>6}")
    for c in results.comparisons:
        print(
            f"{c.name:<22}{c.mean_diff:>10.4f}{c.p_wilcoxon:>12.4g}{c.effect_size_dz:>8.3f}"
            f"{'yes' if c.significant else 'no':>6}"
        )
    print("\nDecision rules:", record["findings"]["verdict"])
    ctrl = record.get("control_comparisons") or {}
    cov = ctrl.get("oracle_vs_uniform", {})
    print("D4 control (seed=%d): oracle_vs_uniform mean_diff=%.4f p=%.4g"
          % (args.control_seed, cov.get("mean_diff", float("nan")),
             cov.get("p_wilcoxon", float("nan"))))
    print("Frozen gate (%s perms):" % args.frozen_perms)
    for k, v in record["frozen"].items():
        print(f"  {k:<18}{v:>10.4f}")


if __name__ == "__main__":
    main()
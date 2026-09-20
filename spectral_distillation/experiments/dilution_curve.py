"""Dilution ladder experiment (guide section 6.8).

Measures the routing-signal response to dilution: as an increasing fraction of
nodes are converted to homogeneous low-pass wiring, the oracle-routing gain
(oracle minus uniform) and the label-free-routing gain (label-free minus
random) are expected to decay toward zero. Runs the ladder across multiple
seeds and reports per-dilution means with the oracle-gain decay slope.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from spectral_distillation.src.evaluation import run_protocol, combine_comparisons
from spectral_distillation.src.mixture import fit_experts, mixture_accuracy, train_test_split
from spectral_distillation.src.planted_control import generate_pc_graph, dilution_ladder
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


def build_conditions(graph: dict, args) -> dict[str, np.ndarray]:
    W, X, regimes = graph["W"], graph["features"], graph["regimes"]
    oracle = oracle_regime_assignment(regimes, 3)
    uniform = uniform_assignment(W.shape[0], 3)
    random = random_assignment(oracle, rng=args.random_seed)
    label_free, _ = label_free_assignment(
        W, X, epochs=args.router_epochs, seed=args.router_seed, device=args.device
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


def run_ladder(graph: dict, args, log) -> dict:
    ladder = dilution_ladder(graph, dilution_values=args.dilutions, seed=args.dilution_seed)
    rows: list[dict] = []
    for g in ladder:
        dil = g["dilution"]
        assignments = build_conditions(g, args)
        results = run_protocol(
            run_fn=lambda cond, split, seed: cell_accuracy(g, assignments, cond, split, seed),
            n_splits=args.splits,
            n_seeds=args.seeds,
        )
        comparisons = combine_comparisons(results)
        oracle_gain = comparisons["oracle_vs_uniform"].mean_diff
        label_free_gain = comparisons["label_free_vs_random"].mean_diff
        log.info(
            "dilution=%.2f oracle_gain=%.4f label_free_gain=%.4f",
            dil,
            oracle_gain,
            label_free_gain,
        )
        rows.append(
            {
                "dilution": dil,
                "oracle_gain": oracle_gain,
                "label_free_gain": label_free_gain,
            }
        )
    return {"ladder": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--n-patches", type=int, default=None)
    parser.add_argument("--splits", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--router-epochs", type=int, default=None)
    parser.add_argument("--dilutions", type=float, nargs="+", default=None)
    parser.add_argument("--router-seed", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--dilution-seed", type=int, default=3)
    parser.add_argument("--out", default="logs")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    log = get_logger("dilution_curve", config.get("logging", {}).get("level", "INFO"))
    pc = config.get("planted_control", {})
    dil_cfg = config.get("experiments", {}).get("dilution", {})

    args.n = args.n or pc.get("n_nodes", 10000)
    args.n_patches = args.n_patches or pc.get("n_patches", 50)
    args.splits = args.splits or 6
    args.seeds = args.seeds or 2
    args.router_epochs = args.router_epochs or 60
    args.device = device_config(args.device)
    if args.dilutions is None:
        args.dilutions = dil_cfg.get("dilution_values", [0.0, 0.25, 0.5, 0.75, 1.0])

    set_seed(args.router_seed)

    t0 = time.perf_counter()
    graph = generate_pc_graph(
        n_nodes=args.n,
        n_patches=args.n_patches,
        regime_ratio=tuple(pc.get("regime_ratio", [0.4, 0.4, 0.2])),
        seed=args.router_seed,
    )
    log.info("generated PC-1c-R graph n=%d in %.1fs", args.n, time.perf_counter() - t0)

    record = run_ladder(graph, args, log)

    out_dir = ensure_dir(Path(args.out) / "dilution")
    csv_path = out_dir / "dilution_curve.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(record["ladder"][0].keys()))
        writer.writeheader()
        writer.writerows(record["ladder"])

    full_path = out_dir / "dilution_curve.json"
    full_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    log.info("wrote %s and %s", csv_path, full_path)

    print("\n=== Dilution ladder ===")
    print(f"{'dilution':>10}{'oracle_gain':>14}{'label_free_gain':>16}")
    for row in record["ladder"]:
        print(
            f"{row['dilution']:>10.2f}{row['oracle_gain']:>14.4f}"
            f"{row['label_free_gain']:>16.4f}"
        )


if __name__ == "__main__":
    main()
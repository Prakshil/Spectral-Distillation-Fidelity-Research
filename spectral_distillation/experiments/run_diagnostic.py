"""Pre-training spectral distortion diagnostic on synthetic ERP attention graphs.

Computes the SD, eigengap, fragility, and predicted routing retention for
threshold, random, degree, and spectral sparsification before any GNN is trained,
and reproduces the golden 4-node example from the guide.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from spectral_distillation.src.baselines import degree_sparsify, random_sparsify, threshold_sparsify
from spectral_distillation.src.distortion import (
    compute_fragility,
    compute_frobenius_norm,
    compute_spectral_distortion,
    predict_retention,
    spectral_distortion_report,
)
from spectral_distillation.src.laplacian import compute_laplacian
from spectral_distillation.src.spectral import count_zero_eigenvalues, fiedler_value, min_eigengap
from spectral_distillation.src.sparsifier import spectral_sparsify
from spectral_distillation.src.synthetic import make_erp_attention
from spectral_distillation.src.utils import ensure_dir, get_logger, load_yaml_config, set_seed

GOLDEN_ATTENTION = np.array(
    [
        [0.0, 0.8, 0.6, 0.3],
        [0.8, 0.0, 0.9, 0.2],
        [0.6, 0.9, 0.0, 0.1],
        [0.3, 0.2, 0.1, 0.0],
    ]
)


def golden_example() -> dict:
    L_full = compute_laplacian(GOLDEN_ATTENTION)
    W_thr = threshold_sparsify(GOLDEN_ATTENTION, 0.5)
    L_sparse = compute_laplacian(W_thr)
    return spectral_distortion_report(L_full, L_sparse)


def _tau_for_budget(W: np.ndarray, budget: int) -> float:
    rows, cols, weights = _edges(W)
    if weights.size == 0:
        return 1.0
    keep = int(max(1, min(budget, weights.size)))
    return float(np.partition(weights, weights.size - keep)[weights.size - keep])


def _edges(W):
    rows, cols = np.triu_indices(W.shape[0], k=1)
    weights = W[rows, cols]
    keep = weights > 0
    return rows[keep], cols[keep], weights[keep]


def _method_results(W_full: np.ndarray, budget: int, seed: int) -> dict:
    L_full = compute_laplacian(W_full)
    e_full = np.sort(np.linalg.eigvalsh(L_full))
    evals_full = e_full
    outputs = {}
    for name, W_sp in (
        ("threshold", threshold_sparsify(W_full, _tau_for_budget(W_full, budget))),
        ("random", random_sparsify(W_full, budget, seed=seed)),
        ("degree", degree_sparsify(W_full, budget, seed=seed)),
        ("spectral", spectral_sparsify(W_full, budget, seed=seed)),
    ):
        L_sp = compute_laplacian(W_sp)
        e_sp = np.sort(np.linalg.eigvalsh(L_sp))
        sd = compute_spectral_distortion(L_full, L_sp)
        outputs[name] = {
            "SD": sd,
            "retention": predict_retention(sd),
            "fragility": compute_fragility(sd, min_eigengap(evals_full)),
            "frobenius_norm": compute_frobenius_norm(L_full, L_sp),
            "fiedler_full": fiedler_value(evals_full),
            "fiedler_sparse": fiedler_value(e_sp),
            "zero_eigenvalues": count_zero_eigenvalues(e_sp),
            "n_edges_sparse": int(np.count_nonzero(np.triu(W_sp, 1))),
        }
    outputs["full"] = {
        "SD": 0.0,
        "fiedler_full": fiedler_value(evals_full),
        "min_eigengap": min_eigengap(evals_full),
        "n_edges_sparse": int(np.count_nonzero(np.triu(W_full, 1))),
    }
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--budget-factor", type=float, default=None)
    parser.add_argument("--out", default="logs")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    log = get_logger("run_diagnostic", config.get("logging", {}).get("level", "INFO"))
    seed_cfg = config.get("experiments", {}).get("diagnostic", {})
    n = args.n or config.get("synthetic", {}).get("n", 256)
    seeds = args.seeds or seed_cfg.get("seeds", 3)
    factor = args.budget_factor or seed_cfg.get("budget_factor", 10)

    set_seed(config.get("seed", 42))

    golden = golden_example()
    log.info(
        "Golden 4-node example: SD=%.4f frobenius=%.4f min_eigengap=%.4f fragility=%.4f retention=%.4f",
        golden["SD"],
        golden["frobenius_norm"],
        golden["min_eigengap"],
        golden["fragility_score"],
        golden["predicted_retention"],
    )

    per_method: dict[str, list] = {}
    for seed in range(seeds):
        graph = make_erp_attention(n, seed=seed)
        W_full = graph["attention"]
        budget = min(int(factor * n), int(np.count_nonzero(np.triu(W_full, 1))))
        t0 = time.perf_counter()
        results = _method_results(W_full, budget, seed)
        log.info(
            "seed=%d n=%d budget=%d: spectral SD=%.4f threshold SD=%.4f (%.2fs)",
            seed,
            n,
            budget,
            results["spectral"]["SD"],
            results["threshold"]["SD"],
            time.perf_counter() - t0,
        )
        for name, metrics in results.items():
            per_method.setdefault(name, []).append(metrics)

    summary: dict = {}
    for name, runs in per_method.items():
        fields = set(runs[0].keys())
        summary[name] = {
            field: {
                "mean": float(statistics.mean(r[field] for r in runs)),
                "std": float(statistics.pstdev(r[field] for r in runs)),
            }
            for field in fields
            if isinstance(runs[0][field], (int, float))
        }

    record = {
        "config": config,
        "golden_4node": golden,
        "erp_diagnostic": {
            "n": n,
            "seeds": seeds,
            "budget": budget,
            "summary": summary,
        },
    }

    out_dir = ensure_dir(Path(args.out) / "diagnostic")
    out_path = out_dir / "diagnostic.json"
    out_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    log.info("wrote %s", out_path)

    print("\n=== ERP diagnostic summary (n=%d, budget=%d, seeds=%d) ===" % (n, budget, seeds))
    header = f"{'method':<10}{'SD':>10}{'retention':>12}{'fragility':>12}{'Fiedler':>10}{'zero_eigs':>10}"
    print(header)
    print("-" * len(header))
    for name, metrics in summary.items():
        sd_mean = metrics["SD"]["mean"]
        ret = metrics.get("retention", {}).get("mean", None)
        frag = metrics.get("fragility", {}).get("mean", None)
        fied = metrics.get("fiedler_sparse", metrics.get("fiedler_full"))
        zero = metrics.get("zero_eigenvalues")
        row = f"{name:<10}{sd_mean:>10.4f}"
        row += f"{ret:>12.4f}" if ret is not None else f"{'-':>12}"
        row += f"{frag:>12.4f}" if frag is not None else f"{'-':>12}"
        row += f"{fied['mean'] if fied else 0:>10.4f}"
        row += f"{int(zero['mean']) if zero else '-':>10}"
        print(row)
    print("\nGolden 4-node (guide narrative): SD=%.4f (threshold disconnects Customer)" % golden["SD"])


if __name__ == "__main__":
    main()
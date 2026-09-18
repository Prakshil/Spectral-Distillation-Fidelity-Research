"""Method comparison: SD vs edge-budget curves for all four distillation methods.

Threshold, random, degree, and spectral sparsification are run at increasing edge
budgets across synthetic ERP graphs of increasing size. Outputs a tidy CSV, a
JSON record, and (when matplotlib is available) PNG curves.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from spectral_distillation.src.baselines import degree_sparsify, random_sparsify, threshold_sparsify
from spectral_distillation.src.distortion import (
    compute_fragility,
    compute_frobenius_norm,
    compute_spectral_distortion,
)
from spectral_distillation.src.laplacian import compute_laplacian
from spectral_distillation.src.spectral import fiedler_value, min_eigengap
from spectral_distillation.src.sparsifier import spectral_sparsify
from spectral_distillation.src.synthetic import make_erp_attention
from spectral_distillation.src.utils import ensure_dir, get_logger, load_yaml_config, set_seed

METHODS = ("threshold", "random", "degree", "spectral")


def _tau_for_budget(W: np.ndarray, budget: int) -> float:
    rows, cols = np.triu_indices(W.shape[0], k=1)
    weights = W[rows, cols]
    weights = weights[weights > 0]
    if weights.size == 0:
        return 1.0
    keep = int(max(1, min(budget, weights.size)))
    return float(np.partition(weights, weights.size - keep)[weights.size - keep])


def evaluate(W_full: np.ndarray, n_edges_budget: int, seed: int) -> dict:
    L_full = compute_laplacian(W_full)
    e_full = np.sort(np.linalg.eigvalsh(L_full))
    min_gap = min_eigengap(e_full)
    fiedler_full = fiedler_value(e_full)

    sparsifiers = {
        "threshold": lambda: threshold_sparsify(W_full, _tau_for_budget(W_full, n_edges_budget)),
        "random": lambda: random_sparsify(W_full, n_edges_budget, seed=seed),
        "degree": lambda: degree_sparsify(W_full, n_edges_budget, seed=seed),
        "spectral": lambda: spectral_sparsify(W_full, n_edges_budget, seed=seed),
    }

    row = {"n_edges_budget": n_edges_budget, "fiedler_full": fiedler_full, "min_eigengap": min_gap}
    for name in METHODS:
        t0 = time.perf_counter()
        W_sp = sparsifiers[name]()
        L_sp = compute_laplacian(W_sp)
        sd = compute_spectral_distortion(L_full, L_sp)
        e_sp = np.sort(np.linalg.eigvalsh(L_sp))
        row[f"{name}_SD"] = sd
        row[f"{name}_fiedler"] = fiedler_value(e_sp)
        row[f"{name}_fragility"] = compute_fragility(sd, min_gap)
        row[f"{name}_frobenius"] = compute_frobenius_norm(L_full, L_sp)
        row[f"{name}_edges"] = int(np.count_nonzero(np.triu(W_sp, 1)))
        row[f"{name}_seconds"] = time.perf_counter() - t0
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--ns", default=None, help="comma-separated node counts")
    parser.add_argument("--budget-factors", default=None, help="comma-separated factors (edges = factor*n)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default="logs")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    log = get_logger("compare_methods", config.get("logging", {}).get("level", "INFO"))
    exp_cfg = config.get("experiments", {}).get("compare_methods", {})
    ns = [int(v) for v in args.ns.split(",")] if args.ns else list(exp_cfg.get("ns", [64, 128, 256]))
    factors = (
        [float(v) for v in args.budget_factors.split(",")]
        if args.budget_factors
        else list(exp_cfg.get("budget_factors", [1, 2, 4, 6, 10, 15]))
    )
    seed = args.seed if args.seed is not None else exp_cfg.get("seed", 0)
    set_seed(seed)

    out_dir = ensure_dir(Path(args.out) / "compare_methods")
    csv_path = out_dir / "results.csv"
    rows: list[dict] = []

    for n in ns:
        graph = make_erp_attention(n, seed=seed)
        W_full = graph["attention"]
        n_edges = int(np.count_nonzero(np.triu(W_full, 1)))
        log.info("n=%d edges=%d", n, n_edges)
        for factor in factors:
            budget = min(int(factor * n), n_edges)
            if budget < 1:
                continue
            row = evaluate(W_full, budget, seed)
            row = {"n": n, "method_group": METHODS, **row}
            rows.append(row)
            log.info(
                "  n=%d factor=%.1f budget=%d  threshold SD=%.4f  spectral SD=%.4f",
                n,
                factor,
                budget,
                row["threshold_SD"],
                row["spectral_SD"],
            )

    fieldnames = ["n", "n_edges_budget", "fiedler_full", "min_eigengap"]
    for method in METHODS:
        fieldnames += [
            f"{method}_SD",
            f"{method}_fiedler",
            f"{method}_fragility",
            f"{method}_frobenius",
            f"{method}_edges",
            f"{method}_seconds",
        ]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})
    log.info("wrote %s (%d rows)", csv_path, len(rows))

    json_path = out_dir / "results.json"
    json_path.write_text(
        json.dumps({"config": config, "seed": seed, "ns": ns, "budget_factors": factors, "rows": rows}, indent=2, default=str),
        encoding="utf-8",
    )

    if not args.no_plots:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            log.info("matplotlib not installed; skipping plots")
        else:
            for n in ns:
                subset = [r for r in rows if r["n"] == n]
                fig, ax = plt.subplots(figsize=(7, 5))
                for method in METHODS:
                    budgets = [r["n_edges_budget"] for r in subset]
                    sds = [r[f"{method}_SD"] for r in subset]
                    ax.plot(budgets, sds, marker="o", label=method)
                ax.set_xlabel("edge budget")
                ax.set_ylabel("spectral distortion (SD)")
                ax.set_title(f"SD vs edge budget, synthetic ERP n={n}")
                ax.legend()
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(out_dir / f"sd_curve_n{n}.png", dpi=150)
                plt.close(fig)
            log.info("wrote SD curves to %s", out_dir)

    print("\n=== SD vs budget summary (spectral vs threshold) ===")
    print(f"{'n':>6}{'budget':>8}{'thr SD':>10}{'spec SD':>10}{'reduction':>10}")
    for row in rows:
        if row["n_edges_budget"] == row["n"] or row["n_edges_budget"] == 2 * row["n"]:
            thr, spec = row["threshold_SD"], row["spectral_SD"]
            print(
                f"{row['n']:>6}{row['n_edges_budget']:>8}{thr:>10.4f}{spec:>10.4f}"
                f"{100 * (thr - spec) / max(thr, 1e-9):>9.1f}%"
            )


if __name__ == "__main__":
    main()
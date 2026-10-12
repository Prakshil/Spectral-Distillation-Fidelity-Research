"""Method comparison: SD vs edge-budget curves for all four distillation methods.

Threshold, random, degree, and spectral sparsification are run at increasing edge
budgets across synthetic ERP graphs of increasing size, over a *grid* of
sparsification tolerances (``epsilon`) and random seeds. Outputs a tidy per-run
CSV, an aggregated band CSV/JSON (mean +/- std across seeds for every metric),
and (when matplotlib is available) PNG curves with shaded bands.

Phase 6 (robustness): the ``epsilon x budget x seed`` grid turns the old
single-point Table 2 numbers into bands, and :func:`check_bands` reports
whether the spectral method stays below the threshold/degree/random baselines
across the whole band rather than at one hand-picked point.

Equal-edge-mass comparison (default; disable with ``--no-mass-match``): each
sparsified graph is rescaled to the full graph's total edge weight before any
distortion is measured, so the metrics compare structure rather than the
incidental weight-scale difference between the inverse-propensity spectral
rescale and the raw-weight heuristics (see :func:`evaluate`).

Note on what ``epsilon`` controls here: with ``sample=False`` (the default) the
sampling rule
    ``q_i = min(1, C * w_i * r_i * log(n) / eps**2)``
ranks edges identically for every ``epsilon`` (the `1/eps**2` factor is a
positive global constant, so a top-budget selection is epsilon-invariant). What
epsilon *does* change is the unbiased weight rescaling ``w/q`` -- smaller epsilon
inflates retained weights less. The epsilon band therefore measures the
sensitivity of the downstream distortion to that rescaling, which is exactly the
``(1 +/- eps)`` spectral guarantee restated for the deterministic variant.
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
    low_frequency_distortion,
    rank_matched_distortion,
)
from spectral_distillation.src.laplacian import compute_laplacian
from spectral_distillation.src.spectral import fiedler_value, min_eigengap
from spectral_distillation.src.sparsifier import spectral_sparsify
from spectral_distillation.src.synthetic import make_erp_attention
from spectral_distillation.src.utils import ensure_dir, get_logger, load_yaml_config, set_seed

METHODS = ("threshold", "random", "degree", "spectral")
# Robustness is reported on several metrics because they disagree on purpose:
# - "sd" is the legacy index-matched distortion (saturates at 1.0 once
#   sparsification fragments the graph),
# - "rank_matched" is the top-8 *largest* (high-frequency) relative error, where
#   heavy-edge methods have an edge,
# - "low_freq" is the smooth/low-mode error a router actually consumes
#   (docs/reproducibility.md); it is the metric the robustness claim is made on.
BAND_METRICS = ("sd", "retention", "rank_matched", "low_freq", "lmax_rel", "fiedler", "fragility")


def _tau_for_budget(W: np.ndarray, budget: int) -> float:
    rows, cols = np.triu_indices(W.shape[0], k=1)
    weights = W[rows, cols]
    weights = weights[weights > 0]
    if weights.size == 0:
        return 1.0
    keep = int(max(1, min(budget, weights.size)))
    return float(np.partition(weights, weights.size - keep)[weights.size - keep])


def evaluate(
    W_full: np.ndarray,
    n_edges_budget: int,
    seed: int,
    epsilon: float = 0.1,
    mass_match: bool = True,
) -> dict:
    L_full = compute_laplacian(W_full)
    e_full = np.sort(np.linalg.eigvalsh(L_full))
    tw_full = float(np.triu(W_full, 1).sum())
    row = {
        "n_edges_budget": n_edges_budget,
        "epsilon": float(epsilon),
        "seed": int(seed),
        "mass_match": bool(mass_match),
        "fiedler_full": fiedler_value(e_full),
        "min_eigengap": min_eigengap(e_full),
    }
    sparsifiers = {
        "threshold": lambda: threshold_sparsify(W_full, _tau_for_budget(W_full, n_edges_budget)),
        "random": lambda: random_sparsify(W_full, n_edges_budget, seed=seed),
        "degree": lambda: degree_sparsify(W_full, n_edges_budget, seed=seed),
        "spectral": lambda: spectral_sparsify(
            W_full, n_edges_budget, epsilon=epsilon, seed=seed
        ),
    }
    for method in METHODS:
        t0 = time.perf_counter()
        W_sparse = sparsifiers[method]()
        tw = float(np.triu(W_sparse, 1).sum())
        row[f"{method}_mass_ratio"] = float(tw / tw_full) if tw_full > 0 else 0.0
        # Equal-edge-mass comparison: the deterministic spectral estimator divides
        # retained weights by q (an inverse-propensity rescale that only makes sense
        # under sampling), so it sheds most of the graph's total weight. Threshold
        # and degree keep heavy edges and barely rescale. Because every distortion
        # metric here lives on the *unnormalized* Laplacian, that global scale
        # difference dominates the comparison; rescaling each sparsified graph to
        # the full graph's total weight removes the artifact so the metrics measure
        # structure, not edge-mass bookkeeping.
        if mass_match and tw > 0:
            W_sparse = W_sparse * (tw_full / tw)
        L_sparse = compute_laplacian(W_sparse)
        sd = compute_spectral_distortion(L_full, L_sparse)
        row[f"{method}_SD"] = sd
        row[f"{method}_retention"] = float(max(0.0, 1.0 - sd))
        row[f"{method}_fiedler"] = fiedler_value(np.linalg.eigvalsh(L_sparse))
        row[f"{method}_fragility"] = compute_fragility(
            sd, min_eigengap(np.linalg.eigvalsh(L_sparse))
        )
        row[f"{method}_edges"] = int(np.count_nonzero(np.triu(W_sparse, 1)))
        row[f"{method}_frobenius"] = compute_frobenius_norm(L_full, L_sparse)
        rm = rank_matched_distortion(L_full, L_sparse, top_ks=(1, 8, 32))
        row[f"{method}_top8_rel"] = rm["top8_rel_error"]
        row[f"{method}_lmax_rel"] = rm["lambda_max_rel_error"]
        lfd = low_frequency_distortion(L_full, L_sparse, k=8)
        row[f"{method}_lowfreq"] = lfd["low_freq_mean_error"]
        row[f"{method}_seconds"] = time.perf_counter() - t0
    return row


def _method_metric(row: dict, method: str, metric: str) -> float:
    if metric == "sd":
        return float(row[f"{method}_SD"])
    if metric == "retention":
        return float(row[f"{method}_retention"])
    if metric == "rank_matched":
        return float(row[f"{method}_top8_rel"])
    if metric == "low_freq":
        return float(row[f"{method}_lowfreq"])
    if metric == "lmax_rel":
        return float(row[f"{method}_lmax_rel"])
    if metric == "fiedler":
        return float(row[f"{method}_fiedler"])
    if metric == "fragility":
        return float(row[f"{method}_fragility"])
    raise ValueError(f"unknown band metric: {metric}")


def aggregate_bands(rows: list[dict], metrics=BAND_METRICS) -> list[dict]:
    """Mean/std/min/max of each method metric across seeds per (n, factor, epsilon)."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        key = (r["n"], r["budget_factor"], r["epsilon"])
        groups.setdefault(key, []).append(r)

    bands: list[dict] = []
    for (n, factor, eps), members in sorted(groups.items()):
        for method in METHODS:
            band = {
                "n": n,
                "budget_factor": factor,
                "n_edges_budget": members[0]["n_edges_budget"],
                "epsilon": eps,
                "method": method,
                "n_seeds": len(members),
                "min_eigengap_mean": float(np.mean([m["min_eigengap"] for m in members])),
            }
            for metric in metrics:
                vals = np.array([_method_metric(m, method, metric) for m in members])
                band[f"{metric}_mean"] = float(vals.mean())
                band[f"{metric}_std"] = float(vals.std())
                band[f"{metric}_min"] = float(vals.min())
                band[f"{metric}_max"] = float(vals.max())
            bands.append(band)
    return bands


def check_bands(bands: list[dict], metric: str = "low_freq") -> dict:
    """Does the spectral method stay below the threshold/random/degree baselines?

    Evaluated at every (n, budget factor, epsilon) grid point -- the robustness
    claim is that this holds across the whole band, not at a single epsilon.
    Defaults to the smooth low-mode distortion (lower is better), the quantity a
    low-pass router consumes; heavy-edge baselines can look good on the
    high-frequency end while collapsing the smooth modes this metric measures.
    """
    key = f"{metric}_mean"
    vals = {
        (b["n"], b["budget_factor"], b["epsilon"], b["method"]): b[key]
        for b in bands
    }
    points: list[dict] = []
    for (n, factor, eps, method) in sorted(vals):
        if method != "spectral":
            continue
        spec = vals[(n, factor, eps, "spectral")]
        below = {
            name: bool(spec < vals[(n, factor, eps, name)])
            for name in ("threshold", "random", "degree")
        }
        points.append(
            {
                "n": n,
                "budget_factor": factor,
                "epsilon": eps,
                "spectral_metric": spec,
                **{f"below_{k}": v for k, v in below.items()},
                "below_all_baselines": bool(all(below.values())),
            }
        )
    n_ok = sum(p["below_all_baselines"] for p in points)
    return {
        "metric": "rank_matched(top8)" if metric == "rank_matched" else metric,
        "n_points": len(points),
        "n_below_all_baselines": n_ok,
        "fraction_below_all_baselines": float(n_ok / max(1, len(points))),
        "holds_everywhere": bool(n_ok == len(points)),
        "points": points,
    }


def _write_plots(rows, bands, ns, factors, epsilons, out_dir, log) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.info("matplotlib not installed; skipping plots")
        return

    # Low-freq distortion vs budget, mean +/- std across seeds, at eps closest to 0.1.
    # (legacy SD saturates and is uninformative for equal-mass-matched graphs.)
    plot_eps = min(epsilons, key=lambda e: abs(e - 0.1))
    for n in ns:
        fig, ax = plt.subplots(figsize=(7, 5))
        for method in METHODS:
            subset = sorted(
                (
                    b
                    for b in bands
                    if b["n"] == n and b["method"] == method and abs(b["epsilon"] - plot_eps) < 1e-9
                ),
                key=lambda b: b["n_edges_budget"],
            )
            if not subset:
                continue
            x = np.array([b["n_edges_budget"] for b in subset])
            y = np.array([b["low_freq_mean"] for b in subset])
            s = np.array([b["low_freq_std"] for b in subset])
            ax.plot(x, y, marker="o", label=method)
            ax.fill_between(x, y - s, y + s, alpha=0.2)
        ax.set_xlabel("edge budget")
        ax.set_ylabel("low-frequency distortion")
        ax.set_title(f"low-freq distortion vs edge budget (mean +/- std, eps={plot_eps:g}), ERP n={n}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"bands_vs_budget_n{n}.png", dpi=120)
        plt.close(fig)

    # Low-freq distortion vs epsilon for a representative budget factor, per method.
    factor = factors[len(factors) // 2]
    for n in ns:
        fig, ax = plt.subplots(figsize=(7, 5))
        for method in METHODS:
            subset = sorted(
                (
                    b
                    for b in bands
                    if b["n"] == n and b["method"] == method and b["budget_factor"] == factor
                ),
                key=lambda b: b["epsilon"],
            )
            if not subset:
                continue
            x = np.array([b["epsilon"] for b in subset])
            y = np.array([b["low_freq_mean"] for b in subset])
            s = np.array([b["low_freq_std"] for b in subset])
            ax.errorbar(x, y, yerr=s, marker="o", capsize=3, label=method)
        ax.set_xscale("log")
        ax.set_xlabel("epsilon")
        ax.set_ylabel("low-frequency distortion")
        ax.set_title(f"low-freq distortion vs epsilon (budget={factor:g}x), ERP n={n}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"bands_vs_epsilon_n{n}.png", dpi=120)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--ns", default=None, help="comma-separated node counts")
    parser.add_argument("--budget-factors", default=None, help="comma-separated factors (edges = factor*n)")
    parser.add_argument("--epsilons", default=None, help="comma-separated sparsification tolerances")
    parser.add_argument("--seed", type=int, default=None, help="graph/base seed")
    parser.add_argument("--seeds", default=None, help="comma-separated per-run seeds (default 0,1,2)")
    parser.add_argument("--out", default="logs")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--no-mass-match",
        action="store_true",
        help="compare raw sparsified graphs instead of equal-edge-mass matched ones",
    )
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
    epsilons = (
        [float(v) for v in args.epsilons.split(",")]
        if args.epsilons
        else list(exp_cfg.get("epsilons", [0.02, 0.05, 0.1, 0.2, 0.4]))
    )
    seeds = (
        [int(v) for v in args.seeds.split(",")]
        if args.seeds
        else list(exp_cfg.get("seeds", [0, 1, 2]))
    )
    base_seed = args.seed if args.seed is not None else exp_cfg.get("seed", 0)
    mass_match = not args.no_mass_match
    set_seed(base_seed)

    out_dir = ensure_dir(Path(args.out) / "compare_methods")
    rows: list[dict] = []

    for n in ns:
        graph = make_erp_attention(n, seed=base_seed)
        W_full = graph["attention"]
        n_edges = int(np.count_nonzero(np.triu(W_full, 1)))
        log.info("n=%d edges=%d", n, n_edges)
        for factor in factors:
            budget = min(int(factor * n), n_edges)
            if budget < 1:
                continue
            for eps in epsilons:
                for s in seeds:
                    row = evaluate(W_full, budget, s, epsilon=eps, mass_match=mass_match)
                    row = {"n": n, "budget_factor": factor, "method_group": METHODS, **row}
                    rows.append(row)
        log.info("  evaluated %d runs for n=%d", len(rows), n)

    fieldnames = [
        "n",
        "budget_factor",
        "n_edges_budget",
        "epsilon",
        "seed",
        "mass_match",
        "min_eigengap",
        "fiedler_full",
    ]
    for method in METHODS:
        fieldnames += [
            f"{method}_SD",
            f"{method}_retention",
            f"{method}_top8_rel",
            f"{method}_lmax_rel",
            f"{method}_lowfreq",
            f"{method}_mass_ratio",
            f"{method}_fiedler",
            f"{method}_fragility",
            f"{method}_edges",
            f"{method}_frobenius",
            f"{method}_seconds",
        ]
    with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    (out_dir / "results.json").write_text(
        json.dumps(
            {
                "config": config,
                "base_seed": base_seed,
                "seeds": seeds,
                "mass_match": mass_match,
                "ns": ns,
                "budget_factors": factors,
                "epsilons": epsilons,
                "rows": rows,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    bands = aggregate_bands(rows)
    band_fields = [
        "n",
        "budget_factor",
        "n_edges_budget",
        "epsilon",
        "method",
        "n_seeds",
        "min_eigengap_mean",
    ]
    for metric in BAND_METRICS:
        band_fields += [f"{metric}_mean", f"{metric}_std", f"{metric}_min", f"{metric}_max"]
    with (out_dir / "bands.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=band_fields)
        writer.writeheader()
        for band in bands:
            writer.writerow({k: band.get(k, "") for k in band_fields})

    claim = check_bands(bands)
    (out_dir / "bands.json").write_text(
        json.dumps(
            {
                "ns": ns,
                "budget_factors": factors,
                "epsilons": epsilons,
                "seeds": seeds,
                "claim": claim,
                "bands": bands,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    log.info(
        "wrote %s results.csv, bands.csv, bands.json",
        out_dir,
    )

    _write_plots(rows, bands, ns, factors, epsilons, out_dir, log)

    print("\n=== Robustness band: spectral vs baselines (low-frequency distortion) ===")
    print(f"grid points: {claim['n_points']}  spectral<=all baselines: "
          f"{claim['n_below_all_baselines']}/{claim['n_points']} "
          f"({100 * claim['fraction_below_all_baselines']:.1f}%)")
    if not claim["holds_everywhere"]:
        print("  counterexamples (first 5):")
        fails = [p for p in claim["points"] if not p["below_all_baselines"]][:5]
        for p in fails:
            print(f"    n={p['n']} factor={p['budget_factor']} eps={p['epsilon']} "
                  f"spec={p['spectral_metric']:.3f} "
                  f"below={[k for k, v in p.items() if k.startswith('below_') and v is False]}")


if __name__ == "__main__":
    main()

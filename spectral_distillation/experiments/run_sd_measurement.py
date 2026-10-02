"""Measure real spectral distortion across the retention ladder.

The committed ladders were produced with ``--skip-sd``, so every
``sd_laplacian`` field is the placeholder ``0.0``. This script replaces those
placeholders with actual measurements.

It deliberately computes **only** the spectra. ``run_sparsify_ladder.py``
interleaves each point with 10x3 protocol cells (expert fitting plus mixture
accuracy), which costs hours per graph; the distortion of a sparsified graph
depends solely on the sparsifier output, so the protocol cells are pure waste
for this measurement.

Two metrics are reported for every point:

* ``legacy`` -- :func:`compute_spectral_distortion`, the max relative error over
  eigenvalue index ``i``. This saturates at exactly 1.0 once the sparsifier
  changes the component count, because index *i* then refers to unrelated modes
  in the two graphs. Reported only so the placeholder values can be shown to be
  meaningless, not because they should be quoted.
* ``rank_matched`` -- :func:`rank_matched_distortion`, which compares the largest
  eigenvalues instead. These are the modes a low-pass filter retains and they
  stay ordered under fragmentation.

The reference spectrum is computed once and reused, so each point costs a single
eigvalsh rather than two.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigvalsh

from spectral_distillation.src.baselines import degree_sparsify, random_sparsify
from spectral_distillation.src.distortion import (
    compute_spectral_distortion,
    rank_matched_distortion,
)
from spectral_distillation.src.effective_resistance import effective_resistance_exact
from spectral_distillation.src.laplacian import compute_laplacian
from spectral_distillation.src.real_fraud import load_real_fraud
from spectral_distillation.src.sparsifier import spectral_sparsify
from spectral_distillation.src.utils import ensure_dir, get_logger, set_seed

DEFAULT_BUDGETS = "0.6,0.3,0.15,0.08,0.04"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="yelpchi", choices=["amazon", "tolokers", "yelpchi"])
    p.add_argument("--data-path", default=None)
    p.add_argument("--max-nodes", type=int, default=None, help="yelpchi only")
    p.add_argument("--budgets", default=DEFAULT_BUDGETS)
    p.add_argument("--methods", default="er,random,degree")
    p.add_argument("--out", default="logs")
    args = p.parse_args()

    log = get_logger("sd_measurement")
    set_seed(0)
    budgets = [float(b) for b in args.budgets.split(",") if b.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    t0 = time.perf_counter()
    graph = load_real_fraud(args.dataset, args.data_path, max_nodes=args.max_nodes or None)
    W = graph["W"]
    n_edges_orig = int(W.sum() / 2)
    log.info("loaded %s n=%d edges=%d pos=%.4f in %.1fs", args.dataset,
             W.shape[0], n_edges_orig, graph["positive_ratio"],
             time.perf_counter() - t0)

    t0 = time.perf_counter()
    ref_evals = np.sort(eigvalsh(compute_laplacian(W)))
    log.info("reference spectrum cached (%d eigenvalues) in %.1fs", ref_evals.size,
             time.perf_counter() - t0)

    resistance = None
    if "er" in methods:
        t0 = time.perf_counter()
        resistance = effective_resistance_exact(compute_laplacian(W))
        log.info("effective resistance ready in %.1fs", time.perf_counter() - t0)

    rows = []
    for method in methods:
        for ret in budgets:
            t_pt = time.perf_counter()
            n_keep = max(1, int(round(n_edges_orig * ret)))
            if method == "er":
                # Same call the ladder uses, so these spectra are comparable
                # with the committed accuracy ladders.
                W_pt = spectral_sparsify(
                    W, n_keep, oracle_score=None, sample=True, seed=1,
                    resistance_matrix=resistance,
                )
            elif method == "random":
                W_pt = random_sparsify(W, n_keep, seed=1)
            elif method == "degree":
                W_pt = degree_sparsify(W, n_keep, seed=1)
            else:
                raise SystemExit(f"unknown method {method}")

            n_kept = int(W_pt.sum() / 2)
            deg = W_pt.sum(axis=1)
            n_isolated = int((deg == 0).sum())
            # Components decide whether the index-wise metric is meaningful.
            n_comp = int(_n_components(W_pt))
            t_spec = time.perf_counter()
            L_pt = compute_laplacian(W_pt)
            rm = rank_matched_distortion(None, L_pt, ref_eigenvalues=ref_evals)
            legacy = compute_spectral_distortion(
                compute_laplacian(W), L_pt, ref_eigenvalues=ref_evals
            )
            t_spec = time.perf_counter() - t_spec
            del L_pt

            row = {
                "method": method,
                "retention": ret,
                "n_edges_kept": n_kept,
                "fraction_kept": n_kept / n_edges_orig,
                "n_isolated_nodes": n_isolated,
                "isolated_fraction": n_isolated / W.shape[0],
                "n_components": n_comp,
                "legacy_SD_indexwise": legacy,
                **{f"rank_matched_{key}": v for key, v in rm.items()},
                "seconds_spectra": t_spec,
                "seconds_total": time.perf_counter() - t_pt,
            }
            rows.append(row)
            log.info(
                "%-6s r=%.2f edges=%7d isolated=%6.2f%% comps=%6d | legacy=%.4f "
                "lam_max_rel=%.4f top8=%.4f top32=%.4f (%.0fs)",
                method, ret, n_kept, 100 * row["isolated_fraction"], n_comp,
                legacy, rm["lambda_max_rel_error"], rm["top8_rel_error"],
                rm["top32_rel_error"], row["seconds_total"],
            )

    valid = [r for r in rows if r["n_components"] == 1]
    summary = {
        "config": {**vars(args), "budgets_list": budgets, "methods_list": methods},
        "graph": {
            "n_nodes": W.shape[0],
            "d_features": graph["d_features"],
            "n_edges": n_edges_orig,
            "positive_ratio": graph["positive_ratio"],
            "source": graph["source"],
        },
        "rows": rows,
        "n_points": len(rows),
        "n_points_connected": len(valid),
        "legacy_metric_saturated_count": sum(
            1 for r in rows if r["legacy_SD_indexwise"] >= 0.999
        ),
        "note": (
            "legacy_SD_indexwise is reported only to demonstrate that it saturates "
            "at ~1.0 whenever the sparsifier changes the component count. Quote "
            "rank_matched_* instead."
        ),
    }
    out_dir = ensure_dir(Path(args.out) / f"{args.dataset}_sd_measurement")
    (out_dir / "sd_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("wrote %s (%d points, %d saturated under the legacy metric)",
             out_dir / "sd_results.json", len(rows), summary["legacy_metric_saturated_count"])


def _n_components(W: np.ndarray) -> int:
    n = W.shape[0]
    seen = np.zeros(n, dtype=bool)
    comps = 0
    adj = [np.flatnonzero(W[i] > 0) for i in range(n)]
    for start in range(n):
        if seen[start]:
            continue
        comps += 1
        stack = [start]
        seen[start] = True
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
    return comps


if __name__ == "__main__":
    main()


"""Sweep --spectral-k for the D2 label-free router on real fraud graphs.

``spectral_neighbor_similarity`` averages inner products over the first
``k`` nontrivial Laplacian modes, so ``k`` trades a low-resolution, very
low-rank signal against a high-resolution one that is increasingly dominated by
noise. Every committed D2 result so far has silently used ``k=8``. This sweep
measures whether that default is actually a good choice per graph, instead of
assuming it.

Two efficiency facts make this cheap:

1. The Laplacian eigendecomposition does **not** depend on ``k``. It is computed
   once and reused for every ``k``, which is the whole cost of the sweep.
2. Under ``strategy="proxy_quantile"`` the assignment is a pure function of the
   ``eig_nb_sim`` score, so no KMeans and no RouterGNN training is needed. The
   D2 comparison is against a fixed ``random`` baseline that also does not depend
   on ``k``, so the oracle/random/uniform conditions are computed once.

The sweep also reports the Spearman correlation between the score and true label
homophily. Note that ``k`` can improve D2 while *lowering* that correlation --
YelpChi already shows this (rho=-0.26 while both routers deliver) -- so the
correlation column is diagnostic only and is never used to pick ``k``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from spectral_distillation.src.laplacian import compute_laplacian, eigh_symmetric
from spectral_distillation.src.router_protocol import (
    _buckets_from_score,
    label_homophily,
    oracle_bucket_assignment,
    random_assignment,
    spectral_neighbor_similarity,
)
from spectral_distillation.src.utils import device_config, ensure_dir, get_logger, set_seed

from spectral_distillation.experiments.run_real_fraud import cell_accuracy
from spectral_distillation.experiments.run_sparsify_ladder import _comparison_dict
from spectral_distillation.src.evaluation import combine_comparisons, run_protocol

DEFAULT_KS = "2,4,8,16,32,64,128"


def _parse_ks(text: str) -> list[int]:
    ks = sorted({int(tok) for tok in text.split(",") if tok.strip()})
    bad = [k for k in ks if k < 1]
    if bad:
        raise SystemExit(f"spectral-k must be >= 1, got {bad}")
    return ks


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="tolokers", choices=["amazon", "tolokers", "yelpchi"])
    p.add_argument("--data-path", default=None)
    p.add_argument("--max-nodes", type=int, default=None,
                   help="yelpchi only: cap the sampled node count")
    p.add_argument("--spectral-ks", default=DEFAULT_KS,
                   help="comma-separated mode counts to sweep")
    p.add_argument("--splits", type=int, default=10)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--top-k", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-adaptive-oracle", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--out", default="logs")
    p.add_argument("--device", default="cpu",
                   help="device for the eigendecomposition; 'cpu' is faster than "
                        "'cuda' at n>=14840 on this machine (360s vs 118s)")
    args = p.parse_args()

    log = get_logger("spectral_k_sweep")
    set_seed(args.seed)
    ks = _parse_ks(args.spectral_ks)
    out_dir = ensure_dir(Path(args.out) / f"{args.dataset}_spectral_k_sweep")

    # Imported here so the module docstring stays readable and the loader is
    # only touched when the sweep actually runs.
    from spectral_distillation.src.real_fraud import load_real_fraud

    t0 = time.perf_counter()
    graph = load_real_fraud(args.dataset, args.data_path, max_nodes=args.max_nodes or None)
    W, y = graph["W"], graph["y"]
    log.info("loaded %s n=%d d=%d edges=%d pos=%.4f in %.1fs", args.dataset,
             graph["n_nodes"], graph["d_features"], graph["n_edges"],
             graph["positive_ratio"], time.perf_counter() - t0)

    # --- k-independent setup, computed once -----------------------------------
    t0 = time.perf_counter()
    L = compute_laplacian(W)
    log.info("computing Laplacian eigendecomposition once (device=%s)...", args.device)
    evals, V = eigh_symmetric(L, device=device_config(args.device))
    log.info("eigendecomposition done in %.1fs (shared by all k)", time.perf_counter() - t0)

    t0 = time.perf_counter()
    oracle = oracle_bucket_assignment(W, y, adaptive=not args.no_adaptive_oracle, top_k=args.top_k)
    uniform = np.zeros((W.shape[0], 3))
    uniform[:, 0] = 1.0
    random = random_assignment(oracle, rng=1)
    fixed = {"oracle": oracle, "random": random, "uniform": uniform}
    for name, a in fixed.items():
        log.info("  %-10s usage=%s", name, np.bincount(np.argmax(a, axis=1), minlength=3).tolist())
    log.info("fixed conditions ready in %.1fs", time.perf_counter() - t0)

    true_h = label_homophily(W, y)
    usable = true_h[~np.isnan(true_h)]

    rows = []
    for k in ks:
        t_k = time.perf_counter()
        score = spectral_neighbor_similarity(W, V, k=k)
        assignment = _buckets_from_score(score, 3)
        build_s = time.perf_counter() - t_k

        rho = float(np.corrcoef(
            np.argsort(np.argsort(score[~np.isnan(true_h)])),
            np.argsort(np.argsort(usable)),
        )[0, 1])

        conds = dict(fixed)
        conds["label_free"] = assignment
        results = run_protocol(
            run_fn=lambda cond, split, seed, _c=conds: cell_accuracy(
                graph, _c, cond, split, seed
            ),
            n_splits=args.splits,
            n_seeds=args.seeds,
            progress=not args.no_progress,
            progress_desc=f"k={k}",
        )
        comparisons = {name: _comparison_dict(c) for name, c in combine_comparisons(results).items()}
        d2 = comparisons["label_free_vs_random"]
        rows.append({
            "spectral_k": k,
            "d2_mean_diff": d2["mean_diff"],
            "d2_p_wilcoxon": d2["p_wilcoxon"],
            "d2_effect_size_dz": d2["effect_size_dz"],
            "d2_significant": d2["significant"],
            "d1_mean_diff": comparisons["oracle_vs_uniform"]["mean_diff"],
            "spearman_score_vs_label_homophily": rho,
            "usage": np.bincount(np.argmax(assignment, axis=1), minlength=3).tolist(),
            "seconds": build_s,
        })
        log.info(
            "k=%-4d D2=%+.5f p=%.5f dz=%5.2f sig=%-5s rho=%+.4f usage=%s",
            k, d2["mean_diff"], d2["p_wilcoxon"], d2["effect_size_dz"],
            d2["significant"], rho, rows[-1]["usage"],
        )

    best = max(rows, key=lambda r: r["d2_mean_diff"])
    default_row = next((r for r in rows if r["spectral_k"] == 8), None)
    summary = {
        "config": {**vars(args), "spectral_ks": ks},
        "graph": {
            "n_nodes": graph["n_nodes"],
            "d_features": graph["d_features"],
            "n_edges": graph["n_edges"],
            "positive_ratio": graph["positive_ratio"],
            "source": graph["source"],
        },
        "rows": rows,
        "best_k_by_d2": best["spectral_k"],
        "best_d2_mean_diff": best["d2_mean_diff"],
        "default_k8_d2_mean_diff": default_row["d2_mean_diff"] if default_row else None,
        "default_k8_is_best": bool(default_row and default_row["spectral_k"] == best["spectral_k"]),
    }
    (out_dir / "sweep.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info(
        "best k=%d (D2 %+.5f); k=8 gave %s -> wrote %s",
        best["spectral_k"], best["d2_mean_diff"],
        f"{default_row['d2_mean_diff']:+.5f}" if default_row else "n/a",
        out_dir / "sweep.json",
    )


if __name__ == "__main__":
    main()

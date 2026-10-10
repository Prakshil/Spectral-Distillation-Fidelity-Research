"""Held-out (out-of-fold) confirmation of the spectral-k router.

``experiments/run_spectral_k_sweep.py`` reported that ``eig_nb_sim`` prefers
``k=128`` on Tolokers and YelpChi, roughly doubling the D2 gain over the never-
tuned default ``k=8``. That sweep *selected* ``k`` on exactly the same 10x3
protocol cells it used to *report* the winner, so the winning ``k`` is a
reporting-selected candidate, not a validated one -- the caveat recorded in
``docs/reproducibility.md``.

This runner separates selection from confirmation without any new data. For every
held-out split ``h`` it picks the winner ``k*(h)`` using only the *other* splits
(across all seeds), then scores ``D2(k*(h))`` on the untouched split ``h``. The
average of those held-out scores is an honest estimate of the D2 you would get
by *choosing* ``k`` this way and deploying it, i.e. of the selection procedure
itself rather than of a hand-picked winner. If ``k=128`` is real, the selector
recovers it on nearly every fold and the held-out D2 matches the in-sample
number; if it is a selection artifact, the held-out D2 collapses toward the
default ``k=8`` or toward zero.

Both protocols are run and reported:

* **self-routing** -- the protocol of the original k-sweep claim (each expert is
  fitted to and scored on its own routing). Inflated, kept for continuity.
* **fixed-expert (Protocol B)** -- the honest protocol: one routing-blind
  k-means expert pool, routing is the only free variable, and ``D2`` is reported
  both against ``random`` and against a genuine no-routing global model.

Artifacts: ``logs/{dataset}_k_confirmation/confirmation.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from spectral_distillation.src.laplacian import compute_laplacian, eigh_symmetric
from spectral_distillation.src.evaluation import evaluate_comparison
from spectral_distillation.src.real_fraud import load_real_fraud
from spectral_distillation.src.router_protocol import (
    _buckets_from_score,
    label_homophily,
    oracle_bucket_assignment,
    random_assignment,
    spectral_neighbor_similarity,
)
from spectral_distillation.src.utils import device_config, ensure_dir, get_logger, set_seed

from spectral_distillation.experiments.run_fixed_expert_protocol import (
    N_EXPERTS,
    build_expert_factory,
    evaluate_all,
    feature_kmeans_routing,
    single_global_condition,
)

DEFAULT_KS = "2,4,8,16,32,64,128"


def _parse_ks(text: str) -> list[int]:
    ks = sorted({int(tok) for tok in text.split(",") if tok.strip()})
    if any(k < 1 for k in ks):
        raise SystemExit(f"spectral-k must be >= 1, got {ks}")
    return ks


def _loo_select(
    acc: dict[str, dict[int, float]], ks: list[int], splits: list[int], k_key
) -> tuple[dict, dict]:
    """Leave-one-split-out selection of ``k`` for one protocol.

    Returns ``(d2, summary)`` where ``d2[k][split]`` is the per-split D2 of
    candidate ``k`` and ``summary`` describes the held-out behaviour of the
    selection procedure.
    """
    d2 = {
        k: {s: float(k_key(acc, k, s)) for s in splits}
        for k in ks
    }
    folds = {}
    for h in splits:
        others = [s for s in splits if s != h]
        means = {k: float(np.mean([d2[k][s] for s in others])) for k in ks}
        k_star = max(ks, key=lambda k: means[k])
        folds[h] = {"k_star": k_star, "held_out_d2": d2[k_star][h]}
    held = np.array([f["held_out_d2"] for f in folds.values()])
    sel_counts = {str(k): sum(1 for f in folds.values() if f["k_star"] == k) for k in ks}
    return d2, {
        "n_folds": len(splits),
        "selection_counts": sel_counts,
        "held_out_d2_mean": float(held.mean()),
        "held_out_d2_std": float(held.std(ddof=1)) if held.size > 1 else 0.0,
        "held_out_d2_per_fold": {str(h): f["held_out_d2"] for h, f in folds.items()},
        "held_out_k_star_per_fold": {str(h): f["k_star"] for h, f in folds.items()},
    }


def _full_grid(
    acc: dict[str, dict[int, float]], ks: list[int], splits: list[int], name
) -> dict:
    """In-sample D2 of every ``k`` against ``random`` (paired over splits)."""
    rows = {}
    for k in ks:
        a = {s: acc[f"k{k}"][s] for s in splits}
        b = {s: acc["random"][s] for s in splits}
        c = evaluate_comparison(a, b, f"k{k}_vs_random")
        rows[str(k)] = {
            "d2": c.mean_diff,
            "p": c.p_wilcoxon,
            "dz": c.effect_size_dz,
        }
    best = max(ks, key=lambda k: rows[str(k)]["d2"])
    return {"per_k": rows, "best_k_by_d2": best,
            "best_d2": rows[str(best)]["d2"], "name": name}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="tolokers", choices=["amazon", "tolokers", "yelpchi"])
    p.add_argument("--data-path", default=None)
    p.add_argument("--max-nodes", type=int, default=None)
    p.add_argument("--spectral-ks", default=DEFAULT_KS)
    p.add_argument("--reported-k", type=int, default=128,
                   help="the k the original sweep reported as best (to check)")
    p.add_argument("--default-k", type=int, default=8,
                   help="the never-tuned default the sweep argued against")
    p.add_argument("--splits", type=int, default=10)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--top-k", type=int, default=32)
    p.add_argument("--no-adaptive-oracle", action="store_true")
    p.add_argument("--expert", default="logistic", choices=["logistic", "mlp"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--out", default="logs")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    log = get_logger("k_confirmation")
    set_seed(args.seed)
    ks = _parse_ks(args.spectral_ks)
    out_dir = ensure_dir(Path(args.out) / f"{args.dataset}_k_confirmation")

    t0 = time.perf_counter()
    graph = load_real_fraud(args.dataset, args.data_path, max_nodes=args.max_nodes or None)
    W, X, y = graph["W"], graph["features"], graph["y"]
    log.info("loaded %s n=%d d=%d edges=%d pos=%.4f in %.1fs", args.dataset,
             graph["n_nodes"], graph["d_features"], graph["n_edges"],
             graph["positive_ratio"], time.perf_counter() - t0)

    t0 = time.perf_counter()
    L = compute_laplacian(W)
    log.info("Laplacian eigendecomposition (device=%s)...", args.device)
    evals, V = eigh_symmetric(L, device=device_config(args.device))
    log.info("eigendecomposition done in %.1fs (shared by all k)", time.perf_counter() - t0)

    true_h = label_homophily(W, y)
    us = ~np.isnan(true_h)
    rho = {
        k: float(np.corrcoef(
            np.argsort(np.argsort(spectral_neighbor_similarity(W, V, k=k)[us])),
            np.argsort(np.argsort(true_h[us])),
        )[0, 1])
        for k in ks
    }

    assignments: dict[str, np.ndarray] = {
        f"k{k}": _buckets_from_score(spectral_neighbor_similarity(W, V, k=k), N_EXPERTS)
        for k in ks
    }
    oracle = oracle_bucket_assignment(W, y, adaptive=not args.no_adaptive_oracle,
                                      top_k=args.top_k)
    km = feature_kmeans_routing(X, N_EXPERTS, seed=args.seed)
    conditions = dict(assignments)
    conditions.update({
        "single_global": single_global_condition(W.shape[0]),
        "oracle": oracle,
        "feature_kmeans": km,
        "random": random_assignment(oracle, rng=args.seed + 1),
    })

    factory = build_expert_factory(args.expert, graph, args.device)
    splits = list(range(args.splits))

    log.info("protocol A (self-routing): %d conditions x %d splits x %d seeds",
             len(conditions), args.splits, args.seeds)
    a_self = evaluate_all(conditions, X, y, args.splits, args.seeds, log,
                          expert_factory=factory)

    log.info("protocol B (fixed k-means pool): %d conditions x %d splits x %d seeds",
             len(conditions), args.splits, args.seeds)
    a_fixed = evaluate_all(conditions, X, y, args.splits, args.seeds, log,
                           expert_assignment=km, expert_factory=factory)

    report: dict = {
        "config": {**vars(args), "spectral_ks": ks, "n_experts": N_EXPERTS},
        "graph": {
            "n_nodes": graph["n_nodes"], "d_features": graph["d_features"],
            "n_edges": graph["n_edges"], "positive_ratio": graph["positive_ratio"],
            "source": graph["source"],
        },
        "spearman_score_vs_label_homophily": rho,
        "protocols": {},
    }

    def run_protocol(acc: dict, key: str, label: str) -> dict:
        # D2(k) against the usage-matched random reference for this protocol.
        d2, loo = _loo_select(
            acc, ks, splits, lambda a, k, s: a[f"k{k}"][s] - a["random"][s]
        )
        full = _full_grid(acc, ks, splits, label)
        # D2 of the reported/default k using the whole grid (in-sample reference).
        in_sample = {
            "reported_k": {
                "k": args.reported_k,
                "d2": full["per_k"][str(args.reported_k)]["d2"] if str(args.reported_k) in full["per_k"] else None,
            },
            "default_k": {
                "k": args.default_k,
                "d2": full["per_k"][str(args.default_k)]["d2"] if str(args.default_k) in full["per_k"] else None,
            },
            "oracle_best_k": {"k": full["best_k_by_d2"], "d2": full["best_d2"]},
        }
        # For the fixed protocol also report the honest no-routing reference.
        if key == "fixed":
            for k in ks:
                a = {s: acc[f"k{k}"][s] for s in splits}
                g = {s: acc["single_global"][s] for s in splits}
                c = evaluate_comparison(a, g, f"k{k}_vs_global")
                full["per_k"][str(k)].update({
                    "d2_vs_no_routing": c.mean_diff,
                    "p_vs_no_routing": c.p_wilcoxon,
                    "dz_vs_no_routing": c.effect_size_dz,
                })
            glob = evaluate_comparison(
                {s: acc["single_global"][s] for s in splits},
                {s: acc["random"][s] for s in splits}, "global_vs_random")
            full["single_global_d2"] = glob.mean_diff
        return {"in_sample_full_grid": full, "leave_one_split_out": loo,
                "in_sample_anchors": in_sample}

    report["protocols"]["self_routing"] = run_protocol(a_self, "self", "self-routing")
    report["protocols"]["fixed_expert"] = run_protocol(a_fixed, "fixed", "fixed-expert")

    (out_dir / "confirmation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ #
    # Console summary
    # ------------------------------------------------------------------ #
    print("\n=== out-of-fold k-selection confirmation: %s (expert=%s) ===" %
          (args.dataset, args.expert))
    for key, label in (("self_routing", "SELF-ROUTING (inflated)"),
                       ("fixed_expert", "FIXED-EXPERT (honest)")):
        r = report["protocols"][key]
        full = r["in_sample_full_grid"]
        loo = r["leave_one_split_out"]
        print(f"\n-- {label} --")
        print("{:>5}{:>13}{:>11}{:>8}{:>22}".format("k", "D2(insample)", "p", "dz", "D2_vs_no_routing"))
        for k in ks:
            row = full["per_k"][str(k)]
            vnr = row.get("d2_vs_no_routing")
            print("{:>5}{:>13.5f}{:>11.5f}{:>8.2f}{:>22}".format(
                k, row["d2"], row["p"], row["dz"],
                f"{vnr:+.5f}" if vnr is not None else "-"))
        def _d2(k):
            row = full["per_k"].get(str(k))
            return f"{row['d2']:+.5f}" if row else "n/a"
        print("in-sample best k = {} (D2 {:+.5f}); reported k={} D2 {}; default k={} D2 {}".format(
            full["best_k_by_d2"], full["best_d2"],
            args.reported_k, _d2(args.reported_k),
            args.default_k, _d2(args.default_k)))
        modal_k = max(loo["selection_counts"], key=loo["selection_counts"].get)
        print("out-of-fold selector: modal k={} on {}/{} folds; "
              "reported k={} chosen {}/{} folds; held-out D2 = {:+.5f} (sd {:.5f})".format(
                  modal_k, loo["selection_counts"][modal_k], loo["n_folds"],
                  args.reported_k, loo["selection_counts"].get(str(args.reported_k), 0),
                  loo["n_folds"],
                  loo["held_out_d2_mean"], loo["held_out_d2_std"]))

    print("\nwrote", out_dir / "confirmation.json")


if __name__ == "__main__":
    main()

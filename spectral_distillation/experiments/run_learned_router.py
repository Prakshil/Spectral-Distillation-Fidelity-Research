"""Learned-router experiment: does RouterGNN-lite beat no routing on real data?

Closes the one gap in the 0/234 result. Every routing scored in the 3x3x26 sweep
is fixed and label-free; this script adds a routing that is *trained* from
training labels, and scores it on the same frozen expert pool and the same splits
as every baseline, so the comparison is like-for-like.

Writes ``logs/<dataset>_learned_router/fixed_expert.json``. The synthetic
positive control (``run_positive_control.py``) is the counterpart that shows the
router can find real structure when structure exists.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from spectral_distillation.experiments.run_fixed_expert_protocol import (
    build_expert_factory,
    feature_kmeans_routing,
)
from spectral_distillation.src.learned_router import evaluate_learned_routing
from spectral_distillation.src.laplacian import normalize_adjacency
from spectral_distillation.src.mixture import train_test_split
from spectral_distillation.src.real_fraud import load_real_fraud
from spectral_distillation.src.router_protocol import oracle_bucket_assignment
from spectral_distillation.src.utils import ensure_dir, get_logger, set_seed


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="amazon", choices=["amazon", "tolokers", "yelpchi"])
    p.add_argument("--data-path", default=None)
    p.add_argument("--max-nodes", type=int, default=None)
    p.add_argument("--splits", type=int, default=3)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--router-frac", type=float, default=0.35)
    p.add_argument("--expert", default="logistic", choices=["logistic", "mlp", "gnn"])
    p.add_argument("--router-mode", default="nodemoe",
                   choices=["nodemoe", "x_only"],
                   help="router gate features. 'nodemoe' is [X, |AX-X|, |A2X-X|]; "
                        "'x_only' removes every graph term, which with a per-row "
                        "expert gives the implicit-routing control arm (EMNLP 2023)")
    p.add_argument("--out", default="logs")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    if args.router_mode == "x_only" and args.expert == "gnn":
        p.error(
            "--router-mode x_only with --expert gnn is not a clean control: the "
            "graph expert still consumes AX internally, so implicit routing would "
            "remain possible and a null result would be uninterpretable. Use "
            "--expert logistic or mlp for this arm."
        )

    log = get_logger("learned_router")
    set_seed(0)
    suffix = "" if args.expert == "logistic" else f"_{args.expert}"
    if args.router_mode != "nodemoe":
        suffix += f"_{args.router_mode}"
    out_dir = ensure_dir(Path(args.out) / f"{args.dataset}_learned_router{suffix}")

    t0 = time.perf_counter()
    graph = load_real_fraud(args.dataset, args.data_path, max_nodes=args.max_nodes or None)
    W, X, y = graph["W"], graph["features"], graph["y"]
    log.info("loaded %s n=%d (%.1fs)", args.dataset, W.shape[0], time.perf_counter() - t0)

    W_norm = normalize_adjacency(W)
    pool = feature_kmeans_routing(X)
    oracle = oracle_bucket_assignment(W, y, adaptive=True)
    factory = build_expert_factory(args.expert, graph, args.device)

    per_split = []
    t1 = time.perf_counter()
    for split in range(args.splits):
        for seed in range(args.seeds):
            train, test = train_test_split(y, split, seed)
            res = evaluate_learned_routing(
                X, y, W_norm, pool, train, test, oracle, expert_factory=factory,
                seed=seed, router_frac=args.router_frac, epochs=args.epochs,
                device=args.device, feature_mode=args.router_mode,
            )
            res["split"], res["seed"] = split, seed
            per_split.append(res)
            a = res["accuracy"]
            log.info(
                "split %d/%d seed %d: global=%.4f learned=%.4f (%+.4f) kmeans=%.4f "
                "random=%.4f oracle=%.4f agree_kmeans=%.3f",
                split + 1, args.splits, seed, a["single_global"], a["learned_router"],
                res["learned_minus_no_routing"], a["feature_kmeans"], a["random"],
                a["oracle"], res["router_vs_kmeans_agreement"],
            )

    keys = list(per_split[0]["accuracy"])
    mean_acc = {k: float(np.mean([r["accuracy"][k] for r in per_split])) for k in keys}
    n = len(per_split)
    wins = int(sum(r["learned_minus_no_routing"] > 0 for r in per_split))
    ens_wins = int(sum(r.get("learned_minus_ensemble", -1) > 0 for r in per_split))

    payload = {
        "config": vars(args),
        "graph": {
            "n_nodes": int(X.shape[0]),
            "d_features": int(X.shape[1]),
            "n_edges": float(W.sum() / 2.0),
            "positive_ratio": float(y.mean()),
        },
        "results": {
            "mean_accuracy": mean_acc,
            "learned_minus_no_routing": mean_acc["learned_router"] - mean_acc["single_global"],
            "learned_minus_random": mean_acc["learned_router"] - mean_acc["random"],
            "learned_minus_ensemble": mean_acc.get("learned_router", 0.0) - mean_acc.get("ensemble_uniform", mean_acc.get("single_global", 0.0)),
            "ensemble_minus_no_routing": mean_acc.get("ensemble_uniform", 0.0) - mean_acc["single_global"],
            "router_vs_kmeans_agreement": float(
                np.mean([r["router_vs_kmeans_agreement"] for r in per_split])
            ),
            "n_runs": n,
            "n_runs_learned_beats_no_routing": wins,
            "n_runs_learned_beats_ensemble": ens_wins,
            "per_run": per_split,
        },
        "interpretation": (
            "Learned routing is scored on the same frozen routing-blind pool as "
            "every baseline, trained on training nodes only. Added Node-MoE gate "
            "[X, |AX-X|, |A2X-X|] and uniform ensemble (Ens-Avg). "
            + ("Router mode 'x_only' removes all graph propagation from both the "
               "gate and the expert, so implicit routing through a frozen expert "
               "(EMNLP 2023) cannot explain a null result."
               if args.router_mode == "x_only" else "")
        ),
    }
    path = out_dir / "fixed_expert.json"
    with open(path, "w", encoding="utf8") as fh:
        json.dump(payload, fh, indent=2)
    log.info(
        "learned=%.4f global=%.4f (%+.4f) | learned won %d/%d runs",
        mean_acc["learned_router"], mean_acc["single_global"],
        mean_acc["learned_router"] - mean_acc["single_global"], wins, n,
    )
    log.info("wrote %s", path)


if __name__ == "__main__":
    main()
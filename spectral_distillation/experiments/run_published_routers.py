"""Score the published routers (RouterGNN, AD-GNN) beside every baseline.

Closes the largest remaining reviewer gap: the negative routing result is scored
against hand-designed scores and an MLP gate, never against the published
*learned* routers. This runs both, on the same frozen routing-blind expert pool
and the same splits as every baseline, so the comparison is like-for-like.

Datasets: the fraud graphs (Amazon, Tolokers, capped YelpChi) and the Node-MoE
heterophily suite (Actor, Chameleon, Squirrel).

Writes ``logs/<dataset>_published_routers/fixed_expert.json``.
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
from spectral_distillation.src.laplacian import normalize_adjacency
from spectral_distillation.src.mixture import train_test_split
from spectral_distillation.src.published_routers import evaluate_published_routers
from spectral_distillation.src.router_protocol import oracle_bucket_assignment
from spectral_distillation.src.utils import ensure_dir, get_logger, set_seed

FRAUD_DATASETS = ("amazon", "tolokers", "yelpchi")
HETEROPHILY_DATASETS = ("actor", "chameleon", "squirrel")


def load_graph(dataset: str, data_path: str | None, max_nodes: int | None) -> dict:
    """Dispatch to the fraud or heterophily loader for ``dataset``."""
    if dataset in HETEROPHILY_DATASETS:
        if max_nodes is not None:
            raise ValueError(
                f"{dataset} is dense-feasible; max_nodes is not supported here."
            )
        from spectral_distillation.src.heterophily import load_heterophily

        return load_heterophily(dataset)
    from spectral_distillation.src.real_fraud import load_real_fraud

    return load_real_fraud(dataset, data_path, max_nodes=max_nodes or None)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset", default="amazon",
        choices=list(FRAUD_DATASETS) + list(HETEROPHILY_DATASETS),
    )
    p.add_argument("--data-path", default=None)
    p.add_argument("--max-nodes", type=int, default=None)
    p.add_argument("--splits", type=int, default=10)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--router-frac", type=float, default=0.35)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--expert", default="logistic", choices=["logistic", "mlp"])
    p.add_argument("--out", default="logs")
    p.add_argument("--device", default="cpu")
    p.add_argument("--gnn-lr", type=float, default=1e-3)
    p.add_argument("--gnn-weight-decay", type=float, default=1e-4)
    p.add_argument("--gnn-grad-clip", type=float, default=1.0)
    p.add_argument(
        "--no-gnn-layer-norm", action="store_true",
        help="train the raw unnormalized RouterGNN stack (diverges on large graphs)",
    )
    args = p.parse_args()

    log = get_logger("published_routers")
    set_seed(0)
    suffix = "" if args.expert == "logistic" else f"_{args.expert}"
    out_dir = ensure_dir(Path(args.out) / f"{args.dataset}_published_routers{suffix}")

    t0 = time.perf_counter()
    graph = load_graph(args.dataset, args.data_path, args.max_nodes)
    W, X, y = graph["W"], graph["features"], graph["y"]
    log.info("loaded %s n=%d d=%d (%.1fs)", args.dataset, X.shape[0], X.shape[1],
             time.perf_counter() - t0)

    W_norm = normalize_adjacency(W)
    pool = feature_kmeans_routing(X)
    oracle = oracle_bucket_assignment(W, y, adaptive=True)
    factory = build_expert_factory(args.expert, graph, args.device)

    per_split = []
    for split in range(args.splits):
        for seed in range(args.seeds):
            train, test = train_test_split(y, split, seed)
            res = evaluate_published_routers(
                X, y, W_norm, pool, train, test, oracle, expert_factory=factory,
                seed=seed, router_frac=args.router_frac, epochs=args.epochs,
                hidden_dim=args.hidden_dim, n_layers=args.n_layers,
                device=args.device, gnn_lr=args.gnn_lr,
                gnn_weight_decay=args.gnn_weight_decay,
                gnn_grad_clip=args.gnn_grad_clip,
                gnn_layer_norm=not args.no_gnn_layer_norm,
            )
            res["split"], res["seed"] = split, seed
            per_split.append(res)
            a = res["accuracy"]
            log.info(
                "split %d/%d seed %d: global=%.4f mlp=%.4f (%+.4f) gnn=%.4f (%+.4f) "
                "adagnn=%.4f (%+.4f) ens=%.4f kmeans=%.4f oracle=%.4f | "
                "gnn loss %.3f->%.3f fitacc=%.2f ch=%d | mlp loss %.3f->%.3f fitacc=%.2f",
                split + 1, args.splits, seed, a["single_global"], a["mlp_gate"],
                res["mlp_gate_minus_no_routing"], a["router_gnn"],
                res["router_gnn_minus_no_routing"], a["ad_gnn"],
                res["ad_gnn_minus_no_routing"], a["ensemble_uniform"],
                a["feature_kmeans"], a["oracle"],
                res["gnn_loss_first"], res["gnn_loss_last"],
                res["gnn_fit_accuracy"], res["gnn_channels_used"],
                res["mlp_loss_first"], res["mlp_loss_last"],
                res["mlp_fit_accuracy"],
            )

    keys = list(per_split[0]["accuracy"])
    mean_acc = {k: float(np.mean([r["accuracy"][k] for r in per_split])) for k in keys}
    n = len(per_split)

    def wins(field: str) -> int:
        return int(sum(r[field] > 0 for r in per_split))

    payload = {
        "config": vars(args),
        "graph": {
            "n_nodes": int(X.shape[0]),
            "d_features": int(X.shape[1]),
            "n_edges": float(W.sum() / 2.0),
            "positive_ratio": (float(y.mean()) if y.max() <= 1 else None),
            "n_classes": int(graph.get("n_classes", int(np.max(y)) + 1)),
        },
        "results": {
            "mean_accuracy": mean_acc,
            "router_gnn_minus_no_routing": mean_acc["router_gnn"] - mean_acc["single_global"],
            "ad_gnn_minus_no_routing": mean_acc["ad_gnn"] - mean_acc["single_global"],
            "mlp_gate_minus_no_routing": mean_acc["mlp_gate"] - mean_acc["single_global"],
            "ensemble_minus_no_routing": mean_acc["ensemble_uniform"] - mean_acc["single_global"],
            "router_gnn_vs_kmeans_agreement": float(
                np.mean([r["router_gnn_vs_kmeans_agreement"] for r in per_split])
            ),
            "ad_gnn_vs_kmeans_agreement": float(
                np.mean([r["ad_gnn_vs_kmeans_agreement"] for r in per_split])
            ),
            "gnn_loss_first": float(np.mean([r["gnn_loss_first"] for r in per_split])),
            "gnn_loss_last": float(np.mean([r["gnn_loss_last"] for r in per_split])),
            "gnn_fit_accuracy": float(np.mean([r["gnn_fit_accuracy"] for r in per_split])),
            "gnn_channels_used": int(max(r["gnn_channels_used"] for r in per_split)),
            "mlp_loss_first": float(np.mean([r["mlp_loss_first"] for r in per_split])),
            "mlp_loss_last": float(np.mean([r["mlp_loss_last"] for r in per_split])),
            "mlp_fit_accuracy": float(np.mean([r["mlp_fit_accuracy"] for r in per_split])),
            "n_runs": n,
            "n_runs_router_gnn_beats_no_routing": wins("router_gnn_minus_no_routing"),
            "n_runs_ad_gnn_beats_no_routing": wins("ad_gnn_minus_no_routing"),
            "n_runs_mlp_gate_beats_no_routing": wins("mlp_gate_minus_no_routing"),
            "per_run": per_split,
        },
        "interpretation": (
            "Published routers (RouterGNN message-passing, AD-GNN per-node depth via "
            "signal preservation) scored on the same frozen routing-blind k-means "
            "pool and splits as every baseline. RouterGNN shares the MLP gate's "
            "cross-fit supervision; AD-GNN is label-free. The message-passing router "
            "is trained with LayerNorm, weight decay and gradient clipping (the raw "
            "unnormalized stack diverges on large graphs); loss first->last, router "
            "fit accuracy and distinct channels emitted are recorded per run so a "
            "collapsed router is never silently reported as a trained one."
        ),
    }
    path = out_dir / "fixed_expert.json"
    with open(path, "w", encoding="utf8") as fh:
        json.dump(payload, fh, indent=2)
    log.info(
        "RouterGNN %+.4f (%d/%d wins) | AD-GNN %+.4f (%d/%d) | MLP gate %+.4f (%d/%d)",
        payload["results"]["router_gnn_minus_no_routing"],
        payload["results"]["n_runs_router_gnn_beats_no_routing"], n,
        payload["results"]["ad_gnn_minus_no_routing"],
        payload["results"]["n_runs_ad_gnn_beats_no_routing"], n,
        payload["results"]["mlp_gate_minus_no_routing"],
        payload["results"]["n_runs_mlp_gate_beats_no_routing"], n,
    )
    log.info("wrote %s", path)


if __name__ == "__main__":
    main()

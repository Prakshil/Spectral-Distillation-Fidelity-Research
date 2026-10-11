"""Learnability lever: where does routing start to pay for itself?

Protocol B reports a null on every real pool: no routing beats one global model.
This script locates the boundary of that null by sweeping the planted
``separability`` lever of :mod:`spectral_distillation.src.learnability_lever`
from 0 (one rule fits every group -- routing is worth nothing) to 1 (the positive
control -- routing is worth a lot). For each setting it scores the same arms the
paper uses: no-routing, oracle, feature-k-means, the learned MLP gate, and the
*learnability-conditioned* selective router that abstains below the boundary.

Two things fall out:

* a threshold ``separability*`` above which routing becomes learnable, so the
  real-data null is a statement about the task, not a broken harness;
* a selective router that, evaluated on real pools, correctly abstains -- "no
  win" becomes "correctly declined".

Run::

    python -m spectral_distillation.experiments.run_learnability_lever \
        --separabilities 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
        --splits 3 --expert logistic --out logs
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from spectral_distillation.src.learnability_lever import (
    learnability_conditioned_router,
    planted_separability_graph,
)
from spectral_distillation.src.laplacian import normalize_adjacency
from spectral_distillation.src.learned_router import evaluate_learned_routing
from spectral_distillation.src.mixture import EXPERT_FACTORIES, mixture_accuracy, train_test_split
from spectral_distillation.src.positive_control import oracle_from_groups
from spectral_distillation.src.real_fraud import load_real_fraud
from spectral_distillation.src.router_protocol import oracle_bucket_assignment
from spectral_distillation.src.utils import ensure_dir, get_logger, set_seed

N_EXPERTS = 3
EXPERT_NAMES = ("logistic", "mlp")
ARMS = ("single_global", "oracle", "feature_kmeans", "learned_router", "ensemble_uniform")


def feature_kmeans_pool(X: np.ndarray, seed: int = 0) -> np.ndarray:
    """Routing-blind pool: k-means on features, matching Protocol B exactly."""
    km = KMeans(n_clusters=N_EXPERTS, n_init=10, random_state=seed)
    lab = km.fit_predict(X)
    out = np.zeros((X.shape[0], N_EXPERTS))
    out[np.arange(X.shape[0]), lab] = 1.0
    return out


def _conditioned_accuracy(res: dict, X, y, te) -> float:
    heads = res["global_experts"] if res["decision"] == "abstain" else res["experts"]
    acc, _ = mixture_accuracy(X, y, res["assignment"], heads, te)
    return float(acc)


def run_cell(
    W_norm: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    pool: np.ndarray,
    oracle: np.ndarray,
    splits: int,
    seeds: int,
    expert_factory,
    epochs: int,
    device: str,
) -> dict:
    """Score every arm + the conditioned router over (split, seed) cells."""
    per_arm: dict[str, list[float]] = {a: [] for a in ARMS}
    cond_acc: list[float] = []
    cond_gain: list[float] = []
    cond_decision: list[str] = []
    cond_val_gain: list[float] = []

    for split in range(splits):
        for seed in range(seeds):
            train, te = train_test_split(y, split, seed)
            ev = evaluate_learned_routing(
                X, y, W_norm, pool, train, te, oracle,
                expert_factory, seed=seed, epochs=epochs, device=device,
            )
            for arm in ARMS:
                per_arm[arm].append(float(ev["accuracy"][arm]))
            cond = learnability_conditioned_router(
                X, y, W_norm, pool, train, expert_factory,
                seed=seed, epochs=epochs, device=device,
            )
            c_acc = _conditioned_accuracy(cond, X, y, te)
            cond_acc.append(c_acc)
            cond_gain.append(c_acc - float(ev["accuracy"]["single_global"]))
            cond_decision.append(cond["decision"])
            cond_val_gain.append(float(cond["validation_gain"]))

    acc = {a: float(np.mean(v)) for a, v in per_arm.items()}
    return {
        "arms": acc,
        "gain_oracle": acc["oracle"] - acc["single_global"],
        "gain_kmeans": acc["feature_kmeans"] - acc["single_global"],
        "gain_learned": acc["learned_router"] - acc["single_global"],
        "conditioned_accuracy": float(np.mean(cond_acc)),
        "conditioned_gain": float(np.mean(cond_gain)),
        "conditioned_validation_gain": float(np.mean(cond_val_gain)),
        "conditioned_route_rate": float(np.mean([d == "route" for d in cond_decision])),
        "conditioned_decisions": cond_decision,
    }


def _score_graph(graph: dict, args, expert_factory) -> dict:
    """Pool + oracle + every arm for one planted graph."""
    W, X, y = graph["W"], graph["features"], graph["y"]
    W_norm = normalize_adjacency(W)
    pool = feature_kmeans_pool(X, seed=args.seed)
    oracle = oracle_from_groups(graph["group"], pool)
    cell = run_cell(W_norm, X, y, pool, oracle, args.splits, args.seeds,
                    expert_factory, args.epochs, args.device)
    cell["n_nodes"] = int(W.shape[0])
    cell["class_balance"] = float(y.mean())
    return cell


def sweep(args, log, expert_factory) -> list[dict]:
    rows = []
    seps = [float(s) for s in args.separabilities.split(",") if s != ""]
    for beta in seps:
        t0 = time.perf_counter()
        graph = planted_separability_graph(
            n=args.n, n_features=args.n_features, seed=args.seed,
            edge_p=args.edge_p, cross_edge_p=args.cross_edge_p, separability=beta,
        )
        cell = _score_graph(graph, args, expert_factory)
        cell["separability"] = beta
        rows.append(cell)
        log.info(
            "beta=%.2f  global=%.4f oracle=%+.4f kmeans=%+.4f learned=%+.4f "
            "cond=%s (gain %+.4f)  %.0fs",
            beta, cell["arms"]["single_global"], cell["gain_oracle"],
            cell["gain_kmeans"], cell["gain_learned"],
            "route" if cell["conditioned_route_rate"] >= 0.5 else "abstain",
            cell["conditioned_gain"], time.perf_counter() - t0,
        )
    return rows


def sweep_homophily(args, log, expert_factory) -> list[dict]:
    """Secondary lever: dilute graph homophily at fixed label separability.

    ``cross_edge_p`` adds random cross-group edges on top of the block-diagonal
    graph. Because the expert pool (feature k-means) and the oracle (latent
    groups) both ignore edges, a flat curve means the routing signal is carried
    by features, not edges; a falling curve means routing needs the graph.
    """
    rows = []
    values = [float(s) for s in args.homophily_cross_edges.split(",") if s != ""]
    for cross in values:
        t0 = time.perf_counter()
        graph = planted_separability_graph(
            n=args.n, n_features=args.n_features, seed=args.seed,
            edge_p=args.edge_p, cross_edge_p=cross,
            separability=args.homophily_separability,
        )
        cell = _score_graph(graph, args, expert_factory)
        cell["cross_edge_p"] = cross
        cell["separability"] = args.homophily_separability
        rows.append(cell)
        log.info(
            "cross_edge_p=%.3f  oracle=%+.4f kmeans=%+.4f learned=%+.4f "
            "cond=%s (gain %+.4f)  %.0fs",
            cross, cell["gain_oracle"], cell["gain_kmeans"], cell["gain_learned"],
            "route" if cell["conditioned_route_rate"] >= 0.5 else "abstain",
            cell["conditioned_gain"], time.perf_counter() - t0,
        )
    return rows


def probe_real(args, log, expert_factory) -> dict:
    out = {}
    for dataset in args.real:
        t0 = time.perf_counter()
        graph = load_real_fraud(dataset)
        W, X, y = graph["W"], graph["features"], graph["y"]
        W_norm = normalize_adjacency(W)
        pool = feature_kmeans_pool(X, seed=args.seed)
        oracle = oracle_bucket_assignment(W, y, adaptive=True)
        cell = run_cell(W_norm, X, y, pool, oracle, args.splits, args.seeds,
                        expert_factory, args.epochs, args.device)
        cell["n_nodes"] = int(W.shape[0])
        cell["class_balance"] = float(y.mean())
        out[dataset] = cell
        log.info(
            "real %-8s n=%d  global=%.4f oracle=%+.4f kmeans=%+.4f learned=%+.4f "
            "cond=%s (val gain %+.4f)  %.0fs",
            dataset, W.shape[0], cell["arms"]["single_global"], cell["gain_oracle"],
            cell["gain_kmeans"], cell["gain_learned"],
            "route" if cell["conditioned_route_rate"] >= 0.5 else "abstain",
            cell["conditioned_validation_gain"], time.perf_counter() - t0,
        )
    return out


def find_thresholds(rows: list[dict]) -> dict:
    """Locate two boundaries on the lever, which need not coincide.

    * ``existence`` -- smallest separability at which routing gain *exists* at
      all (the label-informed oracle beats the global model). This is a property
      of the task, independent of any router.
    * ``deployable`` -- smallest separability at which the selective router both
      decided to route and actually won on the held-out test set. A gap between
      the two is the interesting region: routing is possible before it is
      reliably detectable.
    """
    ordered = sorted(rows, key=lambda r: r["separability"])
    existence = next((float(r["separability"]) for r in ordered if r["gain_oracle"] > 0), None)
    deployable = next(
        (float(r["separability"]) for r in ordered
         if r["conditioned_gain"] > 0 and r["conditioned_route_rate"] >= 0.5),
        None,
    )
    return {"existence": existence, "deployable": deployable}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--separabilities", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--n-features", type=int, default=8)
    p.add_argument("--edge-p", type=float, default=0.02)
    p.add_argument("--cross-edge-p", type=float, default=0.0)
    p.add_argument("--homophily-cross-edges", default="0,0.002,0.005,0.01,0.02",
                   help="cross-group edge probabilities for the secondary homophily lever")
    p.add_argument("--homophily-separability", type=float, default=1.0,
                   help="label separability held fixed during the homophily sweep")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--splits", type=int, default=3)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--expert", default="logistic", choices=EXPERT_NAMES)
    p.add_argument("--real", nargs="*", default=["amazon", "tolokers"],
                   help="real datasets to probe with the conditioned router")
    p.add_argument("--skip-sweep", action="store_true")
    p.add_argument("--skip-real", action="store_true")
    p.add_argument("--out", default="logs")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    log = get_logger("learnability_lever")
    set_seed(0)
    out_dir = ensure_dir(Path(args.out) / "learnability_lever")
    expert_factory = EXPERT_FACTORIES[args.expert]

    payload = {"config": vars(args), "expert": args.expert}

    if not args.skip_sweep:
        log.info("sweeping separability lever (%s)", args.separabilities)
        rows = sweep(args, log, expert_factory)
        payload["sweep"] = rows
        payload["separability_threshold"] = find_thresholds(rows)
        log.info("separability thresholds: %s", payload["separability_threshold"])
        log.info("sweeping homophily lever (cross_edge_p)")
        payload["homophily_sweep"] = sweep_homophily(args, log, expert_factory)

    if not args.skip_real:
        log.info("probing real pools: %s", args.real)
        payload["real"] = probe_real(args, log, expert_factory)

    suffix = "" if args.expert == "logistic" else f"_{args.expert}"
    out_path = out_dir / f"lever{suffix}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    log.info("wrote %s", out_path)


if __name__ == "__main__":
    main()

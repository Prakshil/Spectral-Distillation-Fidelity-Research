"""Fixed-expert (cross-fit) router protocol: does D2 survive honest routing?

The self-routing protocol in ``src/mixture.py`` gives every candidate credit for
free:

    fit_experts      : expert k trains on exactly the nodes routed to k
    mixture_accuracy : expert k is scored on exactly those same nodes

so ``accuracy = sum_g acc(G_g | model trained on G_g)``. Two things follow, and
this script measures both.

**1. Permutation invariance.** The channel *index* is irrelevant, so negating a
score -- swapping which quantile bucket is labelled 0 vs 2 -- cannot change
accuracy. The field's "orient by correlation with label homophily" step is a
no-op, and D2 cannot distinguish a score from its negation.

**2. Self-specialization credit.** Every partition is scored by experts fitted
to that same partition, so any partition with non-degenerate subgroups collects
partial credit regardless of whether its routing is *useful*. D2 then partly
measures "do subgroups support their own specialist", not "does this routing
send nodes to a *better* model".

A purely routing-blind expert pool (random disjoint splits) cannot resolve this:
the experts become near-identical and routing cannot matter at all, so every
condition collapses to the same accuracy. The informative design fixes the expert
pool on a routing-blind but *structurally meaningful* basis -- k-means on the
node features -- and varies only the routing. Then:

  * channel index becomes identifiable again, because expert k is permanently
    bound to feature region k, so score sign finally has content;
  * D2 measures whether *homophily-aligned* routing beats routing under a
    routing-independent specialist pool, i.e. whether label-free homophily
    alignment carries signal beyond feature-space structure alone;
  * feature-k-means enters as a label-free baseline, which the existing protocol
    never compared against.

Run alongside the existing protocol, this script reports both, so every number
can be read as "self-routing" versus "fixed-expert".
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import KMeans

from spectral_distillation.experiments.run_candidate_search import (
    EIG_KS,
    bucket_from_score,
    candidate_scores,
)
from spectral_distillation.src.evaluation import evaluate_comparison, holm_bonferroni
from spectral_distillation.src.mixture import (
    EXPERT_FACTORIES,
    fit_experts,
    mixture_accuracy,
    train_test_split,
)
from spectral_distillation.src.real_fraud import load_real_fraud
from spectral_distillation.src.router_protocol import (
    label_homophily,
    oracle_bucket_assignment,
    random_assignment,
)
from spectral_distillation.src.utils import ensure_dir, get_logger, set_seed

N_EXPERTS = 3
EXPERT_NAMES = ("logistic", "mlp", "gnn")


def build_expert_factory(name: str, graph, device: str):
    """Resolve an expert family name to an ``expert_factory`` callable.

    ``gnn`` is not in ``EXPERT_FACTORIES`` because it needs the graph: each
    expert message-passes over the symmetrically normalized adjacency, which is
    the only way a specialist can exploit the structure a router routes on.
    """
    if name == "gnn":
        from spectral_distillation.src.gnn_expert import gnn_head_factory
        from spectral_distillation.src.laplacian import normalize_adjacency

        W_norm = normalize_adjacency(graph["W"])
        return gnn_head_factory(W_norm=W_norm, n_features=graph["d_features"],
                               device=device)
    return EXPERT_FACTORIES[name]


def feature_kmeans_routing(X: np.ndarray, n_experts: int = N_EXPERTS, seed: int = 0) -> np.ndarray:
    """Routing-blind expert pool: k-means on features, never sees labels."""
    km = KMeans(n_clusters=n_experts, n_init=10, random_state=seed)
    lab = km.fit_predict(X)
    out = np.zeros((X.shape[0], n_experts))
    out[np.arange(X.shape[0]), lab] = 1.0
    return out


def uniform_ensemble_accuracy_fixed(
    X: np.ndarray,
    y: np.ndarray,
    experts: list[object | None],
    test_mask: np.ndarray,
) -> float:
    idx = np.flatnonzero(test_mask)
    if idx.size == 0:
        return 0.0
    all_preds = []
    for k, head in enumerate(experts):
        if head is None:
            continue
        if hasattr(head, "predict_nodes"):
            pred = head.predict_nodes(X, idx)
        else:
            try:
                pred = head.predict(X[idx])
            except Exception:
                continue
        all_preds.append(pred)
    if len(all_preds) == 0:
        return 0.0
    votes = np.stack(all_preds, axis=0)
    try:
        from scipy.stats import mode
        ens = mode(votes, axis=0, keepdims=False).mode
    except Exception:
        ens = votes[0]
    correct = int(np.count_nonzero(ens == y[idx]))
    return float(correct / idx.size)


def evaluate_all(
    conditions: dict[str, np.ndarray],
    X,
    y,
    splits: int,
    seeds: int,
    log,
    expert_assignment: np.ndarray | None = None,
    expert_factory=None,
):
    """Mean accuracy per split, averaged over seeds, for every routing.

    ``expert_assignment`` fixes the expert pool: experts are fitted once per
    (split, seed) from that partition and reused for every routing. This is the
    whole point of protocol B -- without it the experts would be refitted from
    each candidate's own routing and the two protocols would coincide.

    Because that pool does not depend on the condition, it is fitted once per
    (split, seed) and shared across all conditions. Fitting is deterministic in
    ``(X, y, assignment, train)``, so this is a pure speedup: it removes the
    56x duplicate work of refitting an identical pool per condition.
    """
    shared_pool = expert_assignment is not None
    per_cond_acc: dict[str, list[float]] = {c: [] for c in conditions}
    t0 = time.perf_counter()
    for split in range(splits):
        for seed in range(seeds):
            train, te = train_test_split(y, split, seed)
            if shared_pool:
                pool = fit_experts(
                    X, y, expert_assignment, train, expert_factory=expert_factory
                )
                experts_per_cond = {c: pool for c in conditions}
                # The no-routing reference must be a model fitted on *every*
                # training node. Reusing the k-means pool would silently score
                # "route everything to the cluster-0 expert", which never saw
                # the other two clusters -- a weaker, mislabelled baseline. It
                # costs one extra head fit per (split, seed).
                if "single_global" in conditions:
                    experts_per_cond["single_global"] = fit_experts(
                        X, y, np.ones((X.shape[0], 1)), train,
                        expert_factory=expert_factory,
                    )
            else:
                experts_per_cond = {
                    c: fit_experts(X, y, a, train, expert_factory=expert_factory)
                    for c, a in conditions.items()
                }
            for cond, a in conditions.items():
                acc, _ = mixture_accuracy(X, y, a, experts_per_cond[cond], te)
                per_cond_acc[cond].append(acc)
            # ensemble
            if shared_pool:
                ens_acc = uniform_ensemble_accuracy_fixed(X, y, pool, te)
                per_cond_acc.setdefault("ensemble_uniform", []).append(ens_acc)
        log.info("  split %d/%d (%.0fs)", split + 1, splits, time.perf_counter() - t0)

    n_seeds = max(1, seeds)
    out = {
        cond: {
            split: float(np.mean(vals[split * n_seeds:(split + 1) * n_seeds]))
            for split in range(splits)
        }
        for cond, vals in per_cond_acc.items()
    }
    return out


def single_global_condition(n: int) -> np.ndarray:
    """One column of ones -> argmax routes every node to the single expert.

    Fitted against a one-column pool this head sees *all* training nodes, so the
    condition is a genuine no-routing baseline: the accuracy one global model
    gets, which upper-bounds what any routing scheme can contribute. Under a
    shared multi-column pool that would no longer hold, so ``evaluate_all``
    refits this condition against its own one-column pool.
    """
    return np.ones((n, 1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="amazon", choices=["amazon", "tolokers", "yelpchi"])
    p.add_argument("--data-path", default=None)
    p.add_argument("--max-nodes", type=int, default=None)
    p.add_argument("--splits", type=int, default=10)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--out", default="logs")
    p.add_argument("--device", default="cpu")
    p.add_argument("--expert", default="logistic", choices=EXPERT_NAMES,
                   help="expert head family; 'mlp' adds non-linear capacity, "
                        "'gnn' adds message passing over the graph")
    p.add_argument("--skip-self-routing", action="store_true",
                   help="run only the honest fixed-expert protocol (much faster with mlp)")
    args = p.parse_args()

    log = get_logger("fixed_expert")
    set_seed(0)
    # Expert family is part of the artifact identity: a logistic run and an mlp
    # run are different experiments and must not overwrite each other.
    suffix = "" if args.expert == "logistic" else f"_{args.expert}"
    out_dir = ensure_dir(Path(args.out) / f"{args.dataset}_fixed_expert{suffix}")

    t0 = time.perf_counter()
    graph = load_real_fraud(args.dataset, args.data_path, max_nodes=args.max_nodes or None)
    W, X, y = graph["W"], graph["features"], graph["y"]
    log.info("loaded %s n=%d (%.1fs)", args.dataset, W.shape[0], time.perf_counter() - t0)

    scores = candidate_scores(graph, args.device, with_resistance=False, log=log)
    h_true = label_homophily(W, y)
    rho = {}
    for name, s in scores.items():
        m = np.isfinite(s) & np.isfinite(h_true)
        rho[name] = float(spearmanr(s[m], h_true[m]).statistic)

    oracle = oracle_bucket_assignment(W, y, adaptive=True)
    km = feature_kmeans_routing(X)
    single = single_global_condition(W.shape[0])

    base_cond: dict[str, np.ndarray] = {
        "single_global": single,
        "oracle": oracle,
        "feature_kmeans": km,
        "random": random_assignment(oracle, rng=0),
    }
    for name, s in scores.items():
        base_cond[f"{name}::fwd"] = bucket_from_score(s)
        base_cond[f"{name}::neg"] = bucket_from_score(-s)

    factory = build_expert_factory(args.expert, graph, args.device)
    log.info("expert head: %s", args.expert)

    # ------------------------------------------------------------------ #
    # Protocol B: fixed routing-blind expert pool (k-means), routing varies.
    # This is the honest protocol and the one that matters.
    # ------------------------------------------------------------------ #
    log.info("protocol B (fixed k-means expert pool): %d conditions", len(base_cond))
    a_fixed = evaluate_all(
        base_cond, X, y, args.splits, args.seeds, log,
        expert_assignment=km, expert_factory=factory,
    )

    # ------------------------------------------------------------------ #
    # Protocol A: self-routing -- each expert fitted to its own routing.
    # Optional: with MLP/GNN experts this refits per condition and is slow.
    # ------------------------------------------------------------------ #
    if args.skip_self_routing:
        log.info("protocol A skipped (--skip-self-routing)")
        a_self = None
    else:
        log.info("protocol A (self-routing): %d conditions", len(base_cond))
        a_self = evaluate_all(
            base_cond, X, y, args.splits, args.seeds, log, expert_factory=factory
        )

    # ------------------------------------------------------------------ #
    # Mechanism 1: permutation invariance
    # ------------------------------------------------------------------ #
    inv_rows = []
    for name in scores:
        d_fixed = max(
            abs(a_fixed[f"{name}::fwd"][s] - a_fixed[f"{name}::neg"][s])
            for s in range(args.splits)
        )
        d_self = (
            max(
                abs(a_self[f"{name}::fwd"][s] - a_self[f"{name}::neg"][s])
                for s in range(args.splits)
            )
            if a_self is not None
            else None
        )
        inv_rows.append({
            "candidate": name,
            "rho": rho[name],
            "max_abs_delta_self_routing": d_self,
            "max_abs_delta_fixed_expert": d_fixed,
        })
    n_inv_fixed = sum(1 for r in inv_rows if r["max_abs_delta_fixed_expert"] < 1e-12)
    n_inv_self = (
        sum(1 for r in inv_rows if r["max_abs_delta_self_routing"] < 1e-12)
        if a_self is not None
        else None
    )
    log.info("sign-invariant candidates: self-routing %s/%d, fixed-expert %d/%d",
             n_inv_self, len(inv_rows), n_inv_fixed, len(inv_rows))

    # ------------------------------------------------------------------ #
    # Mechanism 2: does homophily routing beat a routing-blind specialist?
    # ------------------------------------------------------------------ #
    def d2_table(acc: dict) -> tuple[list[dict], dict]:
        keys = [f"{n}::fwd" for n in scores]
        comps = [evaluate_comparison(acc[k], acc["random"], f"{k}_vs_random") for k in keys]
        thr = holm_bonferroni([c.p_wilcoxon for c in comps], 0.05)
        rows = []
        for name, c, t in zip(scores, comps, thr):
            # The meaningful reference under a fixed specialist pool: does this
            # routing beat having no routing at all? Positive means the routing
            # actively helps; negative means it is worse than one global model.
            g = evaluate_comparison(acc[f"{name}::fwd"], acc["single_global"],
                                    f"{name}_vs_global")
            rows.append({
                "candidate": name,
                "rho": rho[name],
                "d2": c.mean_diff,
                "p": c.p_wilcoxon,
                "dz": c.effect_size_dz,
                "holm_significant": bool(c.mean_diff > 0 and c.p_wilcoxon < t),
                "d2_vs_no_routing": g.mean_diff,
                "p_vs_no_routing": g.p_wilcoxon,
                "dz_vs_no_routing": g.effect_size_dz,
                "beats_no_routing": bool(g.mean_diff > 0),
            })
        rows.sort(key=lambda r: -r["d2"])
        orc = evaluate_comparison(acc["oracle"], acc["random"], "oracle_vs_random")
        kmc = evaluate_comparison(acc["feature_kmeans"], acc["random"], "kmeans_vs_random")
        glob = evaluate_comparison(acc["single_global"], acc["random"], "global_vs_random")
        refs = {
            "oracle_d2": orc.mean_diff, "oracle_dz": orc.effect_size_dz,
            "feature_kmeans_d2": kmc.mean_diff, "feature_kmeans_p": kmc.p_wilcoxon,
            "feature_kmeans_dz": kmc.effect_size_dz,
            "single_global_d2": glob.mean_diff, "single_global_p": glob.p_wilcoxon,
            "single_global_dz": glob.effect_size_dz,
            "n_holm_significant": sum(1 for r in rows if r["holm_significant"]),
            "n_candidates": len(rows),
            # Absolute accuracies, not just deltas. A negative routing result is
            # only meaningful if the experts are actually competent, and that is
            # checked against these numbers rather than assumed.
            "mean_accuracy": {
                c: float(np.mean(list(acc[c].values())))
                for c in ("single_global", "oracle", "feature_kmeans", "random")
            },
            "best_candidate_accuracy": float(
                np.mean(list(acc[rows[0]["candidate"] + "::fwd"].values()))
            ),
        }
        return rows, refs

    rows_fixed, ref_fixed = d2_table(a_fixed)
    rows_self, ref_self = (d2_table(a_self) if a_self is not None else (None, None))

    print("\n=== permutation invariance: max |acc(s) - acc(-s)| per candidate ===")
    print(f"self-routing sign-invariant: {n_inv_self}/{len(inv_rows)}   "
          f"fixed-expert: {n_inv_fixed}/{len(inv_rows)}")

    hdr = "{:<26s}{:>9s}{:>11s}{:>11s}{:>9s}{:>8s}{:>13s}"

    def report(title, rows, ref):
        print(f"\n=== {title}: D2 vs random, Holm over {ref['n_candidates']} ===")
        print(f"single-global (no routing) D2={ref['single_global_d2']:+.5f} "
              f"(dz {ref['single_global_dz']:.2f}, p={ref['single_global_p']:.4f})")
        print(f"oracle D2={ref['oracle_d2']:+.5f} (dz {ref['oracle_dz']:.2f})   "
              f"feature-kmeans D2={ref['feature_kmeans_d2']:+.5f} "
              f"(dz {ref['feature_kmeans_dz']:.2f}, p={ref['feature_kmeans_p']:.4f})")
        print(hdr.format("candidate", "rho", "D2", "p", "dz", "sig", "vs-no-route"))
        for r in rows[:10]:
            print(hdr.format(
                r["candidate"], f"{r['rho']:+.3f}", f"{r['d2']:+.5f}", f"{r['p']:.5f}",
                f"{r['dz']:+.2f}", "YES" if r["holm_significant"] else "",
                f"{r['d2_vs_no_routing']:+.5f}",
            ))
        print(f"Holm-significant: {ref['n_holm_significant']}/{ref['n_candidates']}"
              f"   beats no-routing: {sum(1 for r in rows if r['beats_no_routing'])}"
              f"/{len(rows)}")

    if rows_self is not None:
        report("SELF-ROUTING (existing)", rows_self, ref_self)
    report("FIXED EXPERTS (k-means pool)", rows_fixed, ref_fixed)

    from scipy.stats import spearmanr as _sp

    if rows_self is not None:
        rs = {r["candidate"]: i for i, r in enumerate(rows_self)}
        rf = {r["candidate"]: i for i, r in enumerate(rows_fixed)}
        common = sorted(set(rs) & set(rf), key=lambda c: rs[c])
        rank_rho = float(_sp([rs[c] for c in common], [rf[c] for c in common]).statistic)
        print(f"\nrank correlation of D2 between protocols: {rank_rho:+.3f} "
              f"({len(common)} candidates)")
    else:
        rank_rho = None

    summary = {
        "config": {**vars(args), "eig_ks": list(EIG_KS), "n_experts": N_EXPERTS},
        "graph": {
            "n_nodes": graph["n_nodes"], "d_features": graph["d_features"],
            "n_edges": graph["n_edges"], "positive_ratio": graph["positive_ratio"],
        },
        "invariance": {
            "n_candidates": len(inv_rows),
            "n_sign_invariant_self_routing": n_inv_self,
            "n_sign_invariant_fixed_expert": n_inv_fixed,
            "per_candidate": inv_rows,
        },
        "self_routing": ({"results": rows_self, **ref_self} if rows_self else None),
        "fixed_expert": {"results": rows_fixed, **ref_fixed},
        "d2_rank_correlation_between_protocols": rank_rho,
    }
    (out_dir / "fixed_expert.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("wrote", out_dir / "fixed_expert.json")


if __name__ == "__main__":
    main()

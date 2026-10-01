"""Label-free router candidate sweep for D2 (real fraud graphs).

D2 asks whether a router that never sees labels can still beat a usage-matched
random assignment. On both real graphs it currently fails. The Tolokers ladder
(`docs/sparsify_ladder_results.md`) showed a concrete lead: degree-based
sparsification *significantly improves* D2 on Tolokers, which means the label-free
signal is recoverable once node-local structure is sharpened -- the problem is
which label-free structural quantity the router should rank on.

The shipped router orders its clusters by *neighbor-averaged node-feature
similarity* (`router_protocol.structural_features`). That is a proxy for
neighborhood similarity in **feature** space, whereas routing utility is governed
by neighborhood similarity in **label** space. This experiment tests whether a
better label-free proxy exists.

Two stages:

1. **Screen** (cheap, no training): Spearman correlation between each candidate
   label-free score and the *true* label homophily h(v). Labels are used here only
   to *diagnose* which proxy is informative -- never to build the assignment.
2. **Evaluate** (expensive): build the same 3-bucket assignment the oracle uses,
   but from the label-free score, and measure the routing metric against the
   shared uniform / random baselines. A candidate with a positive, significant
   gain over random is a candidate D2 fix.

Candidates are oriented so that HIGHER always means "more label-homophilic", so
they are bucketed with the oracle's adaptive-quantile rule unchanged.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.stats import spearmanr

from spectral_distillation.src.effective_resistance import effective_resistance_exact
from spectral_distillation.src.evaluation import evaluate_comparison, holm_bonferroni
from spectral_distillation.src.laplacian import compute_laplacian, eigh_symmetric
from spectral_distillation.src.mixture import fit_experts, mixture_accuracy, train_test_split
from spectral_distillation.src.real_fraud import load_real_fraud
from spectral_distillation.src.router_protocol import (
    label_homophily,
    oracle_bucket_assignment,
    random_assignment,
    uniform_assignment,
)
from spectral_distillation.src.utils import ensure_dir, get_logger, set_seed


def _row_normalize(X: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(nrm, 1e-12)


def candidate_scores(
    graph: dict,
    device: str,
    with_resistance: bool,
    log,
) -> dict[str, np.ndarray]:
    """Label-free per-node scores, each oriented so higher == more homophilic."""
    W = graph["W"]
    X = graph["features"]
    n = W.shape[0]
    Wsp = sp.csr_matrix(W > 0)
    deg = np.asarray(Wsp.sum(axis=1)).ravel()

    scores: dict[str, np.ndarray] = {}

    # 1. neighbor-averaged feature similarity (what the shipped router ranks on)
    Xn = _row_normalize(X)
    S = Xn @ Xn.T
    nb_sum = np.asarray(Wsp.multiply(S).sum(axis=1)).ravel()
    scores["neigh_feat_sim"] = np.divide(nb_sum, deg, out=np.zeros(n), where=deg > 0)
    del S

    # 2. local clustering coefficient via sparse A@A (triangles per node)
    A2 = (Wsp @ Wsp).tocsr()
    tri = np.asarray(Wsp.multiply(A2).sum(axis=1)).ravel()
    denom = deg * (deg - 1)
    scores["local_clust"] = np.divide(
        tri, denom, out=np.zeros(n), where=denom > 0
    )
    del A2

    # 3. degree
    scores["degree"] = deg.astype(float)

    # 4. mean neighbor degree (degree assortativity proxy)
    nb_deg = np.asarray(Wsp @ deg[:, None]).ravel()
    scores["neigh_degree"] = np.divide(nb_deg, deg, out=np.zeros(n), where=deg > 0)

    # 5. spectral neighborhood similarity: how alike a node's nontrivial spectral
    #    embedding is to its neighbors' (smooth core vs periphery). Computed as
    #    <V_i, A_norm V_i> with the neighbor average applied once via sparse matmul.
    L = compute_laplacian(W)
    evals, V = eigh_symmetric(L, device=device)
    k = min(8, max(V.shape[1] - 1, 1))
    Vk = _row_normalize(np.asarray(V[:, 1 : k + 1]))
    Vnb = np.asarray(Wsp @ Vk)
    scores["eig_nb_sim"] = np.divide(
        np.einsum("ij,ij->i", Vk, Vnb), deg, out=np.zeros(n), where=deg > 0
    )
    del Vk, Vnb

    # 6. eigenvector centrality (structural hub-ness)
    scores["eig_centrality"] = np.abs(V[:, 1]) if V.shape[1] > 1 else np.zeros(n)

    # 7. mean effective resistance to neighbors. Low R == tightly coupled
    #    neighborhood == homophilic core, so negate to keep "higher = more
    #    homophilic". This is the most direct label-free mixing-rate proxy but
    #    costs a dense pinvh, hence the flag.
    if with_resistance:
        t0 = time.perf_counter()
        R = effective_resistance_exact(L)
        log.info("exact effective resistance ready in %.1fs", time.perf_counter() - t0)
        if not np.isfinite(R).any():
            log.warning("effective resistance all non-finite; dropping candidate")
        else:
            cap = np.nanmax(R[np.isfinite(R)])
            Rb = np.where(np.isfinite(R), R, cap)
            Rsp = sp.csr_matrix(Rb)
            r_sum = np.asarray(Rsp.multiply(Wsp).sum(axis=1)).ravel()
            mean_r = np.divide(r_sum, deg, out=np.zeros(n), where=deg > 0)
            scores["inv_mean_res_nb"] = -mean_r
            del R, Rb, Rsp
    else:
        log.info("skipping effective-resistance candidate (pass --with-resistance)")

    return {k_: v for k_, v in scores.items() if np.all(np.isfinite(v))}


def bucket_from_score(score: np.ndarray, n_experts: int = 3) -> np.ndarray:
    """Oracle-identical bucketing (adaptive quantiles) but driven by a label-free
    score: high score -> channel 0 (low-pass), mid -> 1, low -> 2 (high-pass)."""
    lo, hi = np.quantile(score, [1 / 3, 2 / 3])
    assignment = np.zeros((score.size, n_experts))
    ch = np.where(score >= hi, 0, np.where(score <= lo, 2, 1))
    assignment[np.arange(score.size), ch] = 1.0
    return assignment


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="tolokers", choices=["amazon", "tolokers", "yelpchi"])
    p.add_argument("--data-path", default=None)
    p.add_argument("--max-nodes", type=int, default=None,
                   help="yelpchi only: cap the sampled node count "
                        "(default loader cap 15000; use 0 for the full graph)")
    p.add_argument("--splits", type=int, default=10)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--with-resistance", action="store_true",
                   help="include the exact effective-resistance candidate (dense pinvh)")
    p.add_argument("--screen-only", action="store_true",
                   help="only report proxy/homophily correlations, skip accuracy eval")
    p.add_argument("--top-k-eval", type=int, default=4,
                   help="evaluate only the top-N most informative candidates")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--out", default="logs")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    log = get_logger("d2_candidates")
    set_seed(0)
    out_dir = ensure_dir(Path(args.out) / f"{args.dataset}_d2_candidates")

    t0 = time.perf_counter()
    graph = load_real_fraud(args.dataset, args.data_path, max_nodes=args.max_nodes or None)
    log.info("loaded %s n=%d d=%d edges=%d pos=%.4f in %.1fs", args.dataset,
             graph["n_nodes"], graph["d_features"], graph["n_edges"],
             graph["positive_ratio"], time.perf_counter() - t0)

    t0 = time.perf_counter()
    scores = candidate_scores(graph, args.device, args.with_resistance, log)
    log.info("computed %d candidate scores in %.1fs", len(scores), time.perf_counter() - t0)

    # ---- stage 1: diagnostic screen against the true label homophily ----
    h_true = label_homophily(graph["W"], graph["y"])
    finite = np.isfinite(h_true) & np.isfinite(graph["y"])
    screen = []
    for name, s in scores.items():
        m = np.isfinite(s) & finite
        rho, pval = spearmanr(s[m], h_true[m])
        screen.append({"candidate": name, "spearman_rho": float(rho), "p": float(pval)})
    screen.sort(key=lambda r: -abs(r["spearman_rho"]))

    print("\n=== stage 1: label-free proxy vs true label homophily (diagnostic) ===")
    print("{:<18s}{:>10s}{:>12s}{:>14s}".format("candidate", "rho", "p", "orientation"))
    for r in screen:
        orient = "as-is" if r["spearman_rho"] >= 0 else "sign-flipped"
        print("{:<18s}{:>10.4f}{:>12.3g}{:>14s}".format(
            r["candidate"], r["spearman_rho"], r["p"], orient))

    record = {"dataset": args.dataset, "screen": screen, "evaluated": []}
    if args.screen_only:
        (out_dir / "candidates.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8")
        return

    # ---- stage 2: evaluate the most informative candidates as real routers ----
    chosen = [r["candidate"] for r in screen[: args.top_k_eval]]
    log.info("evaluating candidates: %s", chosen)

    X, y = graph["features"], graph["y"]
    conditions: dict[str, np.ndarray] = {
        "uniform": uniform_assignment(X.shape[0], 3),
    }
    oracle = oracle_bucket_assignment(graph["W"], y, adaptive=True)
    conditions["oracle"] = oracle
    conditions["random"] = random_assignment(oracle, rng=0)
    for c in chosen:
        # Correlate with the true label homophily to learn the *sign*, then
        # orient the candidate so higher score == more homophilic. Without this,
        # a strongly anti-correlated candidate is fed to the router reversed and
        # can never work (this is what silently killed neigh_degree on amazon).
        rho = next(r["spearman_rho"] for r in screen if r["candidate"] == c)
        signed = scores[c] if rho >= 0 else -scores[c]
        conditions[c] = bucket_from_score(signed)

    t0 = time.perf_counter()
    split_means: dict[str, dict[int, float]] = {c: {} for c in conditions}
    for split in range(args.splits):
        for cond, a in conditions.items():
            accs = []
            for seed in range(args.seeds):
                train, test = train_test_split(y, split, seed)
                experts = fit_experts(X, y, a, train)
                acc, _ = mixture_accuracy(X, y, a, experts, test)
                accs.append(acc)
            split_means[cond][split] = float(np.mean(accs))
    log.info("scored %d conditions x %d splits x %d seeds in %.1fs",
             len(conditions), args.splits, args.seeds, time.perf_counter() - t0)

    comps = []
    for c in chosen:
        comps.append(evaluate_comparison(split_means[c], split_means["random"],
                                        f"{c}_vs_random"))
        comps.append(evaluate_comparison(split_means[c], split_means["uniform"],
                                        f"{c}_vs_uniform"))
    comps.append(evaluate_comparison(split_means["oracle"], split_means["random"],
                                    "oracle_vs_random"))

    # Holm-Bonferroni across the D2 family only (candidate vs random).
    # The oracle is excluded: it is the reference ceiling, not a member of the
    # family of competing label-free routers.
    d2_idx = [
        i for i, c in enumerate(comps)
        if c.name.endswith("_vs_random") and c.name != "oracle_vs_random"
    ]
    corrected = holm_bonferroni([comps[i].p_wilcoxon for i in d2_idx], 0.05)
    d2_results = []
    for i, thresh in zip(d2_idx, corrected):
        comps[i].significant = bool(comps[i].mean_diff > 0 and comps[i].p_wilcoxon < thresh)
        d2_results.append({
            "candidate": comps[i].name.replace("_vs_random", ""),
            "mean_diff": comps[i].mean_diff,
            "p_wilcoxon": comps[i].p_wilcoxon,
            "alpha_corrected": thresh,
            "effect_size_dz": comps[i].effect_size_dz,
            "d2_supported": comps[i].significant,
        })
    d2_results.sort(key=lambda r: -r["mean_diff"])
    record["evaluated"] = [
        {
            "name": c.name,
            "mean_diff": c.mean_diff,
            "ci_low": c.ci_low,
            "ci_high": c.ci_high,
            "p_wilcoxon": c.p_wilcoxon,
            "effect_size_dz": c.effect_size_dz,
            "significant": c.significant,
        }
        for c in comps
    ]
    record["d2_family"] = d2_results

    print(f"\n=== stage 2: does any label-free proxy fix D2 on {args.dataset}? ===")
    print("{:<18s}{:>11s}{:>11s}{:>9s}{:>13s}".format(
        "candidate", "vs_random", "p", "dz", "D2 fixed?"))
    for r in d2_results:
        print("{:<18s}{:>+11.5f}{:>11.6f}{:>9.3f}{:>13s}".format(
            r["candidate"], r["mean_diff"], r["p_wilcoxon"],
            r["effect_size_dz"], "YES" if r["d2_supported"] else "no"))
    ref = next(c for c in comps if c.name == "oracle_vs_random")
    print("{:<18s}{:>+11.5f}{:>11.6f}{:>9.3f}{:>13s}".format(
        "*oracle(ref)", ref.mean_diff, ref.p_wilcoxon, ref.effect_size_dz, "-"))

    (out_dir / "candidates.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    log.info("wrote %s", out_dir / "candidates.json")


if __name__ == "__main__":
    main()

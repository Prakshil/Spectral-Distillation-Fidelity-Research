"""Broad, predeclared candidate search for the D2 label-free router.

The committed screen (``run_d2_candidates.py``) tested 6-7 candidates and
evaluated only the top 4 by |Spearman rho| against label homophily. That is two
weaknesses this script removes:

1. **Selection bias.** Choosing which candidates to evaluate by their correlation
   with the labels, then testing those same labels for significance, is a
   screen-then-test on the same data. Here the pool is *predeclared* and
   **every** candidate is evaluated on the full protocol -- the marginal cost is
   ~0.04 s/cell, so there is no reason to pre-filter.
2. **Orientation.** ``run_d2_candidates.py`` orients each candidate by the sign
   of its label correlation, so an anti-correlated candidate gets flipped using
   the labels. That is a diagnostic, not a deployable router: it cannot be run on
   unseen data. Every candidate here is therefore evaluated twice:
   ``as_defined`` (the sign fixed a priori from the structural rationale, no
   labels) and ``label_oriented`` (sign chosen from rho, upper bound only). A
   candidate that only works label-oriented is reported as such and is not
   counted as fixing D2.

Holm-Bonferroni is applied across the entire declared family of ``as_defined``
D2 comparisons, so the family is known in advance rather than assembled after
seeing results.
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

# k values for the eigenmode-prefix family. 128 is included because the
# spectral-k sweep found it roughly doubles D2 on Tolokers and YelpChi.
EIG_KS = (2, 8, 32, 128)


def _row_normalize(X: np.ndarray) -> np.ndarray:
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def _nbr_avg(Wsp: sp.csr_matrix, vec: np.ndarray, deg: np.ndarray) -> np.ndarray:
    """Mean of ``vec`` over each node's neighbours (0 for isolated nodes)."""
    return np.divide(np.asarray(Wsp @ vec).ravel(), deg, out=np.zeros_like(vec), where=deg > 0)


def _nbr_dot_mean(Wsp: sp.csr_matrix, M: np.ndarray, deg: np.ndarray) -> np.ndarray:
    """Mean over neighbours of the per-pair inner product ``<M[i,j], M[j]>``.

    Equals ``mean_j (W_ij * <M_i, M_j>)``, i.e. a neighbourhood-averaged
    similarity. This is a neighbour-*weighted* sum over the second axis, so it
    cannot be expressed as a plain matrix product.
    """
    n = M.shape[0]
    out = np.zeros(n)
    indptr, indices = Wsp.indptr, Wsp.indices
    for i in range(n):
        nb = indices[indptr[i] : indptr[i + 1]]
        if nb.size:
            out[i] = float(np.dot(M[i], M[nb].sum(axis=0)))
    return np.divide(out, deg, out=np.zeros(n), where=deg > 0)


def candidate_scores(graph: dict, device: str, with_resistance: bool, log) -> dict[str, np.ndarray]:
    """Predeclared label-free per-node scores.

    Each entry documents the *a priori* sign convention: the value returned is
    oriented "higher == more homophilic" per the rationale in its comment, with
    no reference to labels. The empirical rho is measured later, and disagreement
    with this convention is itself a reported result.
    """
    W, X = graph["W"], graph["features"]
    n = W.shape[0]
    Wsp = sp.csr_matrix(W > 0)
    deg = np.asarray(Wsp.sum(axis=1)).ravel().astype(float)
    nbrs = Wsp.tolil().rows
    nbr_count = np.array([len(nb) for nb in nbrs], dtype=float)
    scores: dict[str, np.ndarray] = {}

    # ---- feature-space ------------------------------------------------------
    Xn = _row_normalize(X)
    scores["neigh_feat_sim"] = _nbr_dot_mean(Wsp, Xn, deg)
    # Boundary nodes sit far from every feature centroid: low centrality in
    # feature space == fewer same-label neighbours. Negated.
    Xc = X - X.mean(axis=0)
    scores["feat_anomaly"] = -np.linalg.norm(Xc, axis=1)
    # Spread of a node's neighbours' features: high == heterophilic boundary.
    nbr_var = np.zeros(n)
    for i in range(n):
        if nbr_count[i] > 1:
            nb = np.asarray(nbrs[i])
            nbr_var[i] = float(np.mean(np.var(Xn[nb], axis=0)))
    scores["neigh_feat_var"] = nbr_var
    del Xc, Xn

    # ---- degree family ------------------------------------------------------
    scores["degree"] = deg
    scores["neigh_degree"] = _nbr_avg(Wsp, deg, deg)
    scores["log_degree"] = np.log1p(deg)
    # Assortativity: mean |deg_nb - deg_i| is low for well-mixed homophilous
    # neighbourhoods. Negated.
    assort = np.zeros(n)
    for i in range(n):
        if nbr_count[i] > 0:
            assort[i] = float(np.mean(np.abs(np.asarray(deg[nbrs[i]]) - deg[i])))
    scores["neigh_deg_dissimilarity"] = -assort

    # ---- triangle family ----------------------------------------------------
    A2 = (Wsp @ Wsp).tocsr()
    tri = np.asarray(Wsp.multiply(A2).sum(axis=1)).ravel()
    scores["triangle_count"] = tri
    denom = deg * (deg - 1)
    scores["local_clust"] = np.divide(tri, denom, out=np.zeros(n), where=denom > 0)
    del A2
    # Triforce: neighbours-of-neighbours reached through a triangle.
    scores["triforce"] = tri

    # ---- diffusion ----------------------------------------------------------
    # Personalised PageRank mass: high == central, so negated.
    scores["inv_pagerank"] = -_pagerank(Wsp, n)
    # Random-walk return probability over 2 steps: high == well-mixed == less
    # homophilous, so negated.
    p = np.divide(deg, deg.sum(), out=np.zeros(n), where=deg > 0)
    two_step = np.asarray(Wsp @ (p * deg))
    scores["inv_walk2_mixing"] = -np.divide(two_step, deg, out=np.zeros(n), where=deg > 0)
    del p, two_step

    # ---- spectral: one shared eigendecomposition ----------------------------
    t0 = time.perf_counter()
    L = compute_laplacian(W)
    evals, V = eigh_symmetric(L, device=device)
    log.info("eigendecomposition shared by all spectral candidates in %.1fs",
             time.perf_counter() - t0)
    V = np.asarray(V)

    # Smoothness over an eigenmode prefix at several k.
    for k in EIG_KS:
        kk = min(k, max(V.shape[1] - 1, 1))
        Vk = _row_normalize(V[:, 1 : kk + 1])
        scores[f"eig_nb_sim_k{k}"] = np.divide(
            np.einsum("ij,ij->i", Vk, np.asarray(Wsp @ Vk)), deg,
            out=np.zeros(n), where=deg > 0,
        )
        # Same but on |V|, which makes peripheral nodes score low without
        # cancelling signs between modes.
        Va = _row_normalize(np.abs(V[:, 1 : kk + 1]))
        scores[f"eig_nb_sim_abs_k{k}"] = np.divide(
            np.einsum("ij,ij->i", Va, np.asarray(Wsp @ Va)), deg,
            out=np.zeros(n), where=deg > 0,
        )
        # Eigenmode centrality prefix: mass carried by the leading modes.
        scores[f"eig_centrality_k{k}"] = np.abs(V[:, 1 : kk + 1]).sum(axis=1)

    # Fiedler smoothness (signed, first nontrivial mode only).
    v1 = V[:, 1] if V.shape[1] > 1 else np.zeros(n)
    scores["fiedler_nb_sim"] = _nbr_avg(Wsp, v1, deg)
    # Local spectral energy: sum of squared participation over all modes.
    scores["spectral_energy"] = (V ** 2).sum(axis=1)
    del L, V

    # ---- resistance ---------------------------------------------------------
    if with_resistance:
        t0 = time.perf_counter()
        R = effective_resistance_exact(compute_laplacian(W))
        log.info("exact effective resistance ready in %.1fs", time.perf_counter() - t0)
        if not np.isfinite(R).any():
            log.warning("effective resistance all non-finite; dropping candidate")
        else:
            cap = float(np.nanmax(R[np.isfinite(R)]))
            Rb = np.where(np.isfinite(R), R, cap)
            scores["inv_mean_res_nb"] = -_nbr_avg(sp.csr_matrix(Rb), np.ones(n), deg)
            # Min-resistance edge incident to a node == best-coupled neighbour.
            scores["inv_min_res"] = -np.array(
                [
                    float(min(Rb[i, nb]) ) if len(nb) else cap
                    for i, nb in enumerate(nbrs)
                ]
            )
            del R, Rb
    else:
        log.info("skipping effective-resistance candidates (pass --with-resistance)")

    return {k_: v for k_, v in scores.items() if np.all(np.isfinite(v))}


def _pagerank(Wsp: sp.csr_matrix, n: int, alpha: float = 0.85, iters: int = 100) -> np.ndarray:
    """Power-iteration PageRank on the sparse graph (avoids networkx overhead)."""
    deg = np.asarray(Wsp.sum(axis=1)).ravel()
    inv = np.divide(1.0, deg, out=np.zeros(n), where=deg > 0)
    P = sp.diags(inv) @ Wsp
    x = np.full(n, 1.0 / n)
    for _ in range(iters):
        x = alpha * (P.T @ x) + (1 - alpha) / n
    return x


def bucket_from_score(score: np.ndarray, n_experts: int = 3) -> np.ndarray:
    """Adaptive-quantile buckets, identical to the oracle's rule."""
    lo, hi = np.quantile(score, [1 / 3, 2 / 3])
    assignment = np.zeros((score.size, n_experts))
    ch = np.where(score >= hi, 0, np.where(score <= lo, 2, 1))
    assignment[np.arange(score.size), ch] = 1.0
    return assignment


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="amazon",
                   choices=["amazon", "tolokers", "yelpchi"])
    p.add_argument("--data-path", default=None)
    p.add_argument("--max-nodes", type=int, default=None, help="yelpchi only")
    p.add_argument("--splits", type=int, default=10)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--with-resistance", action="store_true")
    p.add_argument("--out", default="logs")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    log = get_logger("candidate_search")
    set_seed(0)
    out_dir = ensure_dir(Path(args.out) / f"{args.dataset}_candidate_search")

    t0 = time.perf_counter()
    graph = load_real_fraud(args.dataset, args.data_path, max_nodes=args.max_nodes or None)
    W, X, y = graph["W"], graph["features"], graph["y"]
    log.info("loaded %s n=%d edges=%d pos=%.4f in %.1fs", args.dataset, W.shape[0],
             graph["n_edges"], graph["positive_ratio"], time.perf_counter() - t0)

    scores = candidate_scores(graph, args.device, args.with_resistance, log)
    log.info("predeclared pool: %d candidates", len(scores))

    # ---- stage 1: screen (diagnostic only, no pre-filtering) ----------------
    h_true = label_homophily(W, y)
    finite = np.isfinite(h_true)
    screen = []
    for name, s in scores.items():
        m = np.isfinite(s) & finite
        rho, pval = spearmanr(s[m], h_true[m])
        screen.append({"candidate": name, "spearman_rho": float(rho), "p": float(pval)})
    screen.sort(key=lambda r: -abs(r["spearman_rho"]))
    rho_of = {r["candidate"]: r["spearman_rho"] for r in screen}

    print("\n=== stage 1: label-free score vs true label homophily (diagnostic) ===")
    print("{:<26s}{:>10s}{:>12s}{:>16s}".format("candidate", "rho", "p", "a-priori sign"))
    for r in screen:
        agree = "agrees" if r["spearman_rho"] >= 0 else "DISAGREES"
        print("{:<26s}{:>10.4f}{:>12.3g}{:>16s}".format(
            r["candidate"], r["spearman_rho"], r["p"], agree))

    # ---- stage 2: evaluate EVERY candidate, both orientations ---------------
    conditions: dict[str, np.ndarray] = {"uniform": uniform_assignment(W.shape[0], 3)}
    oracle = oracle_bucket_assignment(W, y, adaptive=True)
    conditions["oracle"] = oracle
    conditions["random"] = random_assignment(oracle, rng=0)
    for name, s in scores.items():
        conditions[f"{name}::as_defined"] = bucket_from_score(s)
        if rho_of[name] < 0:
            # Only the flipped variant differs; skip duplicates otherwise.
            conditions[f"{name}::label_oriented"] = bucket_from_score(-s)

    log.info("evaluating %d conditions x %d splits x %d seeds", len(conditions),
             args.splits, args.seeds)
    t0 = time.perf_counter()
    split_means: dict[str, dict[int, float]] = {c: {} for c in conditions}
    for split in range(args.splits):
        train, test = train_test_split(y, split, 0)
        for cond, a in conditions.items():
            accs = []
            for seed in range(args.seeds):
                tr, te = train_test_split(y, split, seed)
                experts = fit_experts(X, y, a, tr)
                acc, _ = mixture_accuracy(X, y, a, experts, te)
                accs.append(acc)
            split_means[cond][split] = float(np.mean(accs))
        log.info("  split %d/%d done (%.0fs)", split + 1, args.splits,
                 time.perf_counter() - t0)

    def cmp(cond: str, ref: str, name: str):
        return evaluate_comparison(split_means[cond], split_means[ref], name)

    # Holm across the entire DECLARED as_defined family: known in advance.
    as_def = [c for c in conditions if c.endswith("::as_defined")]
    comps = [cmp(c, "random", c.replace("::", "_") + "_vs_random") for c in as_def]
    corrected = holm_bonferroni([c.p_wilcoxon for c in comps], 0.05)
    for c, thresh in zip(comps, corrected):
        c.significant = bool(c.mean_diff > 0 and c.p_wilcoxon < thresh)

    label_oriented = {}
    for c in conditions:
        if c.endswith("::label_oriented"):
            r = cmp(c, "random", c.replace("::", "_") + "_vs_random")
            label_oriented[c] = r

    oracle_cmp = cmp("oracle", "random", "oracle_vs_random")

    rows = []
    for key, c, thresh in zip(as_def, comps, corrected):
        base = key[: -len("::as_defined")]
        lo = label_oriented.get(f"{base}::label_oriented")
        rows.append({
            "candidate": base,
            "spearman_rho": rho_of[base],
            "d2_as_defined_mean_diff": c.mean_diff,
            "d2_as_defined_p": c.p_wilcoxon,
            "d2_as_defined_dz": c.effect_size_dz,
            "d2_as_defined_holm_alpha": thresh,
            "d2_as_defined_significant": c.significant,
            "d2_label_oriented_mean_diff": lo.mean_diff if lo else None,
            "d2_label_oriented_p": lo.p_wilcoxon if lo else None,
            "d2_label_oriented_dz": lo.effect_size_dz if lo else None,
        })
    rows.sort(key=lambda r: -r["d2_as_defined_mean_diff"])

    print(f"\n=== stage 2: D2 vs random, 10 splits x 3 seeds, Holm over "
          f"{len(comps)} declared candidates ===")
    print("oracle ceiling D2 = {:+.5f} (dz {:.2f})".format(
        oracle_cmp.mean_diff, oracle_cmp.effect_size_dz))
    print("{:<26s}{:>11s}{:>10s}{:>8s}{:>7s}{:>11s}".format(
        "candidate", "D2(deploy)", "p", "dz", "sig", "D2(oriented)"))
    for r in rows[:20]:
        print("{:<26s}{:>+11.5f}{:>10.5f}{:>+8.2f}{:>7s}{:>+11.5f}".format(
            r["candidate"], r["d2_as_defined_mean_diff"], r["d2_as_defined_p"],
            r["d2_as_defined_dz"], "YES" if r["d2_as_defined_significant"] else "",
            r["d2_label_oriented_mean_diff"] or float("nan")))

    best_deploy = next((r for r in rows if r["d2_as_defined_significant"]), None)
    summary = {
        "config": {**vars(args), "eig_ks": list(EIG_KS)},
        "graph": {
            "n_nodes": graph["n_nodes"], "d_features": graph["d_features"],
            "n_edges": graph["n_edges"], "positive_ratio": graph["positive_ratio"],
            "source": graph["source"],
        },
        "n_candidates": len(scores),
        "holm_family_size": len(comps),
        "oracle_ceiling_d2": oracle_cmp.mean_diff,
        "oracle_ceiling_dz": oracle_cmp.effect_size_dz,
        "screen": screen,
        "results": rows,
        "best_deployable": best_deploy,
        "n_deployable_significant": sum(1 for r in rows if r["d2_as_defined_significant"]),
    }
    (out_dir / "candidates.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\ndeployable candidates with Holm-significant positive D2: "
          f"{summary['n_deployable_significant']}/{len(comps)}")
    print("wrote", out_dir / "candidates.json")


if __name__ == "__main__":
    main()

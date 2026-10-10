"""Protocol B on a real LLM attention graph: frozen routing-blind expert pool.

Why this script exists
----------------------
`run_sd_vs_routing.py` measures *routing stability* (does the label-free
assignment survive sparsification?). That is necessary but not sufficient: the
original SmolLM2 artifact's positive downstream result was retracted because
the experts had been fitted on the same routing they were evaluated under
(Protocol A self-routing).

Protocol B fixes the confound properly:

  1. The expert pool is fitted ONCE, on the DENSE graph, from a k-means
     partition of the node features. It never sees labels and never sees the
     sparsified graph, so it cannot adapt to a degraded router.
  2. Experts are fitted per pool column on dense-graph training nodes and then
     frozen.
  3. Only the ROUTING varies across conditions: for each sparsified graph we
     recompute the label-free structural routing, then evaluate the *same*
     frozen experts under that routing.

So any accuracy change across conditions is attributable to the routing change,
which is what the spectral-distortion claim is about.

Feature mode
------------
Default `--feature-mode per_layer` gives one feature column per transformer
layer. The legacy `agg` mode yields only 3 columns (mean/max/std of layer
entropy), and a classifier over 29 passage labels from 3 features is near
chance regardless of routing, which would make this arm uninformative.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

# Import this package before `transformers`: importing transformers first and
# then this package segfaults (0xC0000005) on this machine. run_real_attention.py
# avoids it only by importing transformers lazily inside main().
from spectral_distillation.src.distortion import (
    connectivity_report,
    low_frequency_distortion,
    routing_survival,
    spectral_distortion_profile,
)
from spectral_distillation.src.llm_export import (
    build_attention_graph,
    export_attention,
    split_dataset,
)
from spectral_distillation.src.learned_router import uniform_ensemble_accuracy
from spectral_distillation.src.laplacian import compute_laplacian
from spectral_distillation.src.mixture import (
    EXPERT_FACTORIES,
    fit_experts,
    mixture_accuracy,
    train_test_split,
)
from spectral_distillation.src.router_protocol import (
    label_free_assignment,
    oracle_bucket_assignment,
    random_assignment,
)
from spectral_distillation.src.baselines import degree_sparsify, random_sparsify
from spectral_distillation.src.sparsifier import spectral_sparsify
from spectral_distillation.src.utils import ensure_dir, get_logger, set_seed

METHODS = {"er": spectral_sparsify, "random": random_sparsify, "degree": degree_sparsify}
N_EXPERTS = 3


def routing_from_graph(W, X, n_experts, epochs, seed, device):
    """Label-free routing recomputed from this condition's own graph.

    Uses the D2-fixed proxy_quantile/eig_nb_sim strategy rather than the legacy
    joint KMeans partition; see docs/d2_router_fix.md.
    """
    assign, _ = label_free_assignment(
        W, X, n_experts=n_experts, epochs=epochs, seed=seed, device=device,
        strategy="proxy_quantile", order_feature="eig_nb_sim",
    )
    return assign


def expert_ens_accuracy_pool(X, y, pool, te):
    """Uniform ensemble over the frozen pool: routing-blind reference."""
    return uniform_ensemble_accuracy(X, y, pool, te)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", required=True)
    p.add_argument("--text-file", default="data/erp_attention/corpus_scaled.txt")
    p.add_argument("--feature-mode", default="per_layer",
                   choices=["agg", "per_layer"])
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--max-passages", type=int, default=None)
    p.add_argument("--max-nodes", type=int, default=2000,
                   help="strided subsample; dense eigendecomposition is O(n^3)")
    p.add_argument("--top-k", type=int, default=32,
                   help="per-node attention cutoff; must match "
                        "configs/default.yaml (32) or the SD numbers will not "
                        "be comparable with run_sd_vs_routing, which reads this "
                        "from config. top_k=None keeps the fully dense "
                        "attention matrix and yields a ~8x denser graph.")
    p.add_argument("--retentions", type=float, nargs="+",
                   default=[1.0, 0.5, 0.25, 0.1, 0.05])
    p.add_argument("--methods", nargs="+", default=["random", "degree", "er"])
    p.add_argument("--n-experts", type=int, default=N_EXPERTS)
    p.add_argument("--expert", default="logistic", choices=["logistic", "mlp"],
                   help="per-row expert head; gnn needs a graph and is not "
                        "meaningful here because experts are frozen on the "
                        "dense graph while routing changes")
    p.add_argument("--router-epochs", type=int, default=50)
    p.add_argument("--splits", type=int, default=5)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--random-seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="logs/llm_protocol_b")
    args = p.parse_args()

    log = get_logger("llm_protocol_b")
    set_seed(args.random_seed)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    text = Path(args.text_file).read_text(encoding="utf-8")
    max_tokens = args.max_tokens
    if max_tokens is None:
        max_tokens = 480
    passages = split_dataset(text, max_tokens=max_tokens)
    if args.max_passages:
        passages = passages[: args.max_passages]
    log.info("exporting %d passages (feature_mode=%s)", len(passages),
             args.feature_mode)

    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, output_attentions=True, attn_implementation="eager",
        dtype=torch.float32)
    records = export_attention(model, tok, passages, max_tokens=max_tokens,
                              device=args.device)
    graph = build_attention_graph(records, how="mean",
                                  max_edges_per_node=args.top_k,
                                  feature_mode=args.feature_mode)
    W_full, X_full, y = graph["W"], graph["features"], graph["y"]
    log.info("dense graph n=%d edges=%d d_features=%d labels=%d",
             W_full.shape[0], int(np.count_nonzero(W_full) // 2),
             X_full.shape[1], len(np.unique(y)))

    # Strided subsample keeps every passage block represented; slicing would
    # keep only the first few passages and their labels.
    if args.max_nodes and W_full.shape[0] > args.max_nodes:
        idx = np.unique(np.linspace(0, W_full.shape[0] - 1, args.max_nodes)
                        .round().astype(int))
        W = W_full[np.ix_(idx, idx)]
        X, y = X_full[idx], y[idx]
        log.warning("subsampled %d -> %d nodes", W_full.shape[0], idx.size)
    else:
        W = W_full
        X = X_full
    n = W.shape[0]

    evals_ref = np.sort(np.linalg.eigvalsh(compute_laplacian(W)))

    # ---------------- Protocol B: one frozen, routing-blind pool --------------
    from sklearn.cluster import KMeans

    set_seed(args.random_seed)
    # Routing-blind: k-means on features only. No labels, no graph.
    pool = KMeans(n_clusters=args.n_experts, n_init=10,
                  random_state=args.random_seed).fit_predict(X)
    pool_oh = np.zeros((n, args.n_experts))
    pool_oh[np.arange(n), pool] = 1.0
    log.info("frozen routing-blind pool: %d experts, sizes=%s",
             args.n_experts, np.bincount(pool).tolist())

    oracle = oracle_bucket_assignment(W, y, adaptive=True)
    refs = {
        "oracle": oracle,
        "random": random_assignment(oracle, rng=args.random_seed),
    }

    conditions: list[dict] = []

    def evaluate(Wc, method, retention):
        """Protocol B accuracy for one condition, plus its SD diagnostics."""
        routing = routing_from_graph(Wc, X, args.n_experts, args.router_epochs,
                                     args.random_seed, args.device)
        accs: dict[str, list[float]] = {}
        for split in range(args.splits):
            for seed in range(args.seeds):
                train, te = train_test_split(y, split, seed)
                # Frozen once per (split, seed) from the DENSE pool. Every
                # condition below reuses these exact experts, so differences
                # across conditions come only from the routing.
                frozen = fit_experts(X, y, pool_oh, train,
                                     expert_factory=EXPERT_FACTORIES[args.expert])
                for rname, r in [("label_free", routing),
                                 ("oracle", refs["oracle"]),
                                 ("random", refs["random"])]:
                    acc, _ = mixture_accuracy(X, y, r, frozen, te)
                    accs.setdefault(rname, []).append(float(acc))
                accs.setdefault("ensemble_pool", []).append(
                    expert_ens_accuracy_pool(X, y, frozen, te))
        L_pt = compute_laplacian(Wc)
        lf = low_frequency_distortion(None, L_pt, ref_eigenvalues=evals_ref)
        conn = connectivity_report(None, L_pt, ref_eigenvalues=evals_ref)
        row = {"method": method, "retention": retention,
               "n_edges_kept": int(np.count_nonzero(Wc) // 2),
               "n_edges_orig": int(np.count_nonzero(W) // 2),
               "protocol_b_accuracy": {k: float(np.mean(v))
                                       for k, v in accs.items()},
               "protocol_b_accuracy_sd": {k: float(np.std(v))
                                          for k, v in accs.items()},
               "legacy_SD": float(spectral_distortion_profile(
                   None, L_pt, ref_eigenvalues=evals_ref)["SD"]),
               "low_frequency_error": float(lf["low_freq_mean_error"]),
               }
        row["n_isolated_perturbed"] = int(conn["n_isolated_perturbed"])
        row["routing_survival"] = float(routing_survival(
            lf["low_freq_mean_error"], conn["n_isolated_perturbed"], n))
        conditions.append(row)
        log.info("[%s r=%.2f] kept=%d lowfreq=%.5f iso=%d surv=%.3f "
                 "acc(label_free)=%.4f acc(oracle)=%.4f",
                 method, retention, row["n_edges_kept"],
                 row["low_frequency_error"], row["n_isolated_perturbed"],
                 row["routing_survival"],
                 row["protocol_b_accuracy"]["label_free"],
                 row["protocol_b_accuracy"]["oracle"])

    evaluate(W, "dense", 1.0)
    for method in args.methods:
        fn = METHODS[method]
        for r in args.retentions:
            if r == 1.0:
                # retention 1.0 is the identity; already recorded as dense.
                continue
            budget = max(1, int(round(r * np.count_nonzero(W) // 2)))
            Wc = (spectral_sparsify(W, budget, sample=True,
                                    seed=args.random_seed)
                  if method == "er" else fn(W, budget, seed=args.random_seed))
            evaluate(Wc, method, r)

    # ---------------- Does distortion predict accuracy loss? ----------------
    dense_acc = conditions[0]["protocol_b_accuracy"]["label_free"]
    for row in conditions:
        row["accuracy_drop_vs_dense"] = (dense_acc
                                         - row["protocol_b_accuracy"]["label_free"])
    lf = np.array([r["low_frequency_error"] for r in conditions])
    surv = np.array([r["routing_survival"] for r in conditions])
    drop = np.array([r["accuracy_drop_vs_dense"] for r in conditions])
    corr = {}
    for nm, v in (("low_frequency_error", lf), ("routing_survival", surv)):
        if len(v) >= 4 and np.ptp(drop) > 0 and np.ptp(v) > 0:
            s = spearmanr(v, drop)
            corr[nm] = {"rho": float(s.statistic), "p": float(s.pvalue)}
        else:
            corr[nm] = {"rho": None, "p": None}

    out = Path(args.out)
    ensure_dir(out)
    payload = {"graph": {"source": args.model_id, "n_nodes": int(n),
                         "n_features": int(X.shape[1]),
                         "n_labels": int(len(np.unique(y))),
                         "feature_mode": args.feature_mode,
                         "passages": len(passages)},
               "protocol": "B (frozen routing-blind k-means pool)",
               "n_experts": args.n_experts,
               "dense_label_free_accuracy": dense_acc,
               "conditions": conditions,
               "sd_vs_accuracy_drop": corr}
    dest = out / "llm_protocol_b.json"
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("=== SD vs Protocol-B accuracy drop ===")
    for nm, c in corr.items():
        if c["rho"] is None:
            log.info("  %-22s n/a", nm)
        else:
            log.info("  %-22s rho=%+.3f (p=%.3g)", nm, c["rho"], c["p"])
    log.info("wrote %s", dest)


if __name__ == "__main__":
    main()
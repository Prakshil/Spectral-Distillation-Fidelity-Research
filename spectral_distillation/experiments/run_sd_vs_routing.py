"""Does spectral distortion actually predict routing-relevant information loss?

This is the experiment the thesis needs and it had never been run. The two halves
of the claim were developed separately:

- SD was measured on fraud graphs and small synthetic graphs, where the headline
  scalar saturated at exactly 1.0 for every sparsified point and carried no
  information.
- Routing was measured on real SmolLM2 attention, but that artifact contains no SD
  number at all.

So nothing tested the actual claim. This runner puts both on the same graph at the
same budget and asks whether low-frequency SD tracks what the router loses.

The primary outcome is ``assignment_stability``: the agreement between the
label-free router fitted on the *dense* graph and the same router fitted on the
*sparse* graph. That is the quantity the thesis is about -- routing-relevant
information loss -- and unlike accuracy it varies continuously with the budget,
so there is something to correlate against. Accuracy is also reported, via
protocol B, but protocol B is flat by construction across the fraud graphs, so it
can only falsify the claim rather than confirm it.

Every point reports the legacy scalar alongside the new one, because the contrast
is itself a result: the legacy value is pinned at a rail and would rank methods
identically.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigvalsh
from scipy.stats import pearsonr, spearmanr

from spectral_distillation.src.baselines import degree_sparsify, random_sparsify
from spectral_distillation.src.distortion import (
    connectivity_report,
    davis_kahan_bound,
    low_frequency_distortion,
    routing_survival,
    spectral_distortion_profile,
)
from spectral_distillation.src.laplacian import compute_laplacian
from spectral_distillation.src.mixture import (
    fit_experts,
    mixture_accuracy,
    train_test_split,
)
from spectral_distillation.src.router_protocol import (
    label_free_assignment,
    oracle_bucket_assignment,
    random_assignment,
    uniform_assignment,
)
from spectral_distillation.src.sparsifier import spectral_sparsify
from spectral_distillation.src.utils import ensure_dir, load_yaml_config

log = logging.getLogger("sd_vs_routing")

METHODS = {"er": spectral_sparsify, "random": random_sparsify, "degree": degree_sparsify}


def _agreement(a: np.ndarray, b: np.ndarray) -> float:
    return float((np.argmax(a, axis=1) == np.argmax(b, axis=1)).mean())


def _label_free(W: np.ndarray, X: np.ndarray, args, seed: int | None = None) -> np.ndarray:
    out, _ = label_free_assignment(
        W, X,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        epochs=args.router_epochs,
        seed=args.router_seed if seed is None else seed,
        device=args.device,
        strategy=args.strategy,
        order_feature=args.order_feature,
        spectral_k=args.spectral_k,
    )
    return out


def _collapse_flag(assign: np.ndarray, threshold: float = 0.90) -> bool:
    """True when the router put almost every node in one expert.

    A single k-means fit can collapse, and that is a property of the optimiser,
    not of the graph. Without this flag a collapsed run reads as catastrophic
    "routing-relevant information loss" even when the spectrum is intact, which
    is exactly the confound the first Amazon sweep hit.
    """
    usage = np.bincount(np.argmax(assign, axis=1),
                        minlength=assign.shape[1]).astype(float)
    usage /= max(1.0, usage.sum())
    return bool(usage.max() >= threshold)


def _load_graph(args):
    if args.dataset == "llm":
        # Import order matters here: importing transformers *before* this package
        # segfaults on this environment (0xC0000005) during the second import,
        # while this package first is stable. run_real_attention.py only avoids
        # it by importing transformers lazily inside main().
        from spectral_distillation.src.llm_export import (
            build_attention_graph,
            export_attention,
            split_dataset,
        )
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        cfg = load_yaml_config(args.config)
        model_id = args.model_id or cfg.get("llm", {}).get("model_id")
        max_tokens = args.max_tokens or cfg.get("llm", {}).get("max_tokens", 480)
        top_k = args.top_k or cfg.get("llm", {}).get("top_k", 32)
        text = Path(args.text_file).read_text(encoding="utf-8")
        passages = split_dataset(text, max_tokens=max_tokens)
        if args.max_passages and len(passages) > args.max_passages:
            passages = passages[: args.max_passages]
        log.info("loaded %d passages from %s", len(passages), args.text_file)
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        # sdpa silently discards attention weights, so output_attentions=True
        # yields nothing and the export fails with "model returned no attention".
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            output_attentions=True,
            attn_implementation="eager",
            torch_dtype=torch.float32 if args.device == "cpu" else torch.float16,
        )
        records = export_attention(model, tok, passages, max_tokens=max_tokens,
                                   device=args.device)
        graph = build_attention_graph(records, how="mean", max_edges_per_node=top_k)
        graph["source"] = model_id
    else:
        from spectral_distillation.src.real_fraud import load_real_fraud

        graph = load_real_fraud(args.dataset, max_nodes=args.max_nodes)
        graph["source"] = args.dataset

    # Cache the dense graph before subsampling so downstream experiments (the
    # Protocol B accuracy arm) can reuse the export instead of paying another
    # full forward pass per passage. Stored sparse: the attention graph is
    # block-diagonal by passage, so dense would be ~500MB for 11.5k nodes.
    if args.save_graph:
        from scipy.sparse import csr_matrix, save_npz
        ensure_dir(Path(args.save_graph).parent)
        save_npz(args.save_graph, csr_matrix(graph["W"]))
        np.savez_compressed(str(Path(args.save_graph).with_suffix("")) + "_meta.npz",
                            features=graph["features"], y=graph["y"])
        log.info("saved dense graph -> %s", args.save_graph)

    # Dense eigendecomposition is O(n^3), so the sweep subsamples. Striding rather
    # than slicing matters: the attention graph is block-diagonal by passage, so
    # taking the first n rows would keep only the first few passages and throw
    # away every later block (and their labels), leaving the router with almost
    # no classes to separate. Striding keeps every block represented.
    if args.max_spectral_nodes and graph["W"].shape[0] > args.max_spectral_nodes:
        total = graph["W"].shape[0]
        n = args.max_spectral_nodes
        idx = np.unique(np.linspace(0, total - 1, n).round().astype(int))
        log.warning("subsampling %d -> %d nodes (strided) for the spectral sweep",
                    total, idx.size)
        graph["W"] = graph["W"][np.ix_(idx, idx)]
        graph["features"] = graph["features"][idx]
        graph["y"] = graph["y"][idx]
        graph["n_nodes"] = int(idx.size)
    return graph


def _protocol_b_accuracy(graph, assign, splits, seeds) -> dict:
    """Protocol B: routing-blind pool, so experts never see their own router."""
    X, y = graph["features"], graph["y"]
    n_routers = assign.shape[1]
    unif = uniform_assignment(X.shape[0], n_routers)
    out = {"label_free": [], "random": [], "uniform": []}
    for cond, a in (("label_free", assign), ("random", random_assignment(
            assign, rng=1)), ("uniform", unif)):
        for split in range(splits):
            for seed in seeds:
                train, test = train_test_split(y, split, seed)
                experts = fit_experts(X, y, a, train)
                acc, _ = mixture_accuracy(X, y, a, experts, test)
                out[cond].append(float(acc))
    means = {k: float(np.mean(v)) for k, v in out.items() if v}
    means["label_free_vs_uniform"] = means.get("label_free", np.nan) - means.get(
        "uniform", np.nan)
    return means


def run_point(graph, ref_evals, method, retention, dense_assign_by_seed,
              oracle, args) -> dict:
    W = graph["W"]
    n_orig = int(np.count_nonzero(W) // 2)
    budget = int(round(retention * n_orig))
    t0 = time.perf_counter()

    if budget >= n_orig:
        W_pt = W
    elif method == "er":
        W_pt = spectral_sparsify(W, budget, sample=True, seed=args.random_seed,
                                 resistance_matrix=args._resistance)
    else:
        W_pt = METHODS[method](W, budget, seed=args.random_seed)

    L_pt = compute_laplacian(W_pt)
    legacy = spectral_distortion_profile(None, L_pt, ref_eigenvalues=ref_evals)
    lf = low_frequency_distortion(None, L_pt, ref_eigenvalues=ref_evals)
    conn = connectivity_report(None, L_pt, ref_eigenvalues=ref_evals)
    surv = routing_survival(lf["low_freq_mean_error"],
                            conn["n_isolated_perturbed"], int(graph["n_nodes"]))

    # Stability is measured per seed against the dense router fitted with that
    # same seed, so seed-to-seed clustering variance cancels instead of being
    # charged to the graph. Averaging over seeds is what separates a real
    # degradation trend from a single collapsed k-means run.
    stab_per_seed, oag_per_seed, collapsed = [], [], 0
    for seed in args.stability_seeds:
        dense_s = dense_assign_by_seed[seed]
        sparse_s = _label_free(W_pt, graph["features"], args, seed=seed)
        stab_per_seed.append(_agreement(dense_s, sparse_s))
        oag_per_seed.append(_agreement(oracle, sparse_s))
        if _collapse_flag(sparse_s):
            collapsed += 1
    stability = float(np.mean(stab_per_seed))
    oracle_ag = float(np.mean(oag_per_seed))

    log.info("[%s r=%.2f] kept=%d/%d legacy_SD=%.4f lowfreq=%.5f isolated=%d "
             "survival=%.4f stability=%.4f oracle_ag=%.4f collapsed=%d/%d (%.1fs)",
             method, retention, int(np.count_nonzero(W_pt) // 2), n_orig,
             legacy["SD"], lf["low_freq_mean_error"], conn["n_isolated_perturbed"],
             surv, stability, oracle_ag, collapsed, len(args.stability_seeds),
             time.perf_counter() - t0)

    return {
        "method": method,
        "retention": float(retention),
        "n_edges_kept": int(np.count_nonzero(W_pt) // 2),
        "n_edges_orig": n_orig,
        "fraction_kept": float(np.count_nonzero(W_pt) // 2 / max(1, n_orig)),
        "legacy_SD": legacy["SD"],
        "legacy_min_eigengap": legacy["min_eigengap"],
        "legacy_fragility": legacy["fragility_score"],
        "low_frequency": lf,
        "connectivity": conn,
        "routing_survival": surv,
        "assignment_stability": stability,
        "assignment_stability_per_seed": stab_per_seed,
        "oracle_agreement": oracle_ag,
        "router_collapse_seeds": int(collapsed),
        "davis_kahan_bound": davis_kahan_bound(
            legacy["median_relative_error"], legacy["min_eigengap"]),
        "protocol_b_accuracy": {},
    }


def summarise(points: list[dict]) -> dict:
    """Correlate spectral distortion against each routing outcome."""
    def col(k, rows=None):
        rows = points if rows is None else rows
        return np.array([p[k] for p in rows], dtype=float)

    lf_err = col("low_frequency_error")
    legacy = col("legacy_SD")
    stab = col("assignment_stability")
    out = {"n_points": len(points), "legacy_SD_distinct_values":
           int(np.unique(np.round(legacy, 6)).size)}

    for name, y in (("assignment_stability", stab),
                    ("oracle_agreement", col("oracle_agreement"))):
        sp = spearmanr(lf_err, y)
        pe = pearsonr(lf_err, y)
        out[f"spearman_lowfreq_vs_{name}"] = float(sp.statistic)
        out[f"pvalue_lowfreq_vs_{name}"] = float(sp.pvalue)
        out[f"pearson_lowfreq_vs_{name}"] = float(pe.statistic)
        # Legacy control: the same correlation using the saturated scalar.
        if np.unique(np.round(legacy, 6)).size > 1:
            spl = spearmanr(legacy, y)
            out[f"spearman_legacy_vs_{name}"] = float(spl.statistic)
        else:
            out[f"spearman_legacy_vs_{name}"] = None

    # Monotonicity within each sparsifier: does stability fall as the budget falls?
    mono = {}
    for m in sorted({p["method"] for p in points}):
        rows = sorted([p for p in points if p["method"] == m],
                      key=lambda p: -p["retention"])
        seq = [round(p["assignment_stability"], 4) for p in rows]
        mono[m] = {
            "stability_by_descending_budget": seq,
            "monotone": all(seq[i] >= seq[i + 1] - 1e-9 for i in range(len(seq) - 1)),
        }
    out["per_method_monotonicity"] = mono
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="amazon",
                   choices=["amazon", "tolokers", "yelpchi", "llm"])
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--model-id", default=None)
    p.add_argument("--text-file", default="data/erp_attention/corpus.txt")
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--max-passages", type=int, default=None,
                   help="cap passages exported; each one is a full forward pass")
    p.add_argument("--save-graph", default=None,
                   help="cache the dense (pre-subsample) graph as sparse npz "
                        "for reuse by downstream experiments")
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--max-nodes", type=int, default=None,
                   help="cap node count when loading a fraud graph")
    p.add_argument("--max-spectral-nodes", type=int, default=3000,
                   help="subsample so the O(n^3) eigendecomposition stays tractable")
    p.add_argument("--methods", nargs="+", default=["random", "degree", "er"])
    p.add_argument("--retentions", type=float, nargs="+",
                   default=[1.0, 0.50, 0.25, 0.10, 0.05])
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--router-epochs", type=int, default=60)
    p.add_argument("--router-seed", type=int, default=0)
    p.add_argument("--stability-seeds", type=int, nargs="+", default=[0, 1, 2],
                   help="router seeds for stability; each sparse fit is compared "
                        "against the dense fit at the same seed")
    p.add_argument("--random-seed", type=int, default=1)
    p.add_argument("--strategy", default="kmeans")
    p.add_argument("--order-feature", default="hx")
    p.add_argument("--spectral-k", type=int, default=0)
    p.add_argument("--top-k-router", dest="top_k_router", type=int, default=32)
    p.add_argument("--adaptive-oracle", action="store_true")
    p.add_argument("--budget-oracle", action="store_true")
    p.add_argument("--splits", type=int, default=2)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--protocol-b-accuracy", action="store_true",
                   help="also fit protocol-B experts at every point (slow)")
    p.add_argument("--skip-sparse-router", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="logs/sd_vs_routing")

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = p.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    graph = _load_graph(args)
    log.info("loaded %s: n=%d edges=%d features=%d (%.1fs)", graph["source"],
             graph["W"].shape[0], int(np.count_nonzero(graph["W"]) // 2),
             graph["features"].shape[1], time.perf_counter() - t0)

    from spectral_distillation.src.effective_resistance import (
        effective_resistance_exact,
    )

    args._resistance = effective_resistance_exact(compute_laplacian(graph["W"]))

    ref_evals = np.sort(eigvalsh(compute_laplacian(graph["W"])))
    dense_lf = low_frequency_distortion(
        None, compute_laplacian(graph["W"]), ref_eigenvalues=ref_evals)
    log.info("dense reference lambda_max=%.4f components=%d",
             ref_evals[-1], dense_lf["n_matched_modes_ref"])

    t0 = time.perf_counter()
    dense_assign_by_seed = {
        seed: _label_free(graph["W"], graph["features"], args, seed=seed)
        for seed in args.stability_seeds
    }
    oracle = oracle_bucket_assignment(graph["W"], graph["y"],
                                      adaptive=args.adaptive_oracle,
                                      budget_normalized=args.budget_oracle,
                                      top_k=args.top_k_router)
    log.info("dense router (%d seeds) + oracle fitted in %.1fs",
             len(args.stability_seeds), time.perf_counter() - t0)

    points = []
    for method in args.methods:
        for retention in args.retentions:
            pt = run_point(graph, ref_evals, method, retention,
                           dense_assign_by_seed, oracle, args)
            pt["low_frequency_error"] = pt["low_frequency"]["low_freq_mean_error"]
            points.append(pt)
            (out_dir / "points.json").write_text(
                json.dumps(points, indent=2, default=str), encoding="utf-8")

    summary = summarise(points)
    result = {
        "config": {k: str(v) for k, v in vars(args).items() if k != "_resistance"},
        "graph": {"source": graph["source"], "n_nodes": int(graph["W"].shape[0]),
               "n_features": int(np.shape(graph["features"])[1]),
               "n_labels": int(len(np.unique(graph["y"]))),
                  "n_edges": int(np.count_nonzero(graph["W"]) // 2)},
        "dense_reference": dense_lf,
        "points": points,
        "summary": summary,
    }
    (out_dir / "sd_vs_routing.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8")

    log.info("=== SD vs routing ===")
    log.info("legacy SD distinct values: %d of %d points",
             summary["legacy_SD_distinct_values"], summary["n_points"])
    log.info("Spearman(lowfreq, stability)  = %+.3f (p=%.3g)",
             summary["spearman_lowfreq_vs_assignment_stability"],
             summary["pvalue_lowfreq_vs_assignment_stability"])
    log.info("Spearman(lowfreq, oracle_ag)  = %+.3f",
             summary["spearman_lowfreq_vs_oracle_agreement"])
    log.info("Spearman(legacy,  stability)  = %s",
             summary["spearman_legacy_vs_assignment_stability"])
    for m, d in summary["per_method_monotonicity"].items():
        log.info("  %-7s monotone=%s %s", m, d["monotone"],
                 d["stability_by_descending_budget"])
    log.info("wrote %s", out_dir / "sd_vs_routing.json")


if __name__ == "__main__":
    main()
"""Real-attention router-usage ablation (regenerates ``docs/real_attention_results.md`` Sec. 6).

The label-free structural router is a KMeans over the five standardized fields
``{snf, hx, spectral_ratio, log_degree, clustering}``. This script:

  1. builds the SmolLM2 attention graph at each requested ``top-k``;
  2. computes the full label-free assignment and one-field-dropped variants;
  3. evaluates **D2 = mean(test accuracy(label_free) - test accuracy(random))**
     against TWO random baselines:
        * ``rand_oracle``  -- random matched to the ORACLE usage (recorded convention);
        * ``rand_router``  -- random matched to the ROUTER usage (the fair control);
  4. writes ``{out}/real_attention/ablation.json`` and ``control.json``.

Motivation: the committed Sec. 6 control artifact was produced by a one-off
script (never committed) whose "full" router was a KMeans over only the four
legacy fields {hx, spectral_ratio, log_degree, clustering} (``snf`` omitted),
which collapses to a degenerate 99%-in-one-expert partition. This script makes
the control reproducible under the CURRENT five-field router and re-tests the
claim that the recorded D2 gain is a usage-imbalance artifact.

Both ``random_assignment`` baselines are usage-matched (same expert histogram,
random per-node) -- that is the recorded convention -- so the only thing that
differs between the two baselines is *which* usage profile is matched.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from spectral_distillation.src.evaluation import run_protocol
from spectral_distillation.src.laplacian import compute_laplacian, eigh_symmetric
from spectral_distillation.src.llm_export import (
    build_attention_graph,
    export_attention,
    split_dataset,
)
from spectral_distillation.src.mixture import fit_experts, mixture_accuracy, train_test_split
from spectral_distillation.src.router_protocol import (
    oracle_bucket_assignment,
    random_assignment,
    structural_features,
)
from spectral_distillation.src.utils import device_config, ensure_dir, get_logger, set_seed

FIELD_NAMES = ("snf", "hx", "spectral_ratio", "log_degree", "clustering")
DROP_FIELDS = ("hx", "snf", "spectral_ratio", "log_degree", "clustering")


def router_assignment(
    W: np.ndarray,
    feats: np.ndarray,
    drop: str | None = None,
    n_experts: int = 3,
    seed: int = 0,
    order_col: int = 1,
) -> np.ndarray:
    """KMeans partition over structural fields, reordered by the ``hx`` column.

    Mirrors ``router_protocol.label_free_assignment(strategy="kmeans")`` but takes
    precomputed ``feats`` so the Laplacian eigendecomposition is reused across
    variants and a single field can be dropped. ``order_col=1`` is ``hx`` (the
    legacy default ordering field).
    """
    from sklearn.cluster import KMeans

    keep = list(range(feats.shape[1]))
    if drop is not None:
        keep = [i for i in keep if FIELD_NAMES[i] != drop]
    used = feats[:, keep]
    std = (used - used.mean(axis=0)) / (used.std(axis=0) + 1e-8)
    km = KMeans(n_clusters=n_experts, n_init=10, random_state=seed).fit(std)
    labels = km.labels_
    score = feats[:, order_col]
    group_score = [float(np.mean(score[labels == k])) for k in range(n_experts)]
    order = np.argsort(group_score)[::-1]
    relabel = np.zeros_like(labels)
    for new_k, old_k in enumerate(order):
        relabel[labels == old_k] = new_k
    assignment = np.zeros((W.shape[0], n_experts))
    assignment[np.arange(W.shape[0]), relabel] = 1.0
    return assignment


def cell_accuracy(graph, assignments, cond: str, split: int, seed: int) -> float:
    X, y = graph["features"], graph["y"]
    train, test = train_test_split(y, split, seed)
    experts = fit_experts(X, y, assignments[cond], train)
    acc, _ = mixture_accuracy(X, y, assignments[cond], experts, test)
    return float(acc)


def usage(a: np.ndarray) -> list[int]:
    return np.bincount(np.argmax(a, axis=1), minlength=3).tolist()


def run_one_k(graph, k: int, args, log) -> tuple[dict, dict]:
    W, X, y = graph["W"], graph["features"], graph["y"]
    log.info("k=%d: eigendecomposition + structural features", k)
    t0 = time.perf_counter()
    L = compute_laplacian(W)
    evals, V = eigh_symmetric(L, device=args.device)
    feats = structural_features(W, X, evals, V, spectral_k=args.spectral_k)
    log.info("k=%d: features %s in %.1fs", k, feats.shape, time.perf_counter() - t0)

    oracle = oracle_bucket_assignment(W, y, top_k=k)
    lf_full = router_assignment(W, feats, drop=None, seed=args.router_seed)
    drop_assignments = {
        f"drop_{f}": router_assignment(W, feats, drop=f, seed=args.router_seed)
        for f in DROP_FIELDS
    }
    rand_oracle = random_assignment(oracle, rng=args.random_seed)
    rand_router = random_assignment(lf_full, rng=args.random_seed)

    assignments = {"rand_oracle": rand_oracle, "rand_router": rand_router, "lf_full": lf_full}
    assignments.update({k2: v for k2, v in drop_assignments.items() if k2 != "lf_full"})

    for name, a in assignments.items():
        log.info("  k=%d %-16s usage=%s", k, name, usage(a))

    pairs = []
    for name in ["lf_full", *drop_assignments.keys()]:
        pairs.append(("rand_oracle", name))
        pairs.append(("rand_router", name))

    results = run_protocol(
        run_fn=lambda cond, split, seed: cell_accuracy(graph, assignments, cond, split, seed),
        n_splits=args.splits,
        n_seeds=args.seeds,
        primary_pairs=pairs,
        progress=not args.no_progress,
        progress_desc=f"ablation k={k}",
    )
    comparisons = {c.name: c.to_dict() for c in results.comparisons}

    ablation_entry = {
        "full": {
            "D2": comparisons["lf_full_vs_rand_oracle"]["mean_diff"],
            "D2_p": comparisons["lf_full_vs_rand_oracle"]["p_wilcoxon"],
            "D2_dz": comparisons["lf_full_vs_rand_oracle"]["effect_size_dz"],
            "usage": usage(lf_full),
        }
    }
    for name in drop_assignments:
        key = name  # already "drop_*"
        ok = comparisons[f"{name}_vs_rand_oracle"]
        ablation_entry[key] = {
            "D2": ok["mean_diff"],
            "D2_p": ok["p_wilcoxon"],
            "D2_dz": ok["effect_size_dz"],
            "usage": usage(assignments[name]),
        }

    control_entry = {
        "k": k,
        "usage": {
            "oracle": usage(oracle),
            "label_free": usage(lf_full),
            "random_oracle_matched": usage(rand_oracle),
            "random_router_matched": usage(rand_router),
        },
        "vs_oracle_random": {
            "D2": comparisons["lf_full_vs_rand_oracle"]["mean_diff"],
            "p": comparisons["lf_full_vs_rand_oracle"]["p_wilcoxon"],
            "dz": comparisons["lf_full_vs_rand_oracle"]["effect_size_dz"],
        },
        "vs_usage_random": {
            "D2": comparisons["lf_full_vs_rand_router"]["mean_diff"],
            "p": comparisons["lf_full_vs_rand_router"]["p_wilcoxon"],
            "dz": comparisons["lf_full_vs_rand_router"]["effect_size_dz"],
        },
        "comparisons": comparisons,
    }
    return ablation_entry, control_entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-1.7B")
    parser.add_argument("--text-file", required=True)
    parser.add_argument("--max-tokens", type=int, default=480)
    parser.add_argument("--top-ks", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--splits", type=int, default=6)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--router-seed", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--spectral-k", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--out", default="logs/rk_ablation")
    args = parser.parse_args()

    args.device = device_config(args.device)
    log = get_logger("router_ablation", "INFO")
    set_seed(args.router_seed)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    text = Path(args.text_file).read_text(encoding="utf-8")
    passages = split_dataset(text, max_tokens=args.max_tokens)
    log.info("loaded %d passages from %s", len(passages), args.text_file)

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, output_attentions=True, torch_dtype=torch.float16
    )
    log.info("loaded model in %.1fs", time.perf_counter() - t0)
    records = export_attention(
        model, tokenizer, passages, max_tokens=args.max_tokens, device=args.device
    )
    log.info("exported attention in %.1fs", time.perf_counter() - t0)

    ablation: dict = {"device": args.device, "config": vars(args)}
    control: dict = {"config": vars(args)}
    for k in args.top_ks:
        graph = build_attention_graph(records, how="mean", max_edges_per_node=k)
        log.info(
            "k=%d graph: n=%d labels=%d", k, graph["n_nodes"], int(np.unique(graph["y"]).size)
        )
        ab, ctl = run_one_k(graph, k, args, log)
        ablation[str(k)] = ab
        control[str(k)] = ctl

    out_dir = ensure_dir(Path(args.out) / "real_attention")
    (out_dir / "ablation.json").write_text(
        json.dumps(ablation, indent=2, default=str), encoding="utf-8"
    )
    (out_dir / "control.json").write_text(
        json.dumps(control, indent=2, default=str), encoding="utf-8"
    )
    log.info("wrote %s", out_dir)

    print("\n=== Router-usage ablation (D2 = label_free - random) ===")
    for k in args.top_ks:
        ctl = control[str(k)]
        print(
            f"k={k}: usage lf={ctl['usage']['label_free']} oracle={ctl['usage']['oracle']}"
        )
        print(
            f"      D2 vs oracle-usage-matched random: "
            f"{ctl['vs_oracle_random']['D2']:+.4f} (p={ctl['vs_oracle_random']['p']:.4g})"
        )
        print(
            f"      D2 vs router-usage-matched random: "
            f"{ctl['vs_usage_random']['D2']:+.4f} (p={ctl['vs_usage_random']['p']:.4g})"
        )


if __name__ == "__main__":
    main()

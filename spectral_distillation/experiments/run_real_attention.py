"""Real-attention D1-D4 protocol (step 1 of the real-attention run).

Runs the *same* D1-D4 decision rules and dilution ladder that the synthetic
PC-1c-R pipeline uses, but on graphs distilled from real LLM attention
weights over real text (public-domain corpus by default). The oracle reads
real sentence-membership labels via homophily buckets; the label-free router
must recover the same routing from attention-entropy features and graph
structure alone, exactly as in the synthetic control. D4 (instrument valid on
the positive control) is re-estimated on a small PC-1c-R graph within the
same run so the instrument check stays honest.

Usage:
    python experiments/run_real_attention.py --model-id SMALL_OPEN_MODEL
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from spectral_distillation.src.evaluation import (
    combine_comparisons,
    run_protocol,
)
from spectral_distillation.src.llm_export import (
    build_attention_graph,
    export_attention,
    split_dataset,
)
from spectral_distillation.src.mixture import fit_experts, mixture_accuracy, train_test_split
from spectral_distillation.src.planted_control import generate_pc_graph
from spectral_distillation.src.router_protocol import (
    evaluate_decision_rules,
    label_free_assignment,
    oracle_bucket_assignment,
    random_assignment,
    uniform_assignment,
)
from spectral_distillation.src.utils import (
    device_config,
    ensure_dir,
    get_logger,
    load_yaml_config,
    set_seed,
)

CONDITIONS = ("oracle", "label_free", "random", "uniform")


def build_conditions(graph: dict, args) -> dict[str, np.ndarray]:
    W, X, y = graph["W"], graph["features"], graph["y"]
    oracle = oracle_bucket_assignment(W, y, adaptive=args.adaptive_oracle,
                                      budget_normalized=args.budget_oracle, top_k=args.top_k)
    uniform = uniform_assignment(W.shape[0], 3)
    random = random_assignment(oracle, rng=args.random_seed)
    label_free, _ = label_free_assignment(
        W,
        X,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        epochs=args.router_epochs,
        seed=args.router_seed,
        device=args.device,
    )
    return {
        "oracle": oracle,
        "label_free": label_free,
        "random": random,
        "uniform": uniform,
    }


def cell_accuracy(graph: dict, assignments: dict, condition: str, split: int, seed: int) -> float:
    X, y = graph["features"], graph["y"]
    train, test = train_test_split(y, split, seed)
    experts = fit_experts(X, y, assignments[condition], train)
    acc, _ = mixture_accuracy(X, y, assignments[condition], experts, test)
    return float(acc)


def run_protocol_on_graph(graph: dict, args, log) -> dict:
    log.info("computing condition assignments...")
    t0 = time.perf_counter()
    assignments = build_conditions(graph, args)
    log.info("assignments ready in %.1fs", time.perf_counter() - t0)
    for name, a in assignments.items():
        log.info("  %-10s usage=%s", name, np.bincount(np.argmax(a, axis=1), minlength=3).tolist())

    results = run_protocol(
        run_fn=lambda cond, split, seed: cell_accuracy(graph, assignments, cond, split, seed),
        n_splits=args.splits,
        n_seeds=args.seeds,
        progress=not args.no_progress,
        progress_desc="main protocol cells",
    )
    comparisons = combine_comparisons(results)
    pairs = {
        name: {
            "mean_diff": c.mean_diff,
            "ci_low": c.ci_low,
            "ci_high": c.ci_high,
            "p_wilcoxon": c.p_wilcoxon,
            "p_paired_t": c.p_paired_t,
            "effect_size_dz": c.effect_size_dz,
            "significant": c.significant,
        }
        for name, c in comparisons.items()
    }

    control = _positive_control_pairs(args, log)
    decisions = evaluate_decision_rules(comparisons=pairs, control_comparisons=control)

    record = {
        "config": args.__dict__,
        "graph": {
            "n_nodes": graph["W"].shape[0],
            "n_labels": int(np.unique(graph["y"]).size),
            "n_layers": graph["n_layers"],
            "n_passages": graph["passages"],
        },
        "assignments_usage": {
            name: np.bincount(np.argmax(a, axis=1), minlength=3).tolist()
            for name, a in assignments.items()
        },
        "findings": decisions.__dict__,
        "comparisons": pairs,
        "control": {"pc_n_nodes": _PC_N, **control},
    }
    return record


_PC_N = 2000


def _positive_control_pairs(args, log) -> dict:
    """D4 instrument check on a small PC-1c-R graph (oracle > uniform)."""
    from spectral_distillation.src.router_protocol import oracle_regime_assignment

    t0 = time.perf_counter()
    graph = generate_pc_graph(n_nodes=_PC_N, n_patches=30, regime_ratio=(0.4, 0.4, 0.2), seed=args.router_seed)
    oracle = oracle_regime_assignment(graph["regimes"], 3)
    uniform = uniform_assignment(_PC_N, 3)
    random = random_assignment(oracle, rng=args.random_seed)
    results = run_protocol(
        run_fn=lambda cond, split, seed: cell_accuracy(
            graph,
            {"oracle": oracle, "uniform": uniform, "random": random, "label_free": uniform},
            cond,
            split,
            seed,
        ),
        n_splits=min(args.splits, 6),
        n_seeds=min(args.seeds, 2),
        progress=not args.no_progress,
        progress_desc="D4 control cells",
    )
    combined = combine_comparisons(results)
    log.info("positive control (D4) ready in %.1fs", time.perf_counter() - t0)
    return {
        name: {
            "mean_diff": c.mean_diff,
            "p_wilcoxon": c.p_wilcoxon,
        }
        for name, c in combined.items()
    }


def run_dilution(graph: dict, args, log) -> list[dict]:
    """Reuse the dilution ladder on the real attention graph."""
    from spectral_distillation.src.planted_control import dilution_ladder

    if not args.no_progress:
        from tqdm import tqdm
        ladder = dilution_ladder(
            {**graph, "p_homophily": args.p_homophily},
            dilution_values=args.dilutions,
            seed=args.dilution_seed,
        )
        iterator = tqdm(ladder, desc="dilution ladder", unit="rung", ncols=100)
    else:
        ladder = dilution_ladder(
            {**graph, "p_homophily": args.p_homophily},
            dilution_values=args.dilutions,
            seed=args.dilution_seed,
        )
        iterator = ladder
    rows = []
    for g in iterator:
        dil = g["dilution"]
        t0 = time.perf_counter()
        assignments = build_conditions(g, args)
        results = run_protocol(
            run_fn=lambda cond, split, seed: cell_accuracy(g, assignments, cond, split, seed),
            n_splits=min(args.splits, 6),
            n_seeds=min(args.seeds, 2),
            progress=not args.no_progress,
            progress_desc=f"rung {dil:.2f} cells",
        )
        comparisons = combine_comparisons(results)
        oracle_gain = comparisons["oracle_vs_uniform"].mean_diff
        label_free_gain = comparisons["label_free_vs_random"].mean_diff
        log.info(
            "dilution=%.2f oracle_gain=%.4f label_free_gain=%.4f (%.1fs)",
            dil,
            oracle_gain,
            label_free_gain,
            time.perf_counter() - t0,
        )
        rows.append(
            {
                "dilution": dil,
                "oracle_gain": oracle_gain,
                "label_free_gain": label_free_gain,
                "elapsed_s": round(time.perf_counter() - t0, 1),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--text-file", default="data/erp_attention/corpus.txt")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--splits", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--router-epochs", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--p-homophily", type=float, default=0.5)
    parser.add_argument("--dilutions", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--router-seed", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--dilution-seed", type=int, default=3)
    parser.add_argument("--skip-dilution", action="store_true", help="skip the slow dilution ladder (D1-D4 verdict only)")
    parser.add_argument("--adaptive-oracle", action="store_true", help="use quantile-based homophily thresholds for the oracle buckets (default off: fixed 0.4/0.6)")
    parser.add_argument("--budget-oracle", action="store_true", help="normalize oracle homophily by the available same-sentence budget (improves k>=32)")
    parser.add_argument("--no-progress", action="store_true", help="disable tqdm progress bars")
    parser.add_argument("--out", default="logs")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    log = get_logger("real_attention", config.get("logging", {}).get("level", "INFO"))
    llm_cfg = config.get("llm_export", {})
    proto = config.get("protocol", {})

    args.model_id = args.model_id or llm_cfg.get("model_id")
    args.max_tokens = args.max_tokens or llm_cfg.get("max_tokens", 480)
    args.top_k = args.top_k or llm_cfg.get("top_k", 32)
    args.splits = args.splits or proto.get("n_splits", 10)
    args.seeds = args.seeds or proto.get("n_seeds", 3)
    args.router_epochs = args.router_epochs or 200
    args.hidden_dim = args.hidden_dim or 64
    args.n_layers = args.n_layers or 3
    args.device = device_config(args.device)
    if args.model_id is None:
        parser.error("--model-id is required (or set llm_export.model_id in config)")

    set_seed(args.router_seed)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        parser.error(f"transformers not installed: {exc}")

    text = Path(args.text_file).read_text(encoding="utf-8")
    passages = split_dataset(text, max_tokens=args.max_tokens)
    log.info("loaded %d passages from %s", len(passages), args.text_file)

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        output_attentions=True,
        torch_dtype=torch.float16,
    )
    log.info("loaded model %s in %.1fs", args.model_id, time.perf_counter() - t0)

    records = export_attention(model, tokenizer, passages, max_tokens=args.max_tokens, device=args.device)
    log.info("exported attention for %d passages in %.1fs", len(records), time.perf_counter() - t0)

    graph = build_attention_graph(records, how="mean", max_edges_per_node=args.top_k)
    log.info(
        "built real-attention graph: n=%d labels=%d layers=%d passages=%d",
        graph["n_nodes"],
        int(np.unique(graph["y"]).size),
        graph["n_layers"],
        graph["passages"],
    )

    record = run_protocol_on_graph(graph, args, log)
    if args.skip_dilution:
        record["dilution"] = None
        record["dilution_skipped"] = True
        log.info("SKIPPED dilution ladder (--skip-dilution); D1-D4 verdict computed above")
    else:
        record["dilution"] = run_dilution(graph, args, log)

    out_dir = ensure_dir(Path(args.out) / "real_attention")
    safe = args.model_id.replace("/", "--")
    out_path = out_dir / f"real_attention_{safe}.json"
    out_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    log.info("wrote %s", out_path)

    print("\n=== Real-attention router fidelity ===")
    print(f"{'comparison':<22}{'mean_diff':>10}{'p_wilcoxon':>12}{'d_z':>8}{'sig':>6}")
    for name, c in record["comparisons"].items():
        print(
            f"{name:<22}{c['mean_diff']:>10.4f}{c['p_wilcoxon']:>12.4g}{c['effect_size_dz']:>8.3f}"
            f"{'yes' if c['significant'] else 'no':>6}"
        )
    print("\nDecision rules:", record["findings"]["verdict"])
    print("\n=== Dilution ladder ===" + (" (SKIPPED)" if args.skip_dilution else ""))
    print(f"{'dilution':>10}{'oracle_gain':>14}{'label_free_gain':>16}")
    for row in record["dilution"] or []:
        print(
            f"{row['dilution']:>10.2f}{row['oracle_gain']:>14.4f}"
            f"{row['label_free_gain']:>16.4f}"
        )


if __name__ == "__main__":
    main()
"""Reproduce all Phase 3 figures from saved JSON/CSV logs.

Reads `logs/protocol/protocol_results.json`, `logs/dilution/dilution_curve.json`
and `logs/ablation/ablation.csv` and writes PNGs next to them:

    logs/protocol/protocol_findings.png       decision-rule mean diffs + frozen gates
    logs/dilution/dilution_curve.png          label-free vs oracle gain vs dilution
    logs/ablation/ablation.png                feature ablation and oracle delta sweep

Mirrors the Makefile `figures` target. Fails fast only on the first present-but-
malformed log; missing logs are skipped with a warning.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from spectral_distillation.src.utils import ensure_dir, get_logger


def _rule_label(key: str) -> str:
    return {
        "oracle_vs_uniform": "D1\noracle > uniform",
        "label_free_vs_random": "D2\nlabel-free > random",
        "oracle_vs_random": "D3\noracle > random",
    }.get(key, key)


def plot_protocol(results: dict, out_dir: Path, log: logging.Logger) -> Path:
    ensure_dir(out_dir)
    fig_path = out_dir / "protocol_findings.png"

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        log.warning("matplotlib unavailable (%s); skipping protocol figure", exc)
        return fig_path

    comps = results.get("comparisons", {})
    names = list(comps.keys())
    means = np.array([comps[k]["mean_diff"] for k in names])
    lows = np.array([comps[k]["ci_low"] for k in names])
    highs = np.array([comps[k]["ci_high"] for k in names])
    errs = np.vstack([means - lows, highs - means])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    xs = np.arange(len(names))
    ax.bar(xs, means, yerr=errs, capsize=5, color=["#4c72b0", "#dd8452", "#55a868"])
    for x, name in zip(xs, names):
        lab = "sig" if comps[name].get("significant") else "ns"
        ax.text(x, highs[x] + 0.01, lab, ha="center", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels([_rule_label(n) for n in names], fontsize=8)
    ax.set_ylabel("mean accuracy diff (alt - base)")
    ax.set_title("Decision rules (95% CI)")
    ax.axhline(0, color="black", lw=0.8)

    ax = axes[1]
    frozen = results.get("frozen", {})
    if frozen:
        fkeys = [k for k in frozen if not k.startswith("node_shuffled")]
        fvals = [frozen[k] for k in fkeys]
        ferrs = np.array([frozen.get("node_shuffled_std", 0.0) if k == "node_shuffled_mean" else 0.0 for k in fkeys])
        ax.bar(np.arange(len(fkeys)), fvals, yerr=ferrs, capsize=5, color="#8172b3")
        ax.set_xticks(np.arange(len(fkeys)))
        ax.set_xticklabels([k.replace("_", "\n") for k in fkeys], fontsize=8)
        ax.set_ylabel("mixture accuracy")
        ax.set_ylim(0.0, 1.0)
        ax.set_title("Frozen-gate intervention")
    else:
        ax.text(0.5, 0.5, "no frozen data", ha="center", transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    log.info("wrote %s", fig_path)
    return fig_path


def plot_dilution(data: dict, out_dir: Path, log: logging.Logger) -> Path:
    fig_path = Path(out_dir) / "dilution" / "dilution_curve.png"
    ensure_dir(fig_path.parent)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        log.warning("matplotlib unavailable (%s); skipping dilution figure", exc)
        return fig_path

    ladder = data["ladder"]
    d = np.array([r["dilution"] for r in ladder])
    lf = np.array([r["label_free_gain"] for r in ladder])
    og = np.array([r["oracle_gain"] for r in ladder])

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(d, lf, "o-", label="label-free gain (D2)", color="#dd8452")
    ax.plot(d, og, "s--", label="oracle gain (D1)", color="#4c72b0")
    ax.set_xlabel("dilution (fraction of regime edges rewritten)")
    ax.set_ylabel("mixture-accuracy gain vs baseline")
    ax.set_title("Dilution curve: routing signal decay")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    log.info("wrote %s", fig_path)
    return fig_path


def plot_ablation(out_dir: Path, log: logging.Logger) -> Path:
    fig_path = Path(out_dir) / "ablation" / "ablation.png"
    ensure_dir(fig_path.parent)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        log.warning("matplotlib unavailable (%s); skipping ablation figure", exc)
        return fig_path

    csv_path = Path(out_dir) / "ablation" / "ablation.csv"
    if not csv_path.exists():
        log.warning("no %s; skipping ablation figure", csv_path)
        return fig_path

    rows = []
    with csv_path.open(newline="") as fh:
        import csv

        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)

    feat_rows = [r for r in rows if r.get("sweep") == "feature_variant"]
    delta_rows = [r for r in rows if r.get("sweep") == "oracle_delta"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    if feat_rows:
        ax = axes[0]
        labels = [r["variant"] for r in feat_rows]
        gains = [float(r["label_free_gain_vs_random"]) for r in feat_rows]
        ax.bar(np.arange(len(labels)), gains, color="#55a868")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("label-free gain vs random")
        ax.set_title("Router feature ablation")
        ax.axhline(gains[0], color="gray", ls=":", lw=1, label="all features")
        ax.legend(frameon=False)
    else:
        axes[0].text(0.5, 0.5, "no feature sweep", ha="center", transform=axes[0].transAxes)

    if delta_rows:
        ax = axes[1]
        d = np.array([float(r["delta"]) for r in delta_rows])
        g = np.array([float(r["oracle_bucket_gain_vs_uniform"]) for r in delta_rows])
        ax.plot(d, g, "o-", color="#4c72b0")
        ax.set_xlabel("oracle bucket width (delta)")
        ax.set_ylabel("oracle-bucket gain vs uniform")
        ax.set_title("Oracle filter threshold sweep")
        ax.grid(alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "no oracle sweep", ha="center", transform=axes[1].transAxes)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    log.info("wrote %s", fig_path)
    return fig_path


def _tex_escape(value: str) -> str:
    return (
        str(value)
        .replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


def _write_table(rows: list[dict], columns: list[str], csv_path: Path, tex_path: Path, caption: str, log: logging.Logger) -> None:
    import csv

    ensure_dir(csv_path.parent)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    md_path = csv_path.with_suffix(".md")
    md_lines = [f"# {caption}", "", "| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        cells = [str(row.get(c, "")).replace("|", r"\|") for c in columns]
        md_lines.append("| " + " | ".join(cells) + " |")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\begin{tabular}{" + "l" + "r" * (len(columns) - 1) + "}",
        r"\hline",
        " & ".join(_tex_escape(c) for c in columns) + r" \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(" & ".join(_tex_escape(row.get(c, "")) for c in columns) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\caption{" + _tex_escape(caption) + "}", r"\end{table}"]
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote %s, %s and %s (%d rows)", csv_path, tex_path, md_path, len(rows))


def export_table1(out_dir: Path, log: logging.Logger) -> None:
    path = out_dir / "protocol" / "protocol_results.json"
    if not path.exists():
        log.warning("no %s; skipping Table 1", path)
        return
    with path.open(encoding="utf-8") as fh:
        results = json.load(fh)

    labels = {
        "oracle_vs_uniform": "D1 oracle > uniform",
        "label_free_vs_random": "D2 label-free > random",
        "oracle_vs_random": "D3 oracle > random",
    }
    rows = []
    for key, comp in results.get("comparisons", {}).items():
        rows.append(
            {
                "rule": labels.get(key, key),
                "mean_diff": f"{comp['mean_diff']:.4f}",
                "ci_low": f"{comp['ci_low']:.4f}",
                "ci_high": f"{comp['ci_high']:.4f}",
                "p_wilcoxon": f"{comp['p_wilcoxon']:.6g}",
                "effect_size_dz": f"{comp.get('effect_size_dz', float('nan')):.2f}",
                "significant": str(comp.get("significant", "")),
            }
        )
    _write_table(
        rows,
        ["rule", "mean_diff", "ci_low", "ci_high", "p_wilcoxon", "effect_size_dz", "significant"],
        out_dir / "protocol" / "table1_findings.csv",
        out_dir / "protocol" / "table1_findings.tex",
        "D1-D4 findings matrix (Protocol A, PC-1c-R planted control).",
        log,
    )

    frozen = results.get("frozen", {})
    if frozen:
        frows = [
            {"gate": k, "mixture_accuracy": f"{v:.4f}"}
            for k, v in frozen.items()
            if not k.endswith("_std")
        ]
        frows.append({"gate": "node_shuffled_std", "mixture_accuracy": f"{frozen.get('node_shuffled_std', float('nan')):.4f}"})
        _write_table(
            frows,
            ["gate", "mixture_accuracy"],
            out_dir / "protocol" / "table1_frozen.csv",
            out_dir / "protocol" / "table1_frozen.tex",
            "Frozen-gate interventions.",
            log,
        )


def export_table2(out_dir: Path, log: logging.Logger) -> None:
    path = out_dir / "compare_methods" / "results.json"
    if not path.exists():
        log.warning("no %s; skipping Table 2", path)
        return
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    methods = ["threshold", "random", "degree", "spectral"]
    groups: dict[tuple, list[dict]] = {}
    for row in data.get("rows", []):
        groups.setdefault((row["n"], row["budget_factor"], row["epsilon"]), []).append(row)

    def _mean(rows: list[dict], key: str) -> float:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(vals)) if vals else float("nan")

    long_rows = []
    for (n, bf, eps), rows in sorted(groups.items()):
        for method in methods:
            long_rows.append(
                {
                    "n": n,
                    "budget_factor": bf,
                    "epsilon": eps,
                    "method": method,
                    "lowfreq": f"{_mean(rows, method + '_lowfreq'):.4f}",
                    "SD": f"{_mean(rows, method + '_SD'):.4f}",
                    "retention": f"{_mean(rows, method + '_retention'):.4f}",
                    "seconds": f"{_mean(rows, method + '_seconds'):.4f}",
                    "n_edges": f"{_mean(rows, method + '_edges'):.0f}",
                }
            )
    _write_table(
        long_rows,
        ["n", "budget_factor", "epsilon", "method", "lowfreq", "SD", "retention", "seconds", "n_edges"],
        out_dir / "compare_methods" / "table2_methods.csv",
        out_dir / "compare_methods" / "table2_methods.tex",
        "Table 2 (long form): per-method low-frequency distortion, SD, retention, runtime (mass-matched).",
        log,
    )

    wide_rows = []
    for (n, bf, eps), rows in sorted(groups.items()):
        entry = {"n": n, "budget_factor": bf, "epsilon": eps}
        for method in methods:
            entry[method] = f"{_mean(rows, method + '_lowfreq'):.4f}"
        wide_rows.append(entry)
    _write_table(
        wide_rows,
        ["n", "budget_factor", "epsilon"] + methods,
        out_dir / "compare_methods" / "table2_lowfreq.csv",
        out_dir / "compare_methods" / "table2_lowfreq.tex",
        "Table 2: mean low-frequency distortion by method (mass-matched).",
        log,
    )


def export_table3(out_dir: Path, log: logging.Logger) -> None:
    path = out_dir / "scalability_reeff" / "table3.json"
    if not path.exists():
        log.warning("no %s; skipping Table 3", path)
        return
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    def _fmt(v, spec: str, na: str = "--") -> str:
        return na if v is None else format(v, spec)

    rows = []
    for r in data.get("rows", []):
        rows.append(
            {
                "n": r["n"],
                "k": r["k"],
                "seconds_exact": _fmt(r.get("seconds_exact"), ".3f"),
                "seconds_topk": _fmt(r.get("seconds_topk"), ".3f"),
                "edge_rel_frobenius": _fmt(r.get("edge_rel_frobenius"), ".3f"),
                "edge_spearman_vs_exact": _fmt(r.get("edge_spearman_vs_exact"), ".3f"),
                "edge_spearman_vs_kmax": _fmt(r.get("edge_spearman_vs_kmax"), ".3f"),
                "rel_to_kmax": _fmt(r.get("rel_to_kmax"), ".2e"),
                "low_freq_rel_vs_exact": _fmt(r.get("low_freq_rel_vs_exact"), ".4f"),
            }
        )
    _write_table(
        rows,
        ["n", "k", "seconds_exact", "seconds_topk", "edge_rel_frobenius", "edge_spearman_vs_exact", "edge_spearman_vs_kmax", "rel_to_kmax", "low_freq_rel_vs_exact"],
        out_dir / "scalability_reeff" / "table3_export.csv",
        out_dir / "scalability_reeff" / "table3.tex",
        "Table 3: exact vs top-k effective resistance scalability (SBM, mass-matched).",
        log,
    )


def export_tables(out_dir: Path, log: logging.Logger) -> None:
    export_table1(out_dir, log)
    export_table2(out_dir, log)
    export_table3(out_dir, log)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    parser.add_argument("--out", default="logs", help="output directory containing the experiment logs")
    parser.add_argument("--tables", dest="tables", action="store_true", default=True, help="export Tables 1-3 as CSV + LaTeX + Markdown")
    parser.add_argument("--no-tables", dest="tables", action="store_false", help="skip Table 1-3 export")
    args = parser.parse_args()

    log = get_logger("reproduce_figures", args.log_level)
    ensure_dir(args.out)

    paths = {
        "protocol": Path(args.out) / "protocol" / "protocol_results.json",
        "dilution": Path(args.out) / "dilution" / "dilution_curve.json",
        "ablation": Path(args.out) / "ablation" / "ablation.csv",
    }

    for name, path in paths.items():
        if not path.exists():
            log.warning("missing %s log (%s); skipping", name, path)

    p = paths["protocol"]
    if p.exists():
        with p.open(encoding="utf-8") as fh:
            results = json.load(fh)
        plot_protocol(results, Path(args.out) / "protocol", log)
    p = paths["dilution"]
    if p.exists():
        with p.open(encoding="utf-8") as fh:
            data = json.load(fh)
        plot_dilution(data, Path(args.out), log)
    if paths["ablation"].exists():
        plot_ablation(Path(args.out), log)

    if args.tables:
        export_tables(Path(args.out), log)

    log.info("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
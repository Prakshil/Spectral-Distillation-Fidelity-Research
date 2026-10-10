"""Is routing loss driven by spectral distortion, or by disconnection?

Reports, per graph and pooled:

- whether the legacy scalar could discriminate at all (distinct values),
- Spearman correlation of each candidate driver against assignment stability,
- the same correlations with router-collapse-contaminated points removed, since a
  collapsed k-means run measures the optimiser rather than the graph,
- the graph's density, which turned out to decide whether the spectral term works
  at all.

The collapse filter matters: on Tolokers and YelpChi, heavy ``degree`` pruning
puts every router seed into a single expert. Those points have very low stability,
but charging that to spectral distortion would be wrong.
"""
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

COLLAPSE = "router_collapse_seeds"


def load(path):
    d = json.load(open(path, encoding="utf-8"))
    rows = []
    n = d["graph"]["n_nodes"]
    for p in d["points"]:
        rows.append({
            "source": d["graph"]["source"],
            "n": n,
            "density": 2 * p["n_edges_orig"] / max(1, n * (n - 1)),
            "method": p["method"],
            "retention": p["retention"],
            "lowfreq": p.get("low_frequency_error",
                             p["low_frequency"]["low_freq_mean_error"]),
            "legacy": p["legacy_SD"],
            "iso": p["connectivity"]["n_isolated_perturbed"] / n,
            "surv": p["routing_survival"],
            "stab": p["assignment_stability"],
            "oag": p["oracle_agreement"],
            "collapsed": p.get(COLLAPSE, 0),
            "n_seeds": len(p.get("assignment_stability_per_seed", []) or [1]),
        })
    return d, rows


def corr(x, y):
    if len(x) < 4 or np.ptp(y) == 0:
        return float("nan"), float("nan")
    s = spearmanr(x, y)
    return float(s.statistic), float(s.pvalue)


def report(rows, label, outcomes=("stab", "oag")):
    names = {"stab": "stability", "oag": "oracle_agreement"}
    lf = np.array([r["lowfreq"] for r in rows])
    iso = np.array([r["iso"] for r in rows])
    surv = np.array([r["surv"] for r in rows])
    legacy = np.array([r["legacy"] for r in rows])

    clean = [i for i, r in enumerate(rows) if r["collapsed"] == 0]
    cc = [i for i, r in enumerate(rows) if r["collapsed"] > 0]

    print(f"--- {label}  (n_points={len(rows)}, collapse-affected={len(cc)}) ---")
    print(f"    lowfreq range [{lf.min():.5f}, {lf.max():.5f}]   "
          f"legacy distinct values={len(set(np.round(legacy, 6)))}")
    for o in outcomes:
        y = np.array([r[o] for r in rows])
        s_all, p_all = corr(lf, y)
        s_cl, p_cl = corr(lf[clean], y[clean])
        sv_all, pv_all = corr(surv, y)
        sv_cl, pv_cl = corr(surv[clean], y[clean])
        iv_all, _ = corr(iso, y)
        print(f"    vs {names[o]}:")
        print(f"      lowfreq      all rho={s_all:+.3f} (p={p_all:.3g})   "
              f"collapse-free rho={s_cl:+.3f} (p={p_cl:.3g})  n={len(clean)}")
        print(f"      survival     all rho={sv_all:+.3f} (p={pv_all:.3g})   "
              f"collapse-free rho={sv_cl:+.3f} (p={pv_cl:.3g})")
        print(f"      isolatedfrac all rho={iv_all:+.3f}")
    # Report significance rather than an absolute threshold. An earlier
    # lowfreq_max>0.02 cutoff was calibrated on the legacy saturated metric and
    # was wrong here: it declared "spectrum not perturbed" for graphs that then
    # showed a significant collapse-free correlation. lowfreq is now normalized
    # by lambda_max, so its scale is per-graph and no fixed cutoff applies.
    s_moved, p_moved = corr(lf[clean], np.array([r["stab"] for r in rows])[clean])
    print(f"    collapse-free lowfreq->stability: "
          f"{'SIGNIFICANT' if p_moved < 0.05 else 'not significant'}"
          f" (rho={s_moved:+.3f}, p={p_moved:.3g})")
    return {"label": label, "n_points": len(rows), "collapse_affected": len(cc),
            "lowfreq_max": float(lf.max()), "density": float(rows[0]["density"]),
            "rho_lowfreq_stability_collapsefree": s_moved,
            "p_lowfreq_stability_collapsefree": p_moved}


def pretty_source(src):
    """Short label. The LLM sweep stores the model snapshot path as `source`,
    which otherwise floods the report with a 120-char path per graph."""
    if src is None:
        return "?"
    s = str(src).replace("\\", "/")
    if "models--" in s:
        return "llm:" + s.split("models--", 1)[1].split("/")[0].replace("--", "/")
    return os.path.basename(s.rstrip("/")) or s


def main():
    paths = sys.argv[1:] or sorted(glob.glob("logs/sd_vs_routing/*/sd_vs_routing.json"))
    all_rows, summaries = [], []
    for p in paths:
        d, rows = load(p)
        all_rows.extend(rows)
        summaries.append(report(rows, f"{pretty_source(d['graph']['source'])} "
                                      f"(n={d['graph']['n_nodes']})"))
    if len(paths) > 1:
        summaries.append(report(all_rows, "POOLED (all graphs)"))
        print()
        print("=== density vs whether the spectral term works ===")
        for s in summaries:
            if s["label"].startswith("POOLED"):
                continue
            works = (s["p_lowfreq_stability_collapsefree"] < 0.05)
            print(f"  {s['label']:<34} density={s['density']:.2e} "
                  f"lowfreq_max={s['lowfreq_max']:.4f} "
                  f"rho={s['rho_lowfreq_stability_collapsefree']:+.3f} "
                  f"p={s['p_lowfreq_stability_collapsefree']:.3g} "
                  f"-> {'PREDICTS' if works else 'no signal'}")
    out = "logs/sd_vs_routing/summary.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summaries, fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
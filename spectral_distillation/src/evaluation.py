"""Statistical evaluation, the D1-D4 protocol runner, and frozen-model
interventions (guide sections 6.5-6.7).

- paired non-parametric (Wilcoxon exact) and parametric (paired t) tests
- Holm-Bonferroni correction across the primary comparisons
- effect size d_z and 95% CI for mean differences
- ``run_protocol``: 10 splits x 3 seeds, averaged within split, that feeds the
  decision rules
- ``frozen_intervention``: learned / node-shuffled / global-mean / equal /
  oracle-one-hot gate settings with permutation baselines (2.3.5)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats


@dataclass
class ComparisonResult:
    name: str
    mean_diff: float
    ci_low: float
    ci_high: float
    p_wilcoxon: float
    p_paired_t: float
    effect_size_dz: float
    n_pairs: int
    alpha_corrected: float | None = None
    significant: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mean_diff": self.mean_diff,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "p_wilcoxon": self.p_wilcoxon,
            "p_paired_t": self.p_paired_t,
            "effect_size_dz": self.effect_size_dz,
            "n_pairs": self.n_pairs,
            "alpha_corrected": self.alpha_corrected,
            "significant": self.significant,
        }


def wilcoxon_paired(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided exact Wilcoxon signed-rank p-value (ties -> normal approx)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d = d[np.abs(d) > 1e-12]
    if d.size == 0:
        return 1.0
    if d.size < 15:
        try:
            stat, p = stats.wilcoxon(d, mode="exact")
            if np.isnan(p):
                raise ValueError
            return float(p)
        except (ValueError, ZeroDivisionError):
            pass
    try:
        _, p = stats.wilcoxon(d)
    except ValueError:
        return 1.0
    return float(p)


def paired_ttest(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided paired t p-value."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if d.size < 2 or np.std(d) < 1e-12:
        return 1.0
    return float(stats.ttest_rel(a, b).pvalue)


def effect_size_dz(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d_z for paired differences = mean(d) / std(d)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if d.size == 0 or np.std(d) < 1e-12:
        return 0.0
    return float(np.mean(d) / np.std(d))


def mean_ci95(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """mean difference and 95% CI (paired, via t-distribution)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    m = float(np.mean(d))
    n = d.size
    if n < 2:
        return m, m, m
    se = float(np.std(d, ddof=1) / np.sqrt(n))
    tcrit = float(stats.t.ppf(0.975, df=n - 1))
    return m, m - tcrit * se, m + tcrit * se


def holm_bonferroni(pvalues: list[float], alpha: float = 0.05) -> list[float]:
    """Per-rank Holm (1979) thresholds ``alpha / (k - rank)``, in input order.

    WARNING: these thresholds are only meaningful when applied *sequentially*
    -- compare the smallest p-value to its threshold first, and once one
    hypothesis fails its threshold every larger p-value is retained. Comparing
    each p-value to its own threshold independently (as callers used to do) does
    **not** control the family-wise error rate. Use :func:`holm_step_down` (or
    the adjusted p-values from :func:`holm_adjusted_pvalues`) for decisions.
    """
    order = np.argsort(pvalues)
    k = len(pvalues)
    thresholds = [0.0] * k
    for rank, idx in enumerate(order):
        thresholds[idx] = alpha / (k - rank)
    return thresholds


def holm_adjusted_pvalues(pvalues: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (monotone, capped at 1.0).

    ``p~(rank) = max_{i <= rank} (k - i + 1) * p_(i)`` on the ascending order.
    Rejecting ``p~ <= alpha`` is exactly the Holm step-down procedure and *does*
    control the FWER under arbitrary dependence.
    """
    p = np.asarray(pvalues, dtype=float)
    k = p.size
    if k == 0:
        return []
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(k, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, float((k - rank) * p[idx]))
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def holm_step_down(pvalues: list[float], alpha: float = 0.05) -> list[bool]:
    """Boolean rejection mask for the Holm step-down procedure.

    ``True`` where the (monotone) adjusted p-value is ``<= alpha``. This is the
    FWER-controlling replacement for the old ``p < threshold`` misuse: because
    the adjusted values are monotone in the sorted p-values, once the smallest
    non-rejected p-value stops the sequence, every larger one is retained.
    """
    return [bool(a <= alpha) for a in holm_adjusted_pvalues(pvalues)]


def combine_comparisons(results) -> dict[str, ComparisonResult]:
    """Re-key a ProtocolResults / list of ComparisonResult by name."""
    if hasattr(results, "comparisons"):
        comparisons = results.comparisons
    else:
        comparisons = list(results)
    return {c.name: c for c in comparisons}


def aggregate_seeds_within_split(
    per_seed_per_split: dict[tuple[int, int], float],
    n_splits: int,
    n_seeds: int,
) -> dict[int, np.ndarray]:
    """Average the metric across seeds within each split, return per-split means."""
    out: dict[int, list[float]] = {s: [] for s in range(n_splits)}
    for (split, seed), value in per_seed_per_split.items():
        out[split].append(value)
    return {s: np.mean(v) for s, v in out.items()}


def evaluate_comparison(
    splits_a: dict[int, float],
    splits_b: dict[int, float],
    name: str,
    alpha: float = 0.05,
) -> ComparisonResult:
    """Pair a and b across the shared split keys and run all statistics."""
    keys = sorted(set(splits_a) & set(splits_b))
    a = np.array([splits_a[k] for k in keys])
    b = np.array([splits_b[k] for k in keys])
    # NOTE: ``significant``/``alpha_corrected`` below are provisional -- they use
    # the raw ``alpha`` and must NOT be read as FWER-controlled. Callers that
    # evaluate several comparisons together (e.g. :func:`run_protocol`) overwrite
    # both fields with Holm-Bonferroni thresholds. Use this function directly
    # only for a single, pre-registered comparison.
    p_ww = wilcoxon_paired(a, b)
    p_t = paired_ttest(a, b)
    dz = effect_size_dz(a, b)
    m, lo, hi = mean_ci95(a, b)
    return ComparisonResult(
        name=name,
        mean_diff=m,
        ci_low=lo,
        ci_high=hi,
        p_wilcoxon=p_ww,
        p_paired_t=p_t,
        effect_size_dz=dz,
        n_pairs=len(keys),
        alpha_corrected=alpha,
        significant=bool(m > 0 and p_ww < alpha),
    )


@dataclass
class ProtocolResults:
    comparisons: list[ComparisonResult] = field(default_factory=list)
    decision_rules: dict[str, Any] = field(default_factory=dict)
    findings: list[dict[str, Any]] = field(default_factory=list)
    frozen: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "n_comparisons": len(self.comparisons),
            "decision_rules": self.decision_rules,
            "findings": self.findings,
            "verdict": self.decision_rules.get("verdict", "n/a"),
        }


def run_protocol(
    run_fn,
    n_splits: int = 10,
    n_seeds: int = 3,
    primary_pairs: list[tuple[str, str]] | None = None,
    alpha: float = 0.05,
    progress: bool = False,
    progress_desc: str = "protocol cells",
) -> ProtocolResults:
    """Generic protocol runner.

    ``run_fn(condition: str, split: int, seed: int) -> float`` returns a
    single performance metric for a condition on a split/seed cell. Metrics are
    averaged within split, then primary pairs are evaluated with all
    statistics. The remaining pairings (oracle>uniform etc.) are provided via
    ``primary_pairs`` as (baseline, alternative) names, comparing
    alternative minus baseline > 0.
    """
    if primary_pairs is None:
        primary_pairs = [
            ("uniform", "oracle"),       # D1 signal exists           (oracle > uniform)
            ("random", "label_free"),    # D2 router finds it         (label_free > random)
            ("random", "oracle"),        # D3 not capacity            (oracle > random)
        ]

    per_cell: dict[str, dict[tuple[int, int], float]] = {
        cond: {} for cond in {c for pair in primary_pairs for c in pair}
    }
    if progress:
        from tqdm import tqdm

        iterator = tqdm(
            ((s, seed, cond) for s in range(n_splits) for seed in range(n_seeds) for cond in per_cell),
            total=n_splits * n_seeds * len(per_cell),
            desc=progress_desc,
            unit="cell",
            ncols=100,
        )
    else:
        iterator = ((s, seed, cond) for s in range(n_splits) for seed in range(n_seeds) for cond in per_cell)
    for split, seed, cond in iterator:
        per_cell[cond][(split, seed)] = float(run_fn(cond, split, seed))

    split_means: dict[str, dict[int, float]] = {}
    for cond, cells in per_cell.items():
        split_means[cond] = aggregate_seeds_within_split(cells, n_splits, n_seeds)

    comparisons: list[ComparisonResult] = []
    for base, alt in primary_pairs:
        comparisons.append(
            evaluate_comparison(split_means[alt], split_means[base], f"{alt}_vs_{base}", alpha)
        )

    corrected = holm_bonferroni([c.p_wilcoxon for c in comparisons], alpha)
    reject = holm_step_down([c.p_wilcoxon for c in comparisons], alpha)
    for c, thresh, rej in zip(comparisons, corrected, reject):
        c.alpha_corrected = float(thresh)
        c.significant = bool(c.mean_diff > 0 and rej)

    return ProtocolResults(comparisons=comparisons)


def permute_within_strata(
    values: np.ndarray, strata: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Permute rows of ``values`` within each stratum independently.

    Preserves every stratum's membership set while destroying the node-to-row
    alignment *inside* each stratum. This is the stricter null for a frozen gate
    on graphs with known regimes: a global permutation can accidentally keep a
    node in a routing channel that matches its regime, whereas a within-stratum
    permutation matches the true regime sizes but scrambles which node gets
    which learned assignment.
    """
    values = np.asarray(values, dtype=float)
    strata = np.asarray(strata)
    out = values.copy()
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        out[idx] = values[idx][rng.permutation(idx.size)]
    return out


def frozen_intervention(
    learned: np.ndarray,
    n_nodes: int,
    run_expert_fn,
    n_experts: int = 3,
    oracle: np.ndarray | None = None,
    gate_settings: list[str] | None = None,
    n_permutations: int = 20,
    seed: int = 0,
    strata: np.ndarray | None = None,
) -> dict[str, Any]:
    """Frozen-model gate intervention (guide 2.3.5).

    ``run_expert_fn(assignment: np.ndarray) -> float`` runs the frozen expert
    mixture with a given routing assignment and returns accuracy.

    Gate settings:
    - ``learned``: the label-free router's learned assignment
    - ``node_shuffled``: global permutation baseline (``n_permutations`` draws)
    - ``within_regime_shuffled``: permutation baseline that only reshuffles the
      learned assignment *within* each stratum (requires ``strata``; falls back
      to a global shuffle when ``strata`` is ``None``)
    - ``global_mean``: every node gets the mean assignment of learned
    - ``equal``: uniform 1/K mixing
    - ``oracle_onehot``: oracle one-hot assignment (only if supplied)
    """
    if gate_settings is None:
        gate_settings = ["learned", "node_shuffled", "global_mean", "equal", "oracle_onehot"]
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {}

    learned = np.asarray(learned, dtype=float)
    n = n_nodes
    K = learned.shape[1] if learned.ndim == 2 else int(n_experts)

    if "learned" in gate_settings:
        out["learned"] = float(run_expert_fn(learned))

    if "oracle_onehot" in gate_settings and oracle is not None:
        out["oracle_onehot"] = float(run_expert_fn(np.asarray(oracle, dtype=float)))

    if "equal" in gate_settings:
        out["equal"] = float(run_expert_fn(np.full((n, K), 1.0 / K)))

    if "global_mean" in gate_settings:
        gmean = np.tile(learned.mean(axis=0), (n, 1))
        out["global_mean"] = float(run_expert_fn(gmean))

    if "node_shuffled" in gate_settings:
        perms = []
        for _ in range(n_permutations):
            perm = rng.permutation(n)
            perms.append(float(run_expert_fn(learned[perm])))
        perms = np.asarray(perms)
        out["node_shuffled_mean"] = float(perms.mean())
        out["node_shuffled_std"] = float(perms.std())
        out["node_shuffled_samples"] = perms.tolist()

    if "within_regime_shuffled" in gate_settings:
        shuffles = []
        for _ in range(n_permutations):
            if strata is None:
                perm = rng.permutation(n)
                shuffled = learned[perm]
            else:
                shuffled = permute_within_strata(learned, strata, rng)
            shuffles.append(float(run_expert_fn(shuffled)))
        shuffles = np.asarray(shuffles)
        out["within_regime_shuffled_mean"] = float(shuffles.mean())
        out["within_regime_shuffled_std"] = float(shuffles.std())
        out["within_regime_shuffled_samples"] = shuffles.tolist()

    return out


def partial_spearman(x, y, z) -> tuple[float, float]:
    """Partial Spearman correlation of ``x`` and ``y`` controlling for ``z``.

    Ranks all three vectors, linearly residualises the ranks of ``x`` and ``y``
    on the rank of ``z``, and returns the Pearson correlation of the residuals
    with its two-sided t-test p-value (``df = n - 3``). This isolates whether
    ``x`` carries information about ``y`` *beyond* the nuisance variable ``z`` --
    e.g. whether spectral distortion predicts routing stability beyond
    incidental node isolation. Returns ``(nan, nan)`` when there are fewer than
    4 points or either residual is constant.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    n = x.size
    if n < 4 or y.size != x.size or z.size != x.size:
        return float("nan"), float("nan")
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    rz = stats.rankdata(z)
    design = np.column_stack([rz, np.ones_like(rz)])
    coef_x, *_ = np.linalg.lstsq(design, rx, rcond=None)
    coef_y, *_ = np.linalg.lstsq(design, ry, rcond=None)
    ex = rx - design @ coef_x
    ey = ry - design @ coef_y
    if np.ptp(ex) == 0 or np.ptp(ey) == 0:
        return float("nan"), float("nan")
    r = float(np.corrcoef(ex, ey)[0, 1])
    if abs(r) >= 1.0:
        return r, 0.0
    n_obs = len(x)
    dof = n_obs - 3
    if dof <= 0:
        return r, float("nan")
    t = r * np.sqrt(dof / (1.0 - r * r))
    p = 2.0 * float(stats.t.sf(abs(t), dof))
    return r, p


def bootstrap_spearman_ci(x, y, n_boot: int = 2000, seed: int = 0,
                          alpha: float = 0.05) -> tuple[float, float, float]:
    """Bootstrap percentile CI for Spearman ``x`` vs ``y`` (resample pairs)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(x)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.ptp(y[idx]) == 0 or np.ptp(x[idx]) == 0:
            continue
        vals.append(float(stats.spearmanr(x[idx], y[idx]).statistic))
    if not vals:
        return float("nan"), float("nan"), float("nan")
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), float(np.mean(vals))
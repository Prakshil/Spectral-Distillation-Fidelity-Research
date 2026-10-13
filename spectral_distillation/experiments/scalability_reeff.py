"""Table 3 — scalability of the top-k effective-resistance approximation.

The dispatcher uses exact effective resistance for ``n <= 2000`` and a top-k
eigendecomposition approximation above that (`src/effective_resistance.py`;
guide §9.4 warning that top-k R is a *lower bound* accurate only near full rank).
This script quantifies that trade-off on synthetic stochastic-block-model graphs
at n = 100 / 1,000 / 10,000:

- **edge-resistance accuracy** — relative Frobenius / mean / max error of the
  top-k edge resistances against the exact ones (available when
  ``n <= exact_n_max``);
- **induced distortion** — smooth/low-frequency distortion (`low_frequency_distortion`)
  of a spectral sparsification built from the top-k resistances, vs the exact-R
  sparsification, and vs the full graph (when ``n <= sd_n_max``, because that path
  needs a dense ``eigvalsh``);
- **wall-clock** — exact pseudoinverse vs ``eigsh`` top-k;
- **self-convergence** — ``||r_k - r_Kmax|| / ||r_Kmax||`` across the k grid, the
  only accuracy reference available when exact is infeasible (n = 10,000).

The exact reference is skipped beyond ``exact_n_max`` (default 3,000) rather than
run a 10,000 x 10,000 pseudoinverse for a table cell; the JSON records this as
``exact_skipped: true`` so the omission is explicit, not silent.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.stats import spearmanr

from spectral_distillation.src.distortion import low_frequency_distortion
from spectral_distillation.src.effective_resistance import effective_resistance_exact
from spectral_distillation.src.laplacian import (
    build_adjacency,
    compute_laplacian,
    edges_from_weights,
)
from spectral_distillation.src.sparsifier import spectral_sparsify
from spectral_distillation.src.utils import ensure_dir, get_logger, load_yaml_config

DEFAULT_NS = [100, 1000, 10000]
DEFAULT_KS = [16, 32, 64, 128]


def vectorized_sbm(
    n: int,
    n_blocks: int = 5,
    p_in: float = 0.5,
    p_out: float = 0.05,
    weight_range: tuple[float, float] = (0.05, 1.0),
    seed: int = 0,
    chunk: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """SBM with a chunked/vectorised sampler (the looped version is O(n^2) in Python).

    Row-blocks of the adjacency are drawn at n x ``chunk`` granularity so the peak
    temporary size is bounded (``chunk`` rows of an n-vector), which keeps n = 10000
    tractable in memory.
    """
    rng = np.random.default_rng(seed)
    block = np.empty(n, dtype=int)
    for b, idx in enumerate(np.array_split(np.arange(n), n_blocks)):
        block[idx] = b
    lo, hi = weight_range
    A = np.zeros((n, n))
    cols = np.arange(n)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        same = block[start:stop, None] == block[None, :]
        prob = np.where(same, p_in, p_out)
        mask = rng.random((stop - start, n)) < prob
        mask &= cols[None, :] > np.arange(start, stop)[:, None]
        vals = np.zeros((stop - start, n))
        vals[mask] = rng.uniform(lo, hi, size=int(mask.sum()))
        A[start:stop] = vals
    return build_adjacency(A), block


def topk_scaled(L: np.ndarray, k: int, tol: float = 1e-10) -> np.ndarray:
    """Return ``V / sqrt(lambda)`` for the k smallest positive modes (n x k)."""
    L = np.asarray(L, dtype=float)
    n = L.shape[0]
    kk = int(max(1, min(k, n - 1)))
    if kk >= n - 1:
        evals, evecs = np.linalg.eigh(L)
    else:
        evals, evecs = eigsh(csr_matrix(L), k=kk + 1, which="LM", sigma=-1e-6)
    order = np.argsort(evals)
    evals, evecs = evals[order], evecs[:, order]
    keep = evals > tol
    evals, evecs = evals[keep], evecs[:, keep]
    return evecs / np.sqrt(evals)[None, :]


def edge_resistance_from_scaled(
    scaled: np.ndarray, rows: np.ndarray, cols: np.ndarray, chunk: int = 2_000_000
) -> np.ndarray:
    """``R_ij = ||scaled_i - scaled_j||^2`` without materialising the n x n matrix.

    Edges are processed in ``chunk``-sized blocks so the ``chunk x k`` working set
    stays small even for n = 10000 (~10^7 edges).
    """
    out = np.empty(rows.size, dtype=float)
    for start in range(0, rows.size, chunk):
        stop = min(start + chunk, rows.size)
        diff = scaled[rows[start:stop]] - scaled[cols[start:stop]]
        out[start:stop] = np.einsum("ij,ij->i", diff, diff)
    return out


def _rank_correlation(est: np.ndarray, ref: np.ndarray, max_points: int = 200_000) -> float:
    """Spearman correlation on a deterministic stride subsample (ranking is what the
    sparsifier consumes, so rank fidelity matters more than absolute R error)."""
    stride = max(1, est.size // max_points)
    idx = np.arange(0, est.size, stride)[:max_points]
    if idx.size < 3:
        return float("nan")
    return float(spearmanr(est[idx], ref[idx]).correlation)


def _rel_errors(r: np.ndarray, ref: np.ndarray, floor: float = 1e-9) -> dict:
    diff = r - ref
    denom = np.linalg.norm(ref) + 1e-30
    safe = np.maximum(np.abs(ref), floor)
    return {
        "rel_frobenius": float(np.linalg.norm(diff) / denom),
        "mean_rel": float(np.mean(np.abs(diff) / safe)),
        "max_rel": float(np.max(np.abs(diff) / safe)),
    }


def run_graph(n: int, args, log) -> list[dict]:
    W, _ = vectorized_sbm(
        n, n_blocks=args.n_blocks, p_in=args.p_in, p_out=args.p_out, seed=args.seed
    )
    rows, cols, weights = edges_from_weights(W)
    L = compute_laplacian(W)
    n_edges = weights.size
    budget = max(1, int(round(args.budget_factor * n)))
    rows_out: list[dict] = []

    exact_available = n <= args.exact_n_max
    r_exact = None
    t_exact = None
    lowfreq_exact = None
    if exact_available:
        t0 = time.perf_counter()
        R_exact = effective_resistance_exact(L)
        r_exact = R_exact[rows, cols]
        t_exact = time.perf_counter() - t0
        if n <= args.sd_n_max:
            W_sparse = spectral_sparsify(
                W, budget, resistance_edges=r_exact, epsilon=args.epsilon
            )
            lowfreq_exact = low_frequency_distortion(
                L, compute_laplacian(W_sparse)
            )["low_freq_mean_error"]

    ks = sorted({int(k) for k in args.ks if 0 < int(k) < n - 1})
    k_max = ks[-1] if ks else None
    r_kmax = None
    scaled_kmax = None
    if k_max is not None:
        scaled_kmax = topk_scaled(L, k_max)
        r_kmax = edge_resistance_from_scaled(scaled_kmax, rows, cols)

    for k in ks:
        t0 = time.perf_counter()
        scaled = topk_scaled(L, k)
        r_k = edge_resistance_from_scaled(scaled, rows, cols)
        t_topk = time.perf_counter() - t0

        row = {
            "n": n,
            "seed": args.seed,
            "k": k,
            "n_edges": n_edges,
            "n_edges_budget": budget,
            "seconds_topk": t_topk,
            "seconds_exact": t_exact,
            "exact_available": exact_available,
            "mean_R": float(np.mean(r_k)),
            "max_R": float(np.max(r_k)),
            "rel_to_kmax": float(np.linalg.norm(r_k - r_kmax) / (np.linalg.norm(r_kmax) + 1e-30))
            if r_kmax is not None
            else None,
        }
        if r_exact is not None:
            row.update(
                {
                    f"edge_{key}": val
                    for key, val in _rel_errors(r_k, r_exact).items()
                }
            )
            row["edge_spearman_vs_exact"] = _rank_correlation(r_k, r_exact)
        if r_kmax is not None and k != k_max:
            row["edge_spearman_vs_kmax"] = _rank_correlation(r_k, r_kmax)
        if lowfreq_exact is not None:
            W_sparse = spectral_sparsify(
                W, budget, resistance_edges=r_k, epsilon=args.epsilon
            )
            lf = low_frequency_distortion(L, compute_laplacian(W_sparse))["low_freq_mean_error"]
            row["low_freq_mean_error"] = lf
            row["low_freq_exact"] = lowfreq_exact
            row["low_freq_rel_vs_exact"] = abs(lf - lowfreq_exact) / (abs(lowfreq_exact) + 1e-30)
        rows_out.append(row)
        log.info(
            "n=%d k=%-3d topk=%.3fs edge_rel_frob=%s sd_rel=%s",
            n,
            k,
            t_topk,
            f"{row.get('edge_rel_frobenius'):.3e}" if "edge_rel_frobenius" in row else "n/a",
            f"{row.get('low_freq_rel_vs_exact'):.3e}" if "low_freq_rel_vs_exact" in row else "n/a",
        )
    if exact_available:
        log.info("n=%d exact R_eff time=%.3fs", n, t_exact)
    else:
        log.info("n=%d exact R_eff skipped (n > exact_n_max=%d)", n, args.exact_n_max)
    return rows_out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--ns", type=int, nargs="+", default=None)
    p.add_argument("--ks", type=int, nargs="+", default=None)
    p.add_argument("--budget-factor", type=int, default=None)
    p.add_argument("--epsilon", type=float, default=None)
    p.add_argument("--exact-n-max", type=int, default=None, help="largest n for exact R_eff")
    p.add_argument("--sd-n-max", type=int, default=None, help="largest n for induced-distortion check")
    p.add_argument("--n-blocks", type=int, default=None)
    p.add_argument("--p-in", type=float, default=None)
    p.add_argument("--p-out", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    defaults = {
        "ns": DEFAULT_NS,
        "ks": [16, 32, 64, 128],
        "budget_factor": 4,
        "epsilon": 0.1,
        "exact_n_max": 3000,
        "sd_n_max": 1000,
        "n_blocks": 5,
        "p_in": 0.5,
        "p_out": 0.05,
        "seed": 0,
    }
    config = load_yaml_config(args.config)
    exp_cfg = (config.get("experiments", {}) or {}).get("scalability_reeff", {}) or {}
    for key, default in defaults.items():
        cli = getattr(args, key)
        setattr(args, key, cli if cli is not None else exp_cfg.get(key, default))
    parent = Path(args.out) if args.out else Path(config.get("logging", {}).get("log_dir", "logs"))
    out_dir = ensure_dir(parent / "scalability_reeff")
    log = get_logger("scalability_reeff")

    all_rows: list[dict] = []
    for n in args.ns:
        log.info("=== n=%d ===", n)
        all_rows.extend(run_graph(n, args, log))

    fieldnames = [
        "n",
        "seed",
        "k",
        "n_edges",
        "n_edges_budget",
        "seconds_topk",
        "seconds_exact",
        "exact_available",
        "mean_R",
        "max_R",
        "rel_to_kmax",
        "edge_rel_frobenius",
        "edge_mean_rel",
        "edge_max_rel",
        "edge_spearman_vs_exact",
        "edge_spearman_vs_kmax",
        "low_freq_mean_error",
        "low_freq_exact",
        "low_freq_rel_vs_exact",
    ]
    with (out_dir / "table3.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k, "") if row.get(k) is not None else "" for k in fieldnames})

    payload = {
        "config": {"config": args.config, **{k: getattr(args, k) for k in defaults}},
        "rows": all_rows,
    }
    (out_dir / "table3.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    log.info("wrote %s table3.csv, table3.json", out_dir)

    print("\n=== Table 3: top-k vs exact effective resistance ===")
    print(f"{'n':>6} {'k':>4} {'t_exact(s)':>10} {'t_k(s)':>8} {'edge_relF':>10} {'rho':>6} {'sd_rel':>9}")
    for row in all_rows:
        te = row.get("seconds_exact")
        t_exact_s = f"{te:.3f}" if te is not None else "skip"
        edge_err = row.get("edge_rel_frobenius")
        edge_s = f"{edge_err:.2e}" if edge_err is not None else "n/a"
        rho = row.get("edge_spearman_vs_exact")
        if rho is None:
            rho = row.get("edge_spearman_vs_kmax")
        rho_s = f"{rho:.3f}" if rho is not None and rho == rho else "n/a"
        sd_err = row.get("low_freq_rel_vs_exact")
        sd_s = f"{sd_err:.2e}" if sd_err is not None else "n/a"
        print(
            f"{row['n']:>6} {row['k']:>4} {t_exact_s:>10} "
            f"{row['seconds_topk']:>8.3f} {edge_s:>10} {rho_s:>6} {sd_s:>9}"
        )


if __name__ == "__main__":
    main()

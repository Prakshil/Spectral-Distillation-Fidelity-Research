"""Empirical verification of Weyl and Davis-Kahan bounds and Cauchy interlacing.

Three distinct results are checked, each under its own (correct) name:

* **Weyl** -- arbitrary perturbation ``E`` shifts every eigenvalue by at most
  ``||E||_2``.
* **Edge-removal Weyl monotonicity** -- removing edge weight ``w`` is a PSD
  rank-1 update with ``||dL||_2 = 2w``, so eigenvalues only decrease, by at
  most ``2w``. (This was previously mislabeled "Cauchy interlacing".)
* **Cauchy interlacing** -- eigenvalues of a *principal submatrix* of a
  symmetric matrix interlace those of the full matrix:
  ``lambda_k <= mu_k <= lambda_{k+1}``.
* **Davis-Kahan** -- eigenvector rotation ``sin(theta_k)`` is bounded by
  ``||E||_2 / gap_k`` with the spectral (not Frobenius) norm and a per-eigenvalue
  gap, so the bound is meaningful instead of vacuously large.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh, eigvalsh

from spectral_distillation.src.spectral import compute_eigenvalues, min_eigengap


def check_weyl(L_a: np.ndarray, L_sparse: np.ndarray, eps: float = 1e-8) -> dict:
    L_a = np.asarray(L_a, dtype=float)
    L_sparse = np.asarray(L_sparse, dtype=float)
    e_a = compute_eigenvalues(L_a)
    e_s = compute_eigenvalues(L_sparse)
    shifts = np.abs(e_s - e_a)
    spectral_norm = float(np.linalg.norm(L_a - L_sparse, 2))
    frob_norm = float(np.linalg.norm(L_a - L_sparse, "fro"))
    violations = np.flatnonzero(shifts > spectral_norm + eps)
    return {
        "holds": bool(violations.size == 0),
        "max_shift": float(shifts.max()),
        "spectral_norm": spectral_norm,
        "frobenius_norm": frob_norm,
        "n_violations": int(violations.size),
    }


def check_davis_kahan(
    L_a: np.ndarray, L_sparse: np.ndarray, eps: float = 1e-8
) -> dict:
    """Davis-Kahan eigenvector-rotation bound ``sin(theta_k) <= ||E||_2 / gap_k``.

    ``E = L_sparse - L_a``. The bound deliberately uses the **spectral** norm
    ``||E||_2`` and a **per-eigenvalue** gap to the nearest other eigenvalue.
    Using the Frobenius norm and a single global gap (as before) made the bound
    vacuously large for Laplacians -- the global gap includes the ``lambda=0``
    eigenvalue, so the bound blew up and every rotation trivially "held". An
    eigenvalue whose gap is (near-)degenerate gets an infinite bound and is
    reported as unbounded instead of being counted as verified.
    """
    L_a = np.asarray(L_a, dtype=float)
    L_sparse = np.asarray(L_sparse, dtype=float)
    lam_a, va = eigh(L_a)
    _, vs = eigh(L_sparse)
    spectral_norm = float(np.linalg.norm(L_sparse - L_a, 2))
    n = L_a.shape[0]
    tol = max(eps, 1e-6)

    cos_angles = np.array([float(abs(va[:, k] @ vs[:, k])) for k in range(n)])
    sines = np.sqrt(np.maximum(0.0, 1.0 - cos_angles ** 2))

    gaps: list[float] = []
    bounds: list[float] = []
    violations: list[int] = []
    for k in range(n):
        others = np.delete(lam_a, k)
        gap = float(np.min(np.abs(others - lam_a[k]))) if others.size else float("inf")
        gaps.append(gap)
        if gap <= tol:
            bounds.append(float("inf"))
            continue
        bound_k = spectral_norm / gap
        bounds.append(bound_k)
        if sines[k] > bound_k + tol:
            violations.append(k)

    finite_bounds = [b for b in bounds if np.isfinite(b)]
    return {
        "holds": bool(len(violations) == 0),
        "bound_sin_theta": float(max(finite_bounds)) if finite_bounds else float("inf"),
        "max_sin_theta": float(sines.max()) if sines.size else 0.0,
        "min_eigengap": float(min_eigengap(lam_a)),
        "per_k_gap": gaps,
        "per_k_bound": bounds,
        "per_k_sin_theta": sines.tolist(),
        "spectral_norm": spectral_norm,
        "frobenius_norm": float(np.linalg.norm(L_sparse - L_a, "fro")),
        "n_violations": int(len(violations)),
        "n_unbounded": int(sum(1 for b in bounds if not np.isfinite(b))),
    }


def edge_removal_eigenvalues(W: np.ndarray, i: int, j: int) -> np.ndarray:
    W_removed = np.asarray(W, dtype=float).copy()
    W_removed[i, j] = 0.0
    W_removed[j, i] = 0.0
    from spectral_distillation.src.laplacian import compute_laplacian

    return compute_eigenvalues(compute_laplacian(W_removed))


def check_edge_removal_weyl(L: np.ndarray, i: int, j: int, eps: float = 1e-8) -> dict:
    """Weyl monotonicity for deleting edge ``(i, j)`` from a Laplacian.

    Removing edge weight ``w = |L[i, j]|`` is the PSD rank-1 update
    ``w (e_i - e_j)(e_i - e_j)^T`` (spectral norm ``2w``), so Weyl's inequality
    gives ``0 <= lambda_full - lambda_removed <= 2w`` componentwise. This is the
    result that was previously (incorrectly) called "Cauchy interlacing".
    """
    L = np.asarray(L, dtype=float)
    n = L.shape[0]
    w = abs(L[i, j])
    e_i = np.zeros(n)
    e_j = np.zeros(n)
    e_i[i] = 1.0
    e_j[j] = 1.0
    L_removed = L - w * np.outer(e_i - e_j, e_i - e_j)
    e_full = compute_eigenvalues(L)
    e_removed = compute_eigenvalues(L_removed)
    return {
        "holds": bool(np.all(e_removed <= e_full + eps)),
        "componentwise_decrease": (e_full - e_removed).tolist(),
        "max_decrease": float((e_full - e_removed).max()),
        "weyl_bound": float(2.0 * w),
        "edge_weight": float(w),
        "eigenvalues_full": e_full.tolist(),
        "eigenvalues_removed": e_removed.tolist(),
    }


def check_cauchy_interlacing(H: np.ndarray, drop_index: int, eps: float = 1e-8) -> dict:
    """Cauchy interlacing for the principal submatrix without ``drop_index``.

    If symmetric ``H`` (``n x n``) has eigenvalues ``lambda_1 <= ... <= lambda_n``
    and its ``(n-1) x (n-1)`` principal submatrix (row/col ``drop_index`` removed)
    has eigenvalues ``mu_1 <= ... <= mu_{n-1}``, then
    ``lambda_k <= mu_k <= lambda_{k+1}`` for ``k = 1 .. n-1``.
    """
    H = np.asarray(H, dtype=float)
    n = H.shape[0]
    submatrix = np.delete(np.delete(H, drop_index, axis=0), drop_index, axis=1)
    lam = np.linalg.eigvalsh(H)
    mu = np.linalg.eigvalsh(submatrix)
    lower_margins = mu - lam[: n - 1]
    upper_margins = lam[1:] - mu
    return {
        "holds": bool(np.all(lower_margins >= -eps) and np.all(upper_margins >= -eps)),
        "lower_margins": lower_margins.tolist(),
        "upper_margins": upper_margins.tolist(),
        "min_lower_margin": float(lower_margins.min()) if lower_margins.size else 0.0,
        "min_upper_margin": float(upper_margins.min()) if upper_margins.size else 0.0,
        "eigenvalues_full": lam.tolist(),
        "eigenvalues_submatrix": mu.tolist(),
        "dropped_index": int(drop_index),
    }


def check_all_bounds(W: np.ndarray, tau: float = 0.5, eps: float = 1e-8) -> dict:
    from spectral_distillation.src.baselines import threshold_sparsify
    from spectral_distillation.src.laplacian import compute_laplacian

    L_full = compute_laplacian(W)
    W_sparse = threshold_sparsify(W, tau)
    L_sparse = compute_laplacian(W_sparse)
    return {
        "weyl": check_weyl(L_full, L_sparse, eps),
        "davis_kahan": check_davis_kahan(L_full, L_sparse, eps),
        "eigenvalues_dense": eigvalsh(L_full).tolist(),
        "eigenvalues_sparse": eigvalsh(L_sparse).tolist(),
    }
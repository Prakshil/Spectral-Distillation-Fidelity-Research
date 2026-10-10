"""Spectral distortion diagnostics: the pre-training routing-quality metric."""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigvalsh


def compute_spectral_distortion(
    L_a: np.ndarray,
    L_sparse: np.ndarray,
    eps: float = 1e-8,
    ref_eigenvalues: np.ndarray | None = None,
) -> float:
    """Worst relative eigenvalue error between a reference and a perturbed Laplacian.

    ``ref_eigenvalues`` lets a caller reuse an already-computed reference
    spectrum. Across a retention ladder the reference Laplacian is identical at
    every point, so recomputing it turns an O(n^3) cost into a no-op and halves
    the cost of measuring SD on a large graph.
    """
    e_a = np.sort(eigvalsh(np.asarray(L_a, dtype=float))) if ref_eigenvalues is None \
        else np.sort(np.asarray(ref_eigenvalues, dtype=float))
    e_s = np.sort(eigvalsh(np.asarray(L_sparse, dtype=float)))
    relative_errors = np.abs(e_a - e_s) / (np.abs(e_a) + eps)
    relative_errors[0] = 0.0
    return float(relative_errors.max())


def rank_matched_distortion(
    L_a: np.ndarray | None,
    L_sparse: np.ndarray,
    eps: float = 1e-8,
    ref_eigenvalues: np.ndarray | None = None,
    top_ks: tuple[int, ...] = (1, 8, 32, 128),
) -> dict:
    """Component-count-robust spectral distortion.

    Why this exists: ``compute_spectral_distortion`` pairs eigenvalue *i* of
    the reference with eigenvalue *i* of the perturbed graph. That pairing is
    only meaningful when both graphs have the same number of connected
    components. Sparsification changes the component count on every
    interesting point -- pruning YelpChi to 4% degree retention leaves 14,460 of
    14,840 nodes isolated, i.e. 14,462 zero eigenvalues against 1 in the
    reference. After sorting, index *i* refers to an unrelated mode in each
    graph, and the metric saturates at exactly 1.0 for every mode whose
    reference eigenvalue sits above ``eps`` while the perturbed one is 0.

    Measured on the YelpChi retention ladder, that legacy scalar returned
    1.0000 at all nine sampled points -- carrying no information at all -- while
    this rank-matched version separates the methods cleanly.

    The fix is to compare the *largest* eigenvalues. These are the modes a
    low-pass filter actually retains, they are ordered stably under
    fragmentation, and they do not depend on how many components the graph has.
    ``top_ks`` reports the worst relative error over each descending prefix.
    """
    e_a = np.sort(eigvalsh(np.asarray(L_a, dtype=float)))[::-1] if ref_eigenvalues is None \
        else np.sort(np.asarray(ref_eigenvalues, dtype=float))[::-1]
    e_s = np.sort(eigvalsh(np.asarray(L_sparse, dtype=float)))[::-1]
    if len(e_a) != len(e_s):
        raise ValueError(
            f"reference spectrum has {len(e_a)} eigenvalues but perturbed "
            f"matrix is {len(e_s)}x{len(e_s)}"
        )

    out: dict = {}
    for k in top_ks:
        if k > len(e_a):
            continue
        a_k, s_k = e_a[:k], e_s[:k]
        rel = np.abs(a_k - s_k) / (a_k + eps)
        out[f"top{k}_rel_error"] = float(rel.max())
    lmax_a, lmax_s = float(e_a[0]), float(e_s[0])
    out["lambda_max_ref"] = lmax_a
    out["lambda_max_perturbed"] = lmax_s
    out["lambda_max_rel_error"] = abs(lmax_a - lmax_s) / (abs(lmax_a) + eps)
    out["lambda_min_ref"] = float(e_a[-1])
    out["lambda_min_perturbed"] = float(e_s[-1])
    return out


def spectral_distortion_profile(
    L_a: np.ndarray | None,
    L_sparse: np.ndarray,
    eps: float = 1e-8,
    ref_eigenvalues: np.ndarray | None = None,
) -> dict:
    """Per-eigenvalue distortion profile, not just the worst-case scalar.

    The headline SD is a max, so a single badly-perturbed mode hides whether
    distortion is spread across the spectrum or concentrated in the low modes
    that actually drive routing. Returns the quantiles of the relative error
    alongside the full error vector, plus the spectral gap on the reference --
    the quantity ``compute_fragility`` divides SD by.

    Shape-validates the reference spectrum so a cached value from a different
    graph is rejected rather than silently producing a nonsense profile.

    ``L_a`` may be ``None`` when ``ref_eigenvalues`` is supplied. A retention
    ladder calls this once per point with a spectrum that never changes, so
    keeping the reference Laplacian alive only to re-derive those eigenvalues
    would hold an O(n^2) buffer for the whole run.
    """
    L_sparse = np.asarray(L_sparse, dtype=float)
    if L_a is None and ref_eigenvalues is None:
        raise ValueError("spectral_distortion_profile needs L_a or ref_eigenvalues")

    e_a = np.sort(eigvalsh(np.asarray(L_a, dtype=float))) if ref_eigenvalues is None \
        else np.sort(np.asarray(ref_eigenvalues, dtype=float))
    if len(e_a) != L_sparse.shape[0]:
        raise ValueError(
            f"reference spectrum has {len(e_a)} eigenvalues but the perturbed "
            f"matrix is {L_sparse.shape[0]}x{L_sparse.shape[1]}; the cached "
            f"reference was computed for a different graph"
        )
    if L_a is not None and np.asarray(L_a).shape != L_sparse.shape:
        raise ValueError(f"shape mismatch: {np.asarray(L_a).shape} vs {L_sparse.shape}")

    e_s = np.sort(eigvalsh(L_sparse))
    rel = np.abs(e_a - e_s) / (np.abs(e_a) + eps)
    rel[0] = 0.0

    gaps = np.diff(e_a)
    min_gap = float(gaps.min()) if gaps.size else float("inf")
    return {
        "SD": float(rel.max()),
        "argmax_index": int(rel.argmax()),
        "median_relative_error": float(np.percentile(rel, 50)),
        "p90_relative_error": float(np.percentile(rel, 90)),
        "p99_relative_error": float(np.percentile(rel, 99)),
        "low_mode_mean_error": float(rel[1: min(len(rel), 33)].mean()),
        "min_eigengap": min_gap,
        "fragility_score": compute_fragility(float(rel.max()), min_gap, eps),
        "lambda_max_ref": float(e_a[-1]),
        "lambda_max_perturbed": float(e_s[-1]),
    }


def connectivity_report(
    L_ref: np.ndarray | None,
    L_sparse: np.ndarray,
    ref_eigenvalues: np.ndarray | None = None,
    eps: float = 1e-8,
) -> dict:
    """Component counts and isolated mass, so fragmentation is visible not hidden.

    The legacy scalars saturate precisely *because* sparsification disconnects
    the graph: the perturbed spectrum acquires a large block of zero eigenvalues,
    which destroys index-matched pairing and pins the metric at 1.0. Reporting
    the component counts explicitly separates "the spectrum is distorted" from
    "the graph fell apart", which are different diagnoses with different fixes.
    """
    e_s = np.sort(eigvalsh(np.asarray(L_sparse, dtype=float)))
    e_a = np.sort(eigvalsh(np.asarray(L_ref, dtype=float))) if L_ref is not None \
        else np.sort(np.asarray(ref_eigenvalues, dtype=float))
    tol_a = max(eps, 1e-10 * float(e_a[-1]))
    tol_s = max(eps, 1e-10 * float(e_s[-1]))
    n_irr_a = int((e_a <= tol_a).sum())
    n_irr_s = int((e_s <= tol_s).sum())
    return {
        "n_components_ref": n_irr_a,
        "n_components_perturbed": n_irr_s,
        "n_isolated_perturbed": max(0, n_irr_s - n_irr_a),
        "largest_ref": float(e_a[-1]),
        "largest_perturbed": float(e_s[-1]),
    }


def low_frequency_distortion(
    L_a: np.ndarray | None,
    L_sparse: np.ndarray,
    eps: float = 1e-8,
    ref_eigenvalues: np.ndarray | None = None,
    k: int = 32,
    zero_tol: float = 1e-10,
) -> dict:
    """Component-count-robust, graded distortion of the smooth global modes.

    This is the metric the routing claim actually needs. A router consumes
    low-pass-filtered features, so what matters is whether the *smooth* modes
    survive sparsification -- not whether the index-matched spectrum lines up.

    Both regimes of the legacy scalar are artefacts of index pairing:

    - ascending pairing breaks when the perturbed graph gains zero eigenvalues
      (every mode above ``eps`` whose partner is 0 scores relative error 1.0,
      so the max pins at exactly 1.0), and
    - it collapses to exactly 0.0 when sparsification keeps every edge, since
      the spectra are then identical.

    Both were observed at all 27 points of the real-graph retention ladder. The
    fix is to discard the zero eigenvalues on *each* side and match the smallest
    surviving modes in order. Fragmentation then costs only what it should --
    the isolated mass, reported by :func:`connectivity_report` -- and the
    remaining error is graded, so methods and budgets separate.

    Errors are normalised by ``lambda_max`` of the reference rather than by the
    individual mode. Per-mode normalisation divides by eigenvalues that are
    arbitrarily close to zero for the smoothest modes, which is precisely the
    region this metric is about, and blows the error up to hundreds. Normalising
    by the graph scale keeps the value in ``[0, ~2]`` and, because it no longer
    depends on ``n``, makes results comparable across graphs of different size.
    """
    e_a = np.sort(eigvalsh(np.asarray(L_a, dtype=float))) if L_a is not None \
        else np.sort(np.asarray(ref_eigenvalues, dtype=float))
    L_sparse = np.asarray(L_sparse, dtype=float)
    e_s = np.sort(eigvalsh(L_sparse))
    if len(e_a) != L_sparse.shape[0]:
        raise ValueError(
            f"reference spectrum has {len(e_a)} eigenvalues but perturbed matrix "
            f"is {L_sparse.shape[0]}x{L_sparse.shape[1]}"
        )

    scale = float(e_a[-1])
    tol_a = max(eps, zero_tol * scale)
    tol_s = max(eps, zero_tol * max(scale, float(e_s[-1])))
    pos_a = e_a[e_a > tol_a]
    pos_s = e_s[e_s > tol_s]

    kk = int(min(k, pos_a.size, pos_s.size))
    if kk == 0:
        return {
            "low_freq_k": 0,
            "low_freq_mean_error": 1.0,
            "low_freq_median_error": 1.0,
            "low_freq_p90_error": 1.0,
            "low_freq_max_error": 1.0,
            "low_freq_gap_ref": 0.0,
            "normalisation": scale,
            "n_matched_modes_ref": int(pos_a.size),
            "n_matched_modes_perturbed": int(pos_s.size),
        }

    a_k, s_k = pos_a[:kk], pos_s[:kk]
    rel = np.abs(a_k - s_k) / (scale + eps)
    gaps = np.diff(a_k)
    return {
        "low_freq_k": kk,
        "low_freq_mean_error": float(rel.mean()),
        "low_freq_median_error": float(np.median(rel)),
        "low_freq_p90_error": float(np.percentile(rel, 90)),
        "low_freq_max_error": float(rel.max()),
        "low_freq_gap_ref": float(gaps.min()) if gaps.size else float("inf"),
        "normalisation": scale,
        "n_matched_modes_ref": int(pos_a.size),
        "n_matched_modes_perturbed": int(pos_s.size),
    }


def routing_survival(
    low_freq_mean_error: float,
    n_isolated_perturbed: int,
    n_nodes: int,
) -> float:
    """Retention proxy charging for isolated mass as well as spectral error.

    ``predict_retention(SD) = 1 - SD`` is degenerate on this problem: SD saturates
    at 1.0, so retention is pinned at 0 regardless of how much structure actually
    survived, which is why the four-node diagnostic reported ~1.3e-8.

    The two penalties multiply rather than combine by ``max``. A ``max`` lets a
    pristine spectrum excuse total disconnection, so a graph that orphaned 72% of
    its nodes still scored like a near-perfect graph. Multiplying requires both
    the smooth modes *and* the connectivity to survive, which is what routing
    actually needs: an orphaned node cannot be routed at all.
    """
    spectral_term = 1.0 - float(np.clip(low_freq_mean_error, 0.0, 1.0))
    connectivity_term = 1.0 - (float(n_isolated_perturbed) / max(1, int(n_nodes)))
    return float(max(0.0, min(1.0, spectral_term * connectivity_term)))


def compute_frobenius_norm(L_a: np.ndarray, L_sparse: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(L_a, dtype=float) - np.asarray(L_sparse, dtype=float), "fro"))


def compute_fragility(SD: float, min_gap: float, eps: float = 1e-8) -> float:
    return float(SD / (min_gap + eps))


def predict_retention(SD: float) -> float:
    return float(max(0.0, 1.0 - SD))


def davis_kahan_bound(frob_norm: float, min_gap: float, eps: float = 1e-8) -> float:
    return float(2.0 * frob_norm / (min_gap + eps))


def spectral_distortion_report(
    L_a: np.ndarray, L_sparse: np.ndarray, eps: float = 1e-8
) -> dict:
    L_a = np.asarray(L_a, dtype=float)
    L_sparse = np.asarray(L_sparse, dtype=float)
    e_a = np.sort(eigvalsh(L_a))
    e_s = np.sort(eigvalsh(L_sparse))
    relative_errors = np.abs(e_a - e_s) / (np.abs(e_a) + eps)
    relative_errors[0] = 0.0
    gaps = np.diff(e_a)
    min_gap = float(gaps.min()) if gaps.size else float("inf")
    frob = float(np.linalg.norm(L_a - L_sparse, "fro"))
    sd = float(relative_errors.max())
    return {
        "SD": sd,
        "min_eigengap": min_gap,
        "frobenius_norm": frob,
        "davis_kahan_bound": davis_kahan_bound(frob, min_gap, eps),
        "predicted_retention": predict_retention(sd),
        "fragility_score": compute_fragility(sd, min_gap, eps),
        "eigenvalues_dense": e_a.tolist(),
        "eigenvalues_sparse": e_s.tolist(),
        "relative_errors": relative_errors.tolist(),
    }
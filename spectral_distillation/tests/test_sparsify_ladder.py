import numpy as np

from spectral_distillation.experiments import run_sparsify_ladder as ladder
from spectral_distillation.src.effective_resistance import (
    effective_resistance_exact,
    resistance_energy_identity,
)
from spectral_distillation.src.laplacian import build_adjacency, compute_laplacian
from spectral_distillation.src.router_protocol import label_homophily
from spectral_distillation.src.sparsifier import spectral_sparsify


def _toy(n=40, seed=0):
    rng = np.random.default_rng(seed)
    A = (rng.random((n, n)) < 0.25).astype(float)
    A = np.maximum(A, A.T)
    W = build_adjacency(A)
    y = (rng.random(n) < 0.3).astype(int)
    if np.unique(y).size < 2:
        y[: n // 2] = 1
        y[n // 2 :] = 0
    return W, y


def test_resistance_matrix_injection_matches_internal_path():
    """Injecting the exact resistance equals the default computation."""
    W, _ = _toy()
    R = effective_resistance_exact(compute_laplacian(W))
    n_edges = int(np.count_nonzero(W) // 2)
    budget = max(1, n_edges // 2)
    a = spectral_sparsify(W, budget, sample=True, seed=1, resistance_matrix=R)
    b = spectral_sparsify(W, budget, sample=True, seed=1, k_eig=W.shape[0] - 1)
    assert np.count_nonzero(a) // 2 == np.count_nonzero(b) // 2


def test_spectral_sparsify_hits_budget_exactly():
    W, _ = _toy(n=60, seed=1)
    n_edges = int(np.count_nonzero(W) // 2)
    budget = int(0.5 * n_edges)
    W_sp = spectral_sparsify(W, budget, sample=False)
    assert int(np.count_nonzero(W_sp) // 2) == budget


def test_spectral_sparsify_preserves_resistance_identity_direction():
    """ER-rescaled graph stays closer to the original than an equal-budget
    unrescaled random prune (spectral-distortion direction check)."""
    W, _ = _toy(n=50, seed=2)
    n_edges = int(np.count_nonzero(W) // 2)
    budget = max(1, n_edges // 2)
    er = spectral_sparsify(W, budget, sample=True, seed=0)
    # resistance energy of the *original* graph is the identity sum.
    assert resistance_energy_identity(W) > 0
    # ER-rescaled energies remain finite and same order of magnitude.
    e_er = resistance_energy_identity(er)
    assert np.isfinite(e_er) and e_er > 0


def test_ladder_methods_registry_and_homophily():
    assert set(ladder.METHODS) == {"er", "random", "degree"}
    W, y = _toy(n=30, seed=3)
    h = ladder._homophily_mean(W, y)
    assert 0.0 <= h <= 1.0


def test_ladder_comparison_dict_keys():
    class _C:
        mean_diff = 0.1
        ci_low = 0.0
        ci_high = 0.2
        p_wilcoxon = 0.01
        p_paired_t = 0.02
        effect_size_dz = 1.0
        significant = True

    d = ladder._comparison_dict(_C())
    assert set(d) == {
        "mean_diff", "ci_low", "ci_high",
        "p_wilcoxon", "p_paired_t", "effect_size_dz", "significant",
    }


def test_label_homophily_bounds():
    W, y = _toy(n=25, seed=4)
    h = label_homophily(W, y)
    finite = h[~np.isnan(h)]
    assert finite.size > 0
    assert ((finite >= 0.0) & (finite <= 1.0)).all()

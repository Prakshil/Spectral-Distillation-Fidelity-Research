"""Tests for the frozen-gate architecture/capacity ablations (Phase 6 WS3)."""

import numpy as np

from spectral_distillation.experiments.frozen_interventions import (
    architecture_configs,
    gate_ordering,
)
from spectral_distillation.src.evaluation import frozen_intervention, permute_within_strata
from spectral_distillation.src.mixture import (
    dendrite_expert_factories,
    fit_experts,
    mlp_head_width,
)


def test_permute_within_strata_preserves_membership_and_sizes():
    values = np.arange(12).reshape(6, 2).astype(float)
    strata = np.array([0, 0, 0, 1, 1, 1])
    rng = np.random.default_rng(0)
    out = permute_within_strata(values, strata, rng)
    for s in (0, 1):
        idx = np.flatnonzero(strata == s)
        assert set(map(tuple, out[idx])) == set(map(tuple, values[idx]))
    assert sorted(map(tuple, out)) == sorted(map(tuple, values))


def test_permute_within_strata_is_identity_on_singletons():
    values = np.eye(3)
    strata = np.array([0, 1, 2])
    out = permute_within_strata(values, strata, np.random.default_rng(3))
    assert np.allclose(out, values)


def test_frozen_intervention_within_regime_gate_returns_stats():
    learned = np.eye(3)[[0, 1, 2, 0, 1, 2]]
    oracle = learned.copy()
    strata = np.array([0, 0, 1, 1, 2, 2])

    def run_expert_fn(assignment):
        return float(np.mean(np.argmax(assignment, axis=1) == np.argmax(oracle, axis=1)))

    out = frozen_intervention(
        learned=learned,
        n_nodes=6,
        run_expert_fn=run_expert_fn,
        oracle=oracle,
        gate_settings=["learned", "within_regime_shuffled"],
        n_permutations=5,
        seed=0,
        strata=strata,
    )
    assert out["learned"] == 1.0
    assert "within_regime_shuffled_mean" in out
    assert len(out["within_regime_shuffled_samples"]) == 5


def test_frozen_intervention_within_regime_falls_back_without_strata():
    learned = np.eye(2)[[0, 1, 0, 1]]
    out = frozen_intervention(
        learned=learned,
        n_nodes=4,
        run_expert_fn=lambda a: 0.5,
        gate_settings=["within_regime_shuffled"],
        n_permutations=3,
        seed=1,
    )
    assert out["within_regime_shuffled_mean"] == 0.5


def test_mlp_head_width_sets_hidden_layer():
    head = mlp_head_width(32)(None)
    assert head.hidden_layer_sizes == (32,)


def test_dendrite_factories_have_distinct_widths():
    factories = dendrite_expert_factories([8, 32, 128])
    widths = [f(None).hidden_layer_sizes for f in factories]
    assert widths == [(8,), (32,), (128,)]


def test_fit_experts_honours_per_expert_factories():
    X = np.random.default_rng(0).normal(size=(60, 4))
    y = (X[:, 0] > 0).astype(int)
    assignment = np.zeros((60, 3))
    assignment[:20, 0] = 1.0
    assignment[20:40, 1] = 1.0
    assignment[40:, 2] = 1.0
    train = np.ones(60, dtype=bool)
    experts = fit_experts(
        X, y, assignment, train, expert_factories=dendrite_expert_factories([8, 32, 128])
    )
    widths = [e.hidden_layer_sizes for e in experts]
    assert widths == [(8,), (32,), (128,)]


def test_architecture_configs_cover_linear_widths_and_dendrite():
    names = [name for name, _ in architecture_configs([8, 64])]
    assert names == ["logistic", "mlp8", "mlp64", "dendrite"]


def test_gate_ordering_detects_clean_ordering():
    result = {
        "learned": 0.62,
        "oracle_onehot": 0.90,
        "node_shuffled_mean": 0.50,
        "within_regime_shuffled_mean": 0.55,
        "global_mean": 0.50,
        "equal": 0.50,
    }
    ordering = gate_ordering(result)
    assert ordering["holds"] is True
    assert ordering["within_regime_gap"] == 0.62 - 0.55
    assert ordering["oracle_ge_learned"] is True


def test_gate_ordering_holds_when_oracle_edged_by_noise():
    result = {
        "learned": 0.862,
        "oracle_onehot": 0.860,
        "node_shuffled_mean": 0.60,
        "within_regime_shuffled_mean": 0.855,
        "global_mean": 0.55,
        "equal": 0.50,
    }
    ordering = gate_ordering(result)
    assert ordering["holds"] is True
    assert ordering["oracle_ge_learned"] is False
    assert ordering["oracle_minus_learned"] == 0.860 - 0.862


def test_gate_ordering_fails_when_learned_loses():
    result = {
        "learned": 0.48,
        "oracle_onehot": 0.90,
        "node_shuffled_mean": 0.50,
        "within_regime_shuffled_mean": 0.55,
        "global_mean": 0.50,
        "equal": 0.50,
    }
    ordering = gate_ordering(result)
    assert ordering["holds"] is False
    assert ordering["learned_gt_node_shuffled"] is False


def test_architecture_configs_include_linear_and_dendrite():
    names = [name for name, _ in architecture_configs([16, 64])]
    assert names[0] == "logistic"
    assert names[-1] == "dendrite"
    assert "mlp16" in names and "mlp64" in names

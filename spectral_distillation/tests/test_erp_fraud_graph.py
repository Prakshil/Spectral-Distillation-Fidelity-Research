"""Tests for the scalable synthetic ERP-fraud graph generator."""

import numpy as np
import pytest

from spectral_distillation.src.synthetic import (
    _ROLE_INVOICE,
    _ROLE_CUSTOMER,
    _ROLE_SHELL,
    _ROLE_SUPPLIER,
    generate_erp_fraud_graph,
)


def test_graph_contract_and_scale():
    g = generate_erp_fraud_graph(n_nodes=4000, seed=42)
    assert g["W"].shape == (4000, 4000)
    for key in ("W", "features", "y", "regimes", "roles", "fraud"):
        assert key in g
    assert g["features"].shape == (4000, 16)
    assert int(np.unique(g["y"]).size) == 5


def test_adjacency_symmetric_and_diagonal_free():
    g = generate_erp_fraud_graph(n_nodes=400, seed=1)
    W = g["W"]
    assert np.allclose(W, W.T)
    assert np.count_nonzero(np.diag(W)) == 0
    assert np.count_nonzero(W) > 0


def test_role_to_regime_mapping():
    g = generate_erp_fraud_graph(n_nodes=2000, seed=3)
    roles, regimes = g["roles"], g["regimes"]
    assert set(roles.tolist()) == {0, 1, 2, 3}
    assert np.all(regimes[roles == _ROLE_SUPPLIER] == 0)
    assert np.all(regimes[roles == _ROLE_INVOICE] == 0)
    assert np.all(regimes[roles == _ROLE_SHELL] == 1)
    assert np.all(regimes[roles == _ROLE_CUSTOMER] == 2)


def test_fraud_mask_is_shell():
    g = generate_erp_fraud_graph(n_nodes=2000, seed=7)
    assert np.array_equal(g["fraud"], (g["roles"] == _ROLE_SHELL).astype(int))
    assert g["fraud"].sum() >= 1


def test_regimes_are_all_populated():
    g = generate_erp_fraud_graph(n_nodes=2000, seed=11)
    for r in (0, 1, 2):
        assert np.count_nonzero(g["regimes"] == r) >= 1


def test_structural_degree_separation():
    g = generate_erp_fraud_graph(n_nodes=2000, seed=5)
    W, regimes = g["W"], g["regimes"]
    mean_deg = {r: float(W[regimes == r].sum(axis=1).mean()) for r in (0, 1, 2)}
    assert mean_deg[1] > mean_deg[0] > mean_deg[2]
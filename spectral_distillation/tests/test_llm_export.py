"""Phase 4 step-1 tests: real-attention export pure functions (no transformers).

Covers the graph-construction math and dataset/label helpers that do not
require the heavy ``transformers`` dependency: attention renormalization,
head aggregation, block-diagonal graph building, entropy features, and
sentence-id labeling. The live model export path is exercised by the
``run_real_attention`` experiment instead.
"""

import numpy as np
import pytest

from spectral_distillation.src.attention_entropy import compute_attention_entropy
from spectral_distillation.src.llm_export import (
    AttentionRecord,
    _sentence_ids_from_offsets,
    aggregate_attention,
    build_attention_graph,
    passage_to_sentence_ids,
    renormalize_attention,
    split_dataset,
)


def _fake_record(n: int = 24, layers: int = 4) -> AttentionRecord:
    rng = np.random.default_rng(0)
    A = rng.uniform(0.0, 1.0, size=(layers, n, n))
    A = renormalize_attention(A)
    return AttentionRecord(
        tokens=[f"t{i}" for i in range(n)],
        attention=A,
        sentence_ids=passage_to_sentence_ids(n, n_sentences=3),
    )


def test_renormalize_rows_sum_to_one():
    A = np.array([[0.5, 1.5], [0.0, 0.0]])
    P = renormalize_attention(A)
    assert np.allclose(P.sum(axis=1), [1.0, 0.0])
    # zero row must not become NaN
    assert not np.isnan(P).any()


def test_aggregate_attention_shape_and_value():
    rec = _fake_record()
    mean = aggregate_attention(rec, how="mean")
    assert mean.shape == (24, 24)
    mx = aggregate_attention(rec, how="max")
    assert np.all(mx >= mean - 1e-12)


def test_build_attention_graph_schema():
    rec = _fake_record()
    g = build_attention_graph(rec)
    assert g["W"].shape == (24, 24)
    assert g["W"].ndim == 2
    assert np.allclose(g["W"], g["W"].T)  # symmetrized
    assert np.all(np.diag(g["W"]) == 0.0)
    assert g["features"].shape == (24, 3)  # mean/max/std entropy
    assert g["y"].shape == (24,)
    assert set(g["y"]).issubset({0, 1, 2})
    assert len(g["tokens"]) == 24
    assert g["n_nodes"] == 24


def test_multi_passage_block_diagonal():
    a = _fake_record(n=10, layers=3)
    b = _fake_record(n=6, layers=3)
    g = build_attention_graph([a, b])
    assert g["W"].shape == (16, 16)
    # no cross-block edges
    assert np.all(g["W"][:10, 10:] == 0.0)
    assert np.all(g["W"][10:, :10] == 0.0)
    assert g["d_features"] == 3
    assert g["passages"] == 2


def test_max_edges_per_node_sparsifies():
    rec = _fake_record()
    g = build_attention_graph(rec, max_edges_per_node=5)
    assert np.all(g["W"].sum(axis=1) > 0)  # every node keeps edges
    assert np.all(g["W"].sum(axis=1) < 1e18)  # sanity, not a check of count


def test_feature_mode_agg_is_legacy_default():
    # The committed artifacts were produced with mean/max/std entropy, so `agg`
    # must stay the default and must keep d_features == 3 for any depth.
    rec = _fake_record(n=12, layers=24)
    g = build_attention_graph(rec)
    assert g["feature_mode"] == "agg"
    assert g["features"].shape == (12, 3)
    assert g["d_features"] == 3


def test_feature_mode_per_layer_gives_one_column_per_layer():
    rec = _fake_record(n=12, layers=24)
    g = build_attention_graph(rec, feature_mode="per_layer")
    assert g["feature_mode"] == "per_layer"
    assert g["features"].shape == (12, 24)
    assert g["d_features"] == 24


def test_feature_mode_does_not_change_the_graph():
    # Feature extraction must not perturb W: Protocol B compares conditions on
    # an identical adjacency, and a leaky feature path would silently change
    # the sparsification target.
    rec = _fake_record(n=12, layers=6)
    a = build_attention_graph(rec, feature_mode="agg")
    b = build_attention_graph(rec, feature_mode="per_layer")
    assert np.array_equal(a["W"], b["W"])
    assert np.array_equal(a["y"], b["y"])


def test_feature_mode_rejects_unknown():
    rec = _fake_record(n=8, layers=3)
    with pytest.raises(ValueError, match="feature_mode"):
        build_attention_graph(rec, feature_mode="nope")


def test_sentence_ids_total_flow():
    ids = passage_to_sentence_ids(100, sentence_lengths=[30, 30, 40])
    assert ids.shape == (100,)
    assert sorted(set(ids.tolist())) == [0, 1, 2]


def test_split_dataset_respects_budget_boundaries():
    text = "\n".join(f"line number {i} of the corpus" for i in range(20))
    passages = split_dataset(text, max_tokens=15)
    assert len(passages) > 1
    for p in passages:
        assert len(p.split()) <= 15 + 7  # word-count budget plus line grouping slack


def test_sentence_ids_from_punctuation_offsets():
    text = "Alpha here. Beta there! Gamma?"
    offsets = [(0, 5), (6, 10), (12, 16), (17, 22), (24, 29), (29, 30)]
    ids = _sentence_ids_from_offsets(text, offsets, len(offsets))
    assert list(ids) == [0, 0, 1, 1, 2, 2]


def test_entropy_features_on_single_layer():
    rng = np.random.default_rng(1)
    A = renormalize_attention(rng.uniform(0, 1, size=(1, 8, 8)))
    H = compute_attention_entropy(np.stack([A[0]], axis=0))
    assert H.shape == (1, 8)
    assert np.all(H >= 0.0)
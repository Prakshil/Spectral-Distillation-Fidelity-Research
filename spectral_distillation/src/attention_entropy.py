"""Attention-entropy routing signals computed before graph construction.

Entropy of the per-node attention distribution is a pre-construction signal:
it can be computed directly from the LLM (or synthetic attention layers) and
aggregated across layers/heads before any graph is built, so graph distillation
cannot corrupt it.
"""

from __future__ import annotations

import numpy as np


def node_attention_entropy(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Per-node Shannon entropy (nats) of a row-stochastic attention matrix."""
    A = np.asarray(A, dtype=float)
    row_sums = A.sum(axis=1)
    P = A / np.maximum(row_sums[:, None], eps)
    P = np.maximum(P, eps)
    H = -(P * np.log(P)).sum(axis=1)
    return H


def compute_attention_entropy(attn_layers: np.ndarray) -> np.ndarray:
    """Entropy per node across layers.

    Accepts:
      - (n, n) single attention matrix -> (n,) shape
      - (L, n, n) layers -> (L, n) shape, one entropy vector per layer
    """
    attn_layers = np.asarray(attn_layers, dtype=float)
    if attn_layers.ndim == 2:
        return node_attention_entropy(attn_layers)
    if attn_layers.ndim == 3:
        return np.stack(
            [node_attention_entropy(attn_layers[i]) for i in range(attn_layers.shape[0])]
        )
    raise ValueError(f"attn_layers must be 2D or 3D, got {attn_layers.ndim}D")


def aggregate_entropy_features(H: np.ndarray) -> np.ndarray:
    """Aggregate (L, n) entropy into (n, d) node features (mean, max, std)."""
    H = np.asarray(H, dtype=float)
    if H.ndim == 1:
        H = H[None, :]
    feats = [H.mean(axis=0), H.max(axis=0), H.std(axis=0)]
    return np.stack(feats, axis=1)


def synthetic_attention_layers(
    W: np.ndarray, n_layers: int = 4, noise: float = 0.1, seed: int | None = None
) -> np.ndarray:
    """Synthesize plausible attention layers from a graph (no LLM required).

    Row-normalizes the weighted adjacency (plus self-loops) and perturbs each
    layer so downstream entropy-based routing can be exercised end-to-end.
    """
    rng = np.random.default_rng(seed)
    W = np.asarray(W, dtype=float)
    n = W.shape[0]
    base = W + np.eye(n)
    row_sums = base.sum(axis=1, keepdims=True)
    P0 = base / np.maximum(row_sums, 1e-12)
    layers = []
    for _ in range(n_layers):
        P = P0 + rng.uniform(0.0, noise, size=(n, n))
        P = P / P.sum(axis=1, keepdims=True)
        layers.append(P)
    return np.stack(layers)
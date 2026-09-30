"""Real-world public fraud graph loaders for the D1-D4 protocol.

The synthetic PC-1c-R / ERP-fraud graphs validate that the protocol machinery
recovers routing structure when the oracle, label-free, random, and uniform
conditions are applied to graphs with planted regimes. This module provides the
same schema consumed by those generators (:key:`W`, :key:`features`,
:key:`y`) for *real* public fraud graphs, so the four condition assignments
(oracle homophily buckets, label-free structural clustering, usage-matched
random, uniform) can be re-run on genuine data.

The primary loader targets the DGL Amazon review-correlation fraud graph
(``https://data.dgl.ai/dataset/FraudAmazon.zip`` -> ``Amazon.mat``), a
co-review network of 11,944 users with 25 handcrafted features and real
benign/fraud labels. The first 3,305 rows of the mat are the documented
unlabeled block; the remaining 8,639 users are fully labeled (7,818 benign,
821 fraudulent, exactly the 9.5% positive ratio GADBench reports) and form a
single connected component, which is the subgraph this loader returns. The
``homo`` field is the union adjacency over the three review-correlation
relations (U-P-U, U-S-U, U-V-U); it is symmetrized and diagonal-free to match
the ``build_adjacency`` contract used by the rest of the pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp

from spectral_distillation.src.laplacian import build_adjacency

DEFAULT_AMAZON_MAT = "data/benchmarks/amazon/raw/Amazon.mat"
DEFAULT_TOLOKERS_DIR = "data/benchmarks/tolokers/raw"

LABELED_BLOCK_START = 3305  # DGL drops rows 0..3304 as unlabeled for amazon


def load_amazon_fraud(path: str | Path = DEFAULT_AMAZON_MAT) -> dict:
    """Load the labeled-induced Amazon co-review fraud subgraph.

    Returns the protocol graph schema: ``W`` (symmetric float adjacency over
    the 8,639 labeled users), ``features`` (25-dim review statistics), ``y``
    (binary labels, 1 = fraudulent user), plus provenance metadata
    (``n_nodes``, ``d_features``, ``source``, ``positive_ratio``,
    ``n_edges``).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Amazon.mat not found at {path}; download from "
            "https://data.dgl.ai/dataset/FraudAmazon.zip and extract"
        )
    mat = sio.loadmat(str(path))
    full = sp.csr_matrix(mat["homo"])
    full = sp.csr_matrix(full + full.T)
    full.data[:] = 1.0
    full.sum_duplicates()

    feat_all = np.asarray(mat["features"].todense() if sp.issparse(mat["features"]) else mat["features"], dtype=float)
    labels = mat["label"].ravel().astype(int)

    if len(labels) <= LABELED_BLOCK_START:
        raise ValueError(
            f"expected >= {LABELED_BLOCK_START + 1} rows in Amazon.mat, got {len(labels)}"
        )

    idx = np.arange(LABELED_BLOCK_START, len(labels))
    W = full[idx][:, idx]
    W = build_adjacency(W.toarray())

    y = labels[idx]
    features = feat_all[idx]
    return {
        "W": W,
        "features": features,
        "y": y,
        "n_nodes": int(y.size),
        "d_features": int(features.shape[1]),
        "source": "DGL Amazon (FraudAmazonDataset, homo block 3305+)",
        "positive_ratio": float(y.mean()),
        "n_edges": int(np.count_nonzero(W) // 2),
    }


def load_tolokers(path: str | Path = DEFAULT_TOLOKERS_DIR) -> dict:
    """Load the Tolokers crowd-worker exclusion graph (GADBench/NEExT release).

    Nodes are Toloka crowd workers; an edge joins two workers who shared at
    least one task; the positive class marks workers banned from a project
    (Platonov et al., ICLR 2023). Unlike the ultra-dense Amazon co-review
    graph this network is comparatively sparse (mean degree ~88 vs ~763) and
    far less homophilic, which makes it the natural second regime for the
    retention ladder and for probing the label-free router (D2).

    Expects ``nodes.parquet`` (node_id, feat_*, label) and ``edges.parquet``
    (src, dst) in ``path``; download with
    ``huggingface_hub.hf_hub_download("JaySuryavanshi/graph-anomaly-tolokers", ...)``.
    """
    import pandas as pd

    path = Path(path)
    nodes_pq, edges_pq = path / "nodes.parquet", path / "edges.parquet"
    if not (nodes_pq.exists() and edges_pq.exists()):
        raise FileNotFoundError(
            f"Tolokers parquet files not found under {path}; download "
            "nodes.parquet/edges.parquet from the HuggingFace mirror "
            "JaySuryavanshi/graph-anomaly-tolokers"
        )

    nodes = pd.read_parquet(nodes_pq)
    edges = pd.read_parquet(edges_pq)

    feat_cols = sorted(
        (c for c in nodes.columns if c.startswith("feat_")),
        key=lambda c: int(c.split("_")[1]),
    )
    if not feat_cols or "label" not in nodes.columns:
        raise ValueError(f"unexpected Tolokers node schema: {list(nodes.columns)}")

    features = nodes[feat_cols].to_numpy(dtype=float)
    y = nodes["label"].to_numpy().astype(int)
    n = int(y.size)

    src = edges["src"].to_numpy(dtype=np.int64)
    dst = edges["dst"].to_numpy(dtype=np.int64)
    if src.max() >= n or dst.max() >= n or src.min() < 0 or dst.min() < 0:
        raise ValueError("Tolokers edge indices out of range for node table")

    adj = sp.coo_matrix(
        (np.ones(src.size, dtype=np.float64), (src, dst)), shape=(n, n)
    )
    adj = ((adj + adj.T) > 0).astype(np.float64)
    adj.setdiag(0.0)
    adj.eliminate_zeros()
    W = build_adjacency(adj.toarray())

    return {
        "W": W,
        "features": features,
        "y": y,
        "n_nodes": n,
        "d_features": int(features.shape[1]),
        "source": "Tolokers (GADBench/NEExT crowd-worker exclusion graph)",
        "positive_ratio": float(y.mean()),
        "n_edges": int(np.count_nonzero(W) // 2),
    }


REAL_FRAUD_LOADERS = {
    "amazon": (load_amazon_fraud, DEFAULT_AMAZON_MAT),
    "tolokers": (load_tolokers, DEFAULT_TOLOKERS_DIR),
}


def load_real_fraud(dataset: str = "amazon", path: str | Path | None = None) -> dict:
    """Dispatch to a named real fraud-graph loader (``amazon`` or ``tolokers``)."""
    key = dataset.strip().lower()
    if key not in REAL_FRAUD_LOADERS:
        raise KeyError(
            f"unknown real fraud dataset {dataset!r}; "
            f"choose from {sorted(REAL_FRAUD_LOADERS)}"
        )
    loader, default_path = REAL_FRAUD_LOADERS[key]
    return loader(default_path if path is None else path)
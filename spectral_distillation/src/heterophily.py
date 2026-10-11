"""Heterophily benchmark loaders (Actor / Chameleon / Squirrel) for the Node-MoE audit.

Node-MoE (arXiv:2406.03464) reports its headline gains on the Geom-GCN
heterophily suite (Pei et al., 2020): **Actor**, **Chameleon**, and **Squirrel**.
Those are its *home* benchmarks, so a critique built only on fraud graphs leaves
the authors one escape hatch -- "fraud topologies are noisy/unhelpful; the method
shines on heterophilic web/actor graphs." Running the identical harness on the
authors' own datasets closes that hatch.

Schema
------
Returns the same protocol graph schema as :mod:`spectral_distillation.src.real_fraud`
(``W`` dense 0/1 adjacency, ``features``, ``y``) plus two extras the fraud loaders
do not need:

* ``n_classes`` -- these datasets are 5-way, not binary, so every head is told the
  number of classes (see ``n_classes`` in :mod:`spectral_distillation.src.node_moe`);
* ``class_counts`` -- for the payload's provenance.

Provenance / leakage note
-------------------------
The **legacy** Chameleon/Squirrel graphs carry duplicate nodes, which leaks across
the published Geom-GCN splits (Platonov et al., ICLR 2023). We deliberately do
**not** use the published splits: the whole pipeline applies this repo's own
class-stratified 50/50 split (:func:`~spectral_distillation.src.mixture.train_test_split`),
so every dataset here and every fraud graph is scored under one identical split
rule and no split-level leakage is inherited. The duplicate-node caveat is
recorded in the payload, not silently ignored.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp

from spectral_distillation.src.laplacian import build_adjacency

DEFAULT_HETEROPHILY_ROOT = "data/heterophily"


def _graph_from_arrays(
    features: np.ndarray,
    edge_index: np.ndarray,
    y: np.ndarray,
    name: str,
    source: str,
) -> dict:
    """Build the protocol graph schema from raw arrays (download-free, testable)."""
    x = np.asarray(features, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    y = np.asarray(y).ravel().astype(int)
    n = x.shape[0]

    ei = np.asarray(edge_index)
    if ei.ndim != 2:
        raise ValueError(f"edge_index must be 2-D, got shape {ei.shape}")
    if ei.shape[0] != 2 and ei.shape[1] == 2:
        ei = ei.T
    if ei.shape[0] != 2:
        raise ValueError(f"edge_index must be (2, m), got shape {ei.shape}")
    src, dst = ei[0].astype(np.int64), ei[1].astype(np.int64)

    A = sp.coo_matrix((np.ones(src.size, dtype=float), (src, dst)), shape=(n, n))
    A = ((A + A.T) > 0).astype(float)
    A.setdiag(0.0)
    A.eliminate_zeros()
    W = build_adjacency(A.toarray())

    classes, counts = np.unique(y, return_counts=True)
    return {
        "W": W,
        "features": x,
        "y": y,
        "n_nodes": int(n),
        "d_features": int(x.shape[1]),
        "n_classes": int(classes.max()) + 1,
        "source": source,
        "class_counts": {int(c): int(k) for c, k in zip(classes, counts)},
        "n_edges": int(np.count_nonzero(W) // 2),
    }


def _load_pyg(name: str, root: str | Path):
    """Fetch a PyG dataset object for ``name`` (downloads on first use)."""
    from torch_geometric.datasets import Actor, WikipediaNetwork

    root = Path(root)
    if name == "actor":
        return Actor(root=str(root / "Actor"))
    if name in ("chameleon", "squirrel"):
        return WikipediaNetwork(root=str(root), name=name)
    raise KeyError(f"unknown heterophily dataset {name!r}")


def load_heterophily(
    dataset: str = "actor",
    root: str | Path | None = None,
) -> dict:
    """Dispatch to a named heterophily loader.

    Actor, Chameleon and Squirrel are all dense-feasible (<= 7,600 nodes), so no
    ``max_nodes`` cap exists or is needed; passing one is rejected rather than
    silently ignored.
    """
    key = dataset.strip().lower()
    if key not in ("actor", "chameleon", "squirrel"):
        raise KeyError(
            f"unknown heterophily dataset {dataset!r}; choose from "
            "['actor', 'chameleon', 'squirrel']"
        )
    root = DEFAULT_HETEROPHILY_ROOT if root is None else root
    data = _load_pyg(key, root)[0]

    x = data.x.detach().cpu().numpy() if hasattr(data.x, "detach") else np.asarray(data.x)
    y = data.y.detach().cpu().numpy() if hasattr(data.y, "detach") else np.asarray(data.y)
    ei = (
        data.edge_index.detach().cpu().numpy()
        if hasattr(data.edge_index, "detach")
        else np.asarray(data.edge_index)
    )
    source = (
        f"PyG {key.title()} (Geom-GCN heterophily suite; legacy graphs, "
        "duplicate-node caveat, our stratified split)"
    )
    return _graph_from_arrays(x, ei, y, key, source)

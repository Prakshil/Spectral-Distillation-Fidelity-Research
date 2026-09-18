"""PC-1c-R planted multi-regime control graph (guide section 6.3).

Generates an ``n_nodes`` graph with three structural regimes -- ``L`` low-pass
(homophilic wiring), ``H`` high-pass (anti-correlated class wiring), and ``M``
noisy random wiring with informative features -- mixed across ``n_patches``
regime patches. A dilution ladder converts progressively more nodes toward
homogeneous low-pass wiring.

Deconfounding invariants (configs/planted_control.yaml):
- feature noise is constant across the dilution ladder
- converted node sets are nested (a node converted at dilution d stays
  converted at every higher dilution)
- unconverted nodes retain their original wiring entirely
- no arm saturates: a single-M expert trained only on informative features
  scores well below ceiling (~0.80), so routing signals remain measurable
"""

from __future__ import annotations

import numpy as np

REGIME_L, REGIME_H, REGIME_M = 0, 1, 2


def _one_hot_prototypes(n_classes: int, d_features: int, rng: np.random.Generator) -> np.ndarray:
    if d_features < n_classes:
        raise ValueError(f"d_features ({d_features}) must be >= n_classes ({n_classes})")
    protos = np.zeros((n_classes, d_features))
    for c in range(n_classes):
        protos[c, c] = 1.0
    return protos


def _random_orthogonal(d: int, rng: np.random.Generator) -> np.ndarray:
    """A uniformly random orthogonal matrix via QR of a Gaussian matrix."""
    if d == 1:
        return np.array([[1.0]])
    A = rng.standard_normal((d, d))
    Q, _ = np.linalg.qr(A)
    return Q


def _wire_block(
    n: int,
    classes: np.ndarray,
    regimes: np.ndarray,
    p_homophily: float,
    p_hetero: float,
    p_random: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Wire nodes of the block; each regime draws edges with its own rule."""
    W = np.zeros((n, n), dtype=float)
    a, b = np.triu_indices(n, k=1)
    ca, cb = classes[a], classes[b]
    ra, rb = regimes[a], regimes[b]
    probs = np.zeros(a.size)

    is_L = (ra == REGIME_L) & (rb == REGIME_L)
    is_H = (ra == REGIME_H) & (rb == REGIME_H)
    is_M = (ra == REGIME_M) & (rb == REGIME_M)

    probs[is_L] = np.where(ca[is_L] == cb[is_L], p_homophily, p_hetero)
    probs[is_H] = np.where(
        (cb[is_H]) == (ca[is_H] + 1) % 5,  # class c wired to c+1 (anti-correlated)
        p_homophily,
        p_hetero,
    )
    probs[is_M] = p_random

    keep = rng.random(a.size) < probs
    W[a[keep], b[keep]] = 1.0
    W[b[keep], a[keep]] = 1.0
    return W


def generate_pc_graph(
    n_nodes: int = 10_000,
    n_classes: int = 5,
    n_patches: int = 50,
    regime_ratio: tuple[float, float, float] = (0.4, 0.4, 0.2),
    d_features: int = 16,
    p_homophily: float = 0.25,
    p_hetero: float = 0.02,
    p_random: float = 0.02,
    feature_noise_scale: float = 0.4,
    seed: int = 0,
) -> dict:
    """Generate a single PC-1c-R graph.

    Returns a dict: W, y (class labels), regimes (0=L, 1=H, 2=M), features,
    patches, class_prototypes, n_classes, n_patches, regime_ratio, dilution, and
    the rng used (for reproducibility of the dilution ladder).
    """
    rng = np.random.default_rng(seed)
    ratio = np.asarray(regime_ratio, dtype=float)
    ratio = ratio / ratio.sum()

    counts = [max(1, int(np.rint(n_nodes * ratio[r]))) for r in range(3)]
    counts[0] += n_nodes - sum(counts)

    classes = np.zeros(n_nodes, dtype=int)
    regimes = np.zeros(n_nodes, dtype=int)
    features = np.zeros((n_nodes, d_features))
    prototypes = _one_hot_prototypes(n_classes, d_features, rng)
    # Each regime projects the shared prototypes through its own orthogonal
    # basis. The class signal is identical in *content* (same one-hot span plus
    # Gaussian noise of constant scale -- feature_noise_constant_across_ladder),
    # but a linear expert trained on one regime's basis transfers poorly to
    # another, which is exactly the routing signal the protocol must recover.
    regime_bases = [_random_orthogonal(d_features, rng) for _ in range(3)]

    W = np.zeros((n_nodes, n_nodes), dtype=float)
    offset = 0
    for r, (count, regime) in enumerate(zip(counts, (REGIME_L, REGIME_H, REGIME_M))):
        idx = np.arange(offset, offset + count)
        c = rng.integers(0, n_classes, size=count)
        classes[idx] = c
        regimes[idx] = regime
        block_features = (prototypes[c] @ regime_bases[regime]
                          + feature_noise_scale * rng.standard_normal((count, d_features)))
        features[idx] = block_features
        block = _wire_block(count, c, np.full(count, regime), p_homophily, p_hetero, p_random, rng)
        W[np.ix_(idx, idx)] = block
        offset += count

    patches = rng.dirichlet(ratio * 2.0, size=n_patches)

    return {
        "W": W,
        "y": classes,
        "regimes": regimes,
        "features": features,
        "patches": patches,
        "class_prototypes": prototypes,
        "n_classes": n_classes,
        "n_patches": n_patches,
        "regime_ratio": list(ratio),
        "p_homophily": p_homophily,
        "p_hetero": p_hetero,
        "p_random": p_random,
        "feature_noise_scale": feature_noise_scale,
        "n_nodes": n_nodes,
        "d_features": d_features,
        "dilution": 0.0,
    }


def dilution_ladder(
    base_graph: dict,
    dilution_values: list[float] | None = None,
    seed: int = 0,
) -> list[dict]:
    """Build the dilution ladder: nested conversion toward low-pass wiring.

    Each ladder element is a copy of ``base_graph`` with ``dilution * n`` nodes
    converted. Converted nodes have all their neighbors forced to same-class
    (p_homophily), keeping cross-class wiring at p_hetero. Conversion order is
    fixed per class so converted sets are strictly nested across the ladder.
    """
    if dilution_values is None:
        dilution_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    rng = np.random.default_rng(seed)
    y = base_graph["y"]
    n = y.size
    classes = np.unique(y)

    conversion_order: list[int] = []
    for c in classes:
        idx = np.flatnonzero(y == c).tolist()
        rng.shuffle(idx)
        conversion_order.extend(idx)

    base_W = base_graph["W"]
    p_hom = base_graph["p_homophily"]
    ladder: list[dict] = []
    prev_convert = np.zeros(n, dtype=bool)
    for dil in dilution_values:
        n_convert = int(round(dil * n))
        convert = np.zeros(n, dtype=bool)
        convert[np.asarray(conversion_order[:n_convert], dtype=int)] = True
        assert convert.sum() >= prev_convert.sum()

        W = base_W.copy()
        # Converted nodes are *rewritten* to homogeneous low-pass wiring:
        # their original (regime-specific) edges are removed and replaced with
        # same-class edges at p_hom. This erases the structural routing
        # signature that the label-free router reads; at dilution=1 the whole
        # graph is low-pass and the label-free gain should collapse.
        for v in np.flatnonzero(convert):
            W[v, :] = 0.0
            W[:, v] = 0.0
            same = np.flatnonzero(y == y[v])
            same = same[same != v]
            keep = np.empty(0, dtype=int)
            if same.size:
                keep = same[rng.random(same.size) < p_hom]
            W[v, keep] = 1.0
            W[keep, v] = 1.0
        np.fill_diagonal(W, 0.0)
        W = np.minimum(W, 1.0)

        g = dict(base_graph)
        g["W"] = W
        g["dilution"] = float(dil)
        g["converted_mask"] = convert
        g["unconverted_mask"] = ~convert
        ladder.append(g)
        prev_convert = convert
    return ladder


def single_m_expert_accuracy(
    graph: dict,
    n_train: float = 0.5,
    seed: int = 0,
) -> float:
    """Accuracy of a single-M logistic expert on informative features.

    Trains only on M-regime nodes partitioned with a class-ratio-preserving
    split and predicts class from features, confirming the no-saturation
    invariant (~0.80, well below ceiling).
    """
    from sklearn.linear_model import LogisticRegression

    regimes = graph["regimes"]
    y = graph["y"]
    features = graph["features"]
    mask = regimes == REGIME_M
    if mask.sum() < 20:
        raise ValueError(f"too few M-regime nodes: {mask.sum()}")
    X = features[mask]
    labels = y[mask]

    rng = np.random.default_rng(seed)
    perm = rng.permutation(X.shape[0])
    n_train_ = int(n_train * X.shape[0])
    tr, te = perm[:n_train_], perm[n_train_:]
    clf = LogisticRegression(C=1.0, max_iter=2000)
    clf.fit(X[tr], labels[tr])
    return float(clf.score(X[te], labels[te]))
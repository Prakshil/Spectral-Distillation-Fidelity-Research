"""Synthetic ERP attention graphs: stochastic block models with planted fraud.

Emulates the failure mode from the guide: strong supplier/invoice backbone plus
weak customer edges whose collective importance is invisible to thresholding.
"""

from __future__ import annotations

import numpy as np

from spectral_distillation.src.laplacian import build_adjacency


def _one_hot_prototypes(n_classes: int, d_features: int, rng: np.random.Generator) -> np.ndarray:
    if d_features < n_classes:
        raise ValueError(f"d_features ({d_features}) must be >= n_classes ({n_classes})")
    protos = np.zeros((n_classes, d_features))
    for c in range(n_classes):
        protos[c, c] = 1.0
    return protos


def stochastic_block_model(
    n: int,
    n_blocks: int,
    p_in: float = 0.5,
    p_out: float = 0.05,
    weight_range: tuple[float, float] = (0.05, 1.0),
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    sizes = np.array_split(np.arange(n), n_blocks)
    blocks = np.zeros(n, dtype=int)
    for block_id, idx in enumerate(sizes):
        blocks[idx] = block_id

    A = np.zeros((n, n))
    lo, hi = weight_range
    for i in range(n):
        for j in range(i + 1, n):
            prob = p_in if blocks[i] == blocks[j] else p_out
            if rng.random() < prob:
                A[i, j] = A[j, i] = rng.uniform(lo, hi)
    return build_adjacency(A), blocks


def _connectivity_backbone(A: np.ndarray, rng: np.random.Generator, weight: float = 0.3) -> None:
    n = A.shape[0]
    perm = rng.permutation(n)
    for step in range(n - 1):
        i, j = perm[step], perm[step + 1]
        A[i, j] = max(A[i, j], weight)
        A[j, i] = A[i, j]


def make_erp_attention(n: int, seed: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    A = np.zeros((n, n))
    labels = np.zeros(n, dtype=int)
    cluster = np.zeros(n, dtype=int)

    if n >= 12:
        n_suppliers = max(1, int(0.2 * n))
        n_invoices = max(1, int(0.35 * n))
        n_shell = max(1, int(0.05 * n))
        n_customers = max(1, n - n_suppliers - n_invoices - n_shell)
    else:
        n_suppliers = max(1, n // 4)
        n_invoices = max(1, n // 4)
        n_shell = max(1, n // 4)
        n_customers = max(1, n - n_suppliers - n_invoices - n_shell)

    roles = [0] * n_suppliers + [1] * n_invoices + [2] * n_shell + [3] * n_customers
    roles = np.array(rng.permutation(roles))
    supplier_idx = np.where(roles == 0)[0]
    invoice_idx = np.where(roles == 1)[0]
    shell_idx = np.where(roles == 2)[0]
    customer_idx = np.where(roles == 3)[0]

    def connect(idx_a, idx_b, prob, rng, lo, hi):
        for i in idx_a:
            for j in idx_b:
                if i < j and rng.random() < prob:
                    A[i, j] = A[j, i] = rng.uniform(lo, hi)

    connect(supplier_idx, supplier_idx, 0.6, rng, 0.6, 1.0)
    connect(invoice_idx, invoice_idx, 0.4, rng, 0.4, 1.0)
    connect(customer_idx, customer_idx, 0.3, rng, 0.1, 0.5)
    connect(supplier_idx, invoice_idx, 0.5, rng, 0.4, 1.0)
    connect(supplier_idx, customer_idx, 0.2, rng, 0.05, 0.3)
    connect(invoice_idx, customer_idx, 0.2, rng, 0.05, 0.3)
    connect(supplier_idx, shell_idx, 0.8, rng, 0.7, 1.0)
    connect(customer_idx, shell_idx, 0.2, rng, 0.05, 0.2)

    _connectivity_backbone(A, rng)

    labels[shell_idx] = 1
    cluster[roles == 1] = 1
    cluster[roles == 2] = 2
    cluster[roles == 3] = 3

    A = build_adjacency(A)
    return {
        "attention": A,
        "roles": roles,
        "labels": labels,
        "cluster_ids": cluster,
        "n_suppliers": n_suppliers,
        "n_invoices": n_invoices,
        "n_shell": n_shell,
        "n_customers": n_customers,
    }


_ROLE_SUPPLIER, _ROLE_INVOICE, _ROLE_SHELL, _ROLE_CUSTOMER = 0, 1, 2, 3

# Regime mapping for the ERP-fraud graph (protocol oracle contract):
#   0 = L low-pass  : suppliers + invoices (homophilic backbone)
#   1 = H high-pass : shell companies (anti-correlated fraud ring)
#   2 = M noisy     : customers (weak edges, informative features)
_ROLE_REGIME = {
    _ROLE_SUPPLIER: 0,
    _ROLE_INVOICE: 0,
    _ROLE_SHELL: 1,
    _ROLE_CUSTOMER: 2,
}

# Per-role-pair wiring: (base probability, weight range).
_ROLE_WIRING = {
    (_ROLE_SUPPLIER, _ROLE_SUPPLIER): (0.6, (0.6, 1.0)),
    (_ROLE_INVOICE, _ROLE_INVOICE): (0.4, (0.4, 1.0)),
    (_ROLE_SUPPLIER, _ROLE_INVOICE): (0.5, (0.4, 1.0)),  # transaction backbone
    (_ROLE_SHELL, _ROLE_SHELL): (0.05, (0.05, 0.3)),  # anti-correlated fraud ring
    (_ROLE_CUSTOMER, _ROLE_CUSTOMER): (0.3, (0.1, 0.5)),  # weak, class-independent
    (_ROLE_SUPPLIER, _ROLE_SHELL): (0.8, (0.7, 1.0)),  # fraud hub edges
    (_ROLE_INVOICE, _ROLE_SHELL): (0.8, (0.7, 1.0)),  # fraud hub edges
    (_ROLE_SUPPLIER, _ROLE_CUSTOMER): (0.2, (0.05, 0.3)),
    (_ROLE_INVOICE, _ROLE_CUSTOMER): (0.2, (0.05, 0.3)),
    (_ROLE_SHELL, _ROLE_CUSTOMER): (0.2, (0.05, 0.2)),
}

_ROLE_NAMES = {0: "supplier", 1: "invoice", 2: "shell", 3: "customer"}


def generate_erp_fraud_graph(
    n_nodes: int = 4_000,
    n_classes: int = 5,
    d_features: int = 16,
    n_patches: int = 50,
    feature_noise_scale: float = 0.4,
    p_homophily: float = 0.25,
    p_hetero: float = 0.02,
    p_random: float = 0.3,
    seed: int = 0,
) -> dict:
    """Generate a scalable synthetic ERP-fraud graph for the D1-D4 protocol.

    Mirrors ``make_erp_attention``'s stochastic block semantics (strong
    supplier/invoice backbone, weak customer edges, dense shell-fraud hubs)
    but with vectorized wiring so it scales to thousands of nodes, and returns
    the protocol contract (:key:`W`, :key:`features`, :key:`y`,
    :key:`regimes`) so ``run_protocol``'s oracle/random/uniform/label-free
    conditions work unchanged.

    Roles -> regimes: supplier/invoice -> L(0), shell -> H(1), customer ->
    M(2). Class labels are shared across regimes and projected through
    regime-specific orthogonal bases (same deconfounding construction as
    PC-1c-R), so a linear expert trained in one regime transfers poorly to
    another -- the routing signal the protocol must recover.

    Edge wiring:
    - L-L backbone edges are class-homophilic (same class p_homophily, else
      p_hetero); supplier<->invoice transaction edges are dense and
      class-independent.
    - H-H shell edges are anti-correlated (class c wired to c+1).
    - supplier/invoice <-> shell edges are dense and class-independent -> the
      fraud-hub signature.
    - all customer edges are weak and class-independent (M noisy regime).

    Returns a dict with ``W``, ``y``, ``regimes``, ``features``, ``roles``,
    ``fraud`` (shell mask), ``n_classes``, ``n_patches``, ``regime_ratio``,
    plus the shared prototype/basis artifacts for diagnostics.
    """
    rng = np.random.default_rng(seed)
    if n_nodes < 12:
        raise ValueError(f"n_nodes must be >= 12, got {n_nodes}")

    n_suppliers = max(1, int(0.20 * n_nodes))
    n_invoices = max(1, int(0.35 * n_nodes))
    n_shell = max(1, int(0.05 * n_nodes))
    n_customers = max(1, n_nodes - n_suppliers - n_invoices - n_shell)

    roles = np.zeros(n_nodes, dtype=int)
    roles[:n_suppliers] = _ROLE_SUPPLIER
    roles[n_suppliers : n_suppliers + n_invoices] = _ROLE_INVOICE
    roles[n_suppliers + n_invoices : n_suppliers + n_invoices + n_shell] = _ROLE_SHELL
    roles[n_suppliers + n_invoices + n_shell :] = _ROLE_CUSTOMER
    roles = rng.permutation(roles)

    regimes = np.array([_ROLE_REGIME[r] for r in roles], dtype=int)
    classes = rng.integers(0, n_classes, size=n_nodes)
    fraud = (roles == _ROLE_SHELL).astype(int)

    # Shared class prototypes projected through regime-specific bases (as PC-1c-R).
    prototypes = _one_hot_prototypes(n_classes, d_features, rng)
    regime_bases = _random_orthogonal_bases(d_features, 3, rng)
    features = np.zeros((n_nodes, d_features))
    for r in range(3):
        mask = regimes == r
        features[mask] = (
            prototypes[classes[mask]] @ regime_bases[r]
            + feature_noise_scale * rng.standard_normal((int(mask.sum()), d_features))
        )

    W = np.zeros((n_nodes, n_nodes), dtype=float)
    for (ra, rb), (prob, (lo, hi)) in _ROLE_WIRING.items():
        ia = np.flatnonzero(roles == ra)
        ib = np.flatnonzero(roles == rb)
        if ia.size == 0 or ib.size == 0:
            continue
        if ra == rb:
            a, b = np.triu_indices(ia.size, k=1)
            _write_block(
                W, ia[a], ia[b], prob, lo, hi,
                classes[ia[a]], classes[ia[b]], regimes[ia[a]],
                rng, p_homophily, p_hetero, p_random, n_classes,
            )
        else:
            gx, gy = np.meshgrid(ia, ib, indexing="ij")
            _write_block(W, gx.ravel(), gy.ravel(), prob, lo, hi, None, None, None,
                         rng, p_homophily, p_hetero, p_random, n_classes)

    _connectivity_backbone(W, rng)
    W = build_adjacency(W)

    ratio = np.array([0.55, 0.05, 0.40], dtype=float)
    n_users = int(np.count_nonzero(regimes == 2))
    ratio[2] = n_users / max(1, n_nodes)
    ratio[0] = 1.0 - ratio[1] - ratio[2]
    patches = rng.dirichlet(np.maximum(ratio, 1e-3) * 2.0, size=n_patches)

    return {
        "W": W,
        "y": classes,
        "regimes": regimes,
        "features": features,
        "roles": roles,
        "fraud": fraud,
        "role_names": _ROLE_NAMES,
        "patches": patches,
        "class_prototypes": prototypes,
        "regime_bases": regime_bases,
        "n_classes": n_classes,
        "n_patches": n_patches,
        "regime_ratio": ratio.tolist(),
        "p_homophily": p_homophily,
        "p_hetero": p_hetero,
        "p_random": p_random,
        "feature_noise_scale": feature_noise_scale,
        "n_nodes": n_nodes,
        "d_features": d_features,
        "dilution": 0.0,
    }


def _random_orthogonal_bases(d: int, n_bases: int, rng: np.random.Generator) -> list[np.ndarray]:
    return [_orthogonal_basis(d, rng) for _ in range(n_bases)]


def _orthogonal_basis(d: int, rng: np.random.Generator) -> np.ndarray:
    if d == 1:
        return np.array([[1.0]])
    A = rng.standard_normal((d, d))
    Q, _ = np.linalg.qr(A)
    return Q


def _write_block(
    W: np.ndarray,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    prob_base: float,
    lo: float,
    hi: float,
    ca: np.ndarray | None,
    cb: np.ndarray | None,
    reg_a: np.ndarray | None,
    rng: np.random.Generator,
    p_homophily: float,
    p_hetero: float,
    p_random: float,
    n_classes: int,
) -> None:
    """Write a symmetric role-pair block into ``W`` with per-pair class wiring.

    When ``ca``/``cb``/``reg_a`` are provided (same-role block), within-regime
    edges become class-conditioned: L homophilic, H anti-correlated (class c to
    (c+1) mod n_classes), M class-independent. Cross-role blocks use
    ``prob_base`` for every pair (the fraud-hub and weak-customer edges are
    class-independent).
    """
    if idx_a.size != idx_b.size:
        raise ValueError("block index arrays must be the same length")
    n_pairs = idx_a.size
    if n_pairs == 0:
        return
    probs = np.full(n_pairs, prob_base)
    if ca is not None and cb is not None and reg_a is not None:
        same = ca == cb
        anti = cb == (ca + 1) % max(2, n_classes)
        is_L = reg_a == 0
        is_H = reg_a == 1
        probs[is_L] = np.where(same[is_L], p_homophily, p_hetero)
        probs[is_H] = np.where(anti[is_H], p_homophily, p_hetero)
        probs[~is_L & ~is_H] = p_random
    keep = rng.random(n_pairs) < probs
    if not np.any(keep):
        return
    ia, ib = idx_a[keep], idx_b[keep]
    w = rng.uniform(lo, hi, ia.size)
    W[ia, ib] = w
    W[ib, ia] = w
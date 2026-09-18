"""Synthetic ERP attention graphs: stochastic block models with planted fraud.

Emulates the failure mode from the guide: strong supplier/invoice backbone plus
weak customer edges whose collective importance is invisible to thresholding.
"""

from __future__ import annotations

import numpy as np

from spectral_distillation.src.laplacian import build_adjacency


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
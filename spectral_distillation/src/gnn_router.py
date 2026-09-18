"""Trainable GNN router that maps structural node fields to expert assignments.

Implements a Message-Passing (GraphSAGE-style) router: node features
{hx, spectral_ratio, log_degree, clustering} are aggregated over the graph and
projected to a soft assignment over the expert channels
[low_pass, high_pass, identity]. Used by the label-free condition of the
D1-D4 evaluation protocol; the oracle/random/uniform conditions freeze their
assignment instead of learning it.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from spectral_distillation.src.spectral_position import spectral_ratio

EXPERT_TYPES = ("low_pass", "high_pass", "identity")


def build_router_gnn(
    n_features: int,
    hidden_dim: int = 64,
    n_layers: int = 3,
    n_experts: int = 3,
    dropout: float = 0.1,
) -> nn.Module:
    """Build the routing GNN; expects arrays, returns a Torch module."""
    model = RouterGNN(
        n_features_in=n_features,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_experts=n_experts,
        dropout=dropout,
    )
    return model


class RouterGNN(nn.Module):
    def __init__(
        self,
        n_features_in: int,
        hidden_dim: int = 64,
        n_layers: int = 3,
        n_experts: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers = []
        in_dim = n_features_in
        for i in range(n_layers):
            out_dim = hidden_dim if i < n_layers - 1 else n_experts
            layers.append(
                nn.Linear(in_dim if i == 0 else hidden_dim, out_dim)
            )
        self.linears = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, W):
        if isinstance(W, torch.Tensor) and W.layout == torch.sparse_coo:
            def agg(h):
                return torch.sparse.mm(W, h)
        else:
            def agg(h):
                return torch.mm(W, h)
        h = x
        for layer in self.linears:
            mixed = agg(h) + h
            h = layer(mixed)
            if layer is not self.linears[-1]:
                h = F.relu(h)
                h = self.dropout(h)
        return F.softmax(h, dim=-1)


_SPARSE_DENSE_THRESHOLD = 3000


def _to_sparse_tensor(W_norm: np.ndarray) -> torch.Tensor:
    W_norm = np.asarray(W_norm, dtype=float)
    coo = W_norm != 0
    rows, cols = np.nonzero(coo)
    vals = W_norm[rows, cols]
    idx = torch.LongTensor(np.stack([rows, cols], axis=0))
    return torch.sparse_coo_tensor(idx, torch.FloatTensor(vals), W_norm.shape)


def _as_torch_W(W_norm: np.ndarray) -> torch.Tensor:
    """Dense for small graphs, sparse COO past the threshold."""
    if W_norm.size > _SPARSE_DENSE_THRESHOLD * _SPARSE_DENSE_THRESHOLD:
        return _to_sparse_tensor(W_norm)
    return torch.FloatTensor(np.asarray(W_norm, dtype=float))


def route_nodes(pi: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Convert soft assignment logits into a normalized routing matrix.

    ``pi`` is the (n, K) per-node soft assignment; normalization is a no-op
    when the router already emits probabilities, and guards against float noise.
    """
    pi = np.asarray(pi, dtype=float)
    row_sum = pi.sum(axis=1, keepdims=True)
    normalized = pi / np.maximum(row_sum, 1e-12)
    return normalized


def gnn_router_forward(model: nn.Module, x: np.ndarray, W_norm: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Apply the router to numpy inputs and return a numpy assignment."""
    model.eval()
    with torch.no_grad():
        xt = torch.as_tensor(x, dtype=torch.float32, device=device)
        Wt = _as_torch_W(W_norm).to(device)
        out = model(xt, Wt).cpu().numpy()
    return route_nodes(out, x)


def train_condition(
    x: np.ndarray,
    W: np.ndarray,
    W_norm: np.ndarray,
    target_assignment: np.ndarray,
    hidden_dim: int = 64,
    n_layers: int = 3,
    n_experts: int = 3,
    epochs: int = 200,
    lr: float = 1e-2,
    dropout: float = 0.1,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[nn.Module, list[float]]:
    """Train the label-free router to mimic an assignment signal (imitation).

    Used only to produce the *learned* assignment: the router is trained on the
    full graph against a supervision target (e.g., oracle buckets or a spectral
    positional target), so the assignment can be evaluated under frozen-model
    interventions without peeking at labels beyond the target.

    Returns the trained module and the recorded training losses.
    """
    torch.manual_seed(seed)
    model = RouterGNN(
        n_features_in=x.shape[1],
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_experts=n_experts,
        dropout=dropout,
    ).to(device)
    xt = torch.as_tensor(x, dtype=torch.float32, device=device)
    Wt = _as_torch_W(W_norm).to(device)
    target = torch.as_tensor(target_assignment, dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses: list[float] = []
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(xt, Wt)
        loss = F.kl_div((pred + 1e-8).log(), target, reduction="batchmean")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return model, losses


def make_structural_features(
    W: np.ndarray,
    V: np.ndarray,
    evals: np.ndarray,
    hx: np.ndarray | None = None,
    clustering: np.ndarray | None = None,
) -> np.ndarray:
    """Assemble the {hx, spectral_ratio, log_degree, clustering} router fields.

    - ``spectral_ratio`` is the normalized spectral position s_spectral(v).
    - ``log_degree`` is log(1 + degree).
    - ``clustering`` (local clustering coefficient) defaults to a degree-based
      proxy when not provided.
    - ``hx`` (feature homophily) defaults to zero when not provided.
    """
    degrees = W.sum(axis=1)
    spectral_pos = spectral_ratio(spectral_density(V, evals))
    log_degree = np.log1p(degrees)
    if clustering is None:
        clustering = degree_clustering_proxy(degrees)
    if hx is None:
        hx = np.zeros(W.shape[0])
    return np.stack([hx, spectral_pos, log_degree, clustering], axis=1)


def spectral_density(V: np.ndarray, evals: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Per-node spectral density s(v) = sum_k v_k(v)^2 * lambda_k."""
    V = np.asarray(V, dtype=float)
    evals = np.asarray(evals, dtype=float)
    return (V ** 2) @ evals + eps


def degree_clustering_proxy(degrees: np.ndarray) -> np.ndarray:
    """Degree-based local-clustering proxy: (d-2)/(d+1) clipped to [0, 1]."""
    degrees = np.asarray(degrees, dtype=float)
    proxy = (degrees - 2.0) / (degrees + 1.0)
    return np.clip(proxy, 0.0, 1.0)


def expert_mask_from_assignment(assignment: np.ndarray, expert_index: int) -> np.ndarray:
    """Boolean mask of nodes routed to a given expert."""
    assignment = np.asarray(assignment, dtype=float)
    return assignment[:, expert_index] >= assignment.max(axis=1) * 0.99
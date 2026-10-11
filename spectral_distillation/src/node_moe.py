"""Node-MoE (arXiv:2406.03464): faithful expert construction + honest scoring.

Motivation
----------
Every routing scored elsewhere in this repo uses *weak or generic* experts: a
per-row logistic/MLP head, or a single-hop GraphSAGE head. The standing
objection is that the negative routing result is an artifact of the experts, not
of the graphs. The published state of the art for node-wise mixture-of-experts on
graphs is **Node-MoE**, and its distinctive choice is the *expert construction*:
each expert is a **ChebNetII** spectral filter (learnable Chebyshev coefficients,
He et al. 2022), and the experts are deliberately initialised as *different*
filter types -- low-pass, high-pass, band-pass, ... -- before training. The gate
is a small GNN whose input is the neighbourhood-discrepancy descriptor
``[X, |AX-X|, |A^2X-X|]``, so neighbouring nodes tend to pick the same expert.

`learned_router.py` already reuses Node-MoE's *gate features* but keeps the
weak per-row experts and a frozen k-means pool. This module closes the remaining
gap: it reproduces the *experts* and the *gate*, and it scores the whole thing
under the same honest fixed-expert (Protocol B) protocol used for every other
router here.

What is reproduced
------------------
* ``ChebNetIIFilter`` -- the learnable Chebyshev filter of ChebNetII. The filter
  is parameterised by coefficients ``gamma`` at the ``K+1`` Chebyshev nodes and
  evaluated by Chebyshev interpolation, exactly as in the paper. Because the
  nodes are the Chebyshev-Gauss points, initialising ``gamma = g(nodes)`` makes
  the filter the Chebyshev interpolant of the target shape ``g`` -- this is how
  differentiated initialisation is done (``differentiated_filter_types``).
* ``NodeMoEGate`` -- a one-hop low-pass GNN over ``[X, |AX-X|, |A^2X-X|]``.
* ``NodeMoEHead`` -- the joint model: ``y_hat = h( sum_o g_o z_o )`` trained
  end-to-end with the task loss plus Node-MoE's filter-smoothing loss
  ``L_s = sum_o sum_i (f_o(x_i) - f_o(x_{i-1}))^2``.
* ``evaluate_node_moe_protocol_b`` -- the honest cross-fit protocol: the pool of
  differentiated ChebNetII experts is fitted **routing-blind** (each expert, with
  a distinct filter init, on every fit node), then a gate is trained on a
  **disjoint** router half and scored against a genuine no-routing model.

Two ways the published method is tested
---------------------------------------
1. **Faithful replication** (``NodeMoEHead``): train gate+experts jointly on the
   training split, score on test -- the paper's own protocol -- against a single
   global ChebNetII and a global MLP.
2. **Protocol B** (``evaluate_node_moe_protocol_b``): freeze the differentiated
   expert pool, cross-fit the gate, and ask whether *any* routing beats one
   global model. This is the same protocol that produced the repo's 0/234 null.

If Node-MoE wins under (1) but not (2), the gain is co-adaptation / ensemble
capacity rather than genuine per-node expert selection -- the central claim of
the evaluation-critique line.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.dummy import DummyClassifier

from spectral_distillation.src.gnn_router import _as_torch_W
from spectral_distillation.src.learned_router import (
    build_router_targets,
    uniform_ensemble_accuracy,
)

# Ordered by how distinct each target shape is; ``differentiated_filter_types``
# takes the first ``m``. low/high are the canonical homophily/heterophily pair,
# band-pass/band-reject probe the middle of the spectrum, comb and constant add
# further independent shapes.
FILTER_TYPES = ("low_pass", "high_pass", "band_pass", "band_reject", "comb", "constant")


def differentiated_filter_types(n_experts: int) -> list[str]:
    """Pick ``n_experts`` distinct filter shapes, most-signal-bearing first."""
    if n_experts < 1:
        raise ValueError("need at least one expert")
    if n_experts > len(FILTER_TYPES):
        raise ValueError(
            f"at most {len(FILTER_TYPES)} distinct filter types are defined, got {n_experts}"
        )
    return list(FILTER_TYPES[:n_experts])


def chebyshev_nodes(K: int) -> np.ndarray:
    """Chebyshev-Gauss interpolation nodes on ``(-1, 1)``: ``cos((j+.5)pi/(K+1))``."""
    if K < 1:
        raise ValueError("Chebyshev order K must be >= 1")
    j = np.arange(K + 1)
    return np.cos((j + 0.5) * np.pi / (K + 1))


def chebyshev_matrix(x: np.ndarray, K: int) -> np.ndarray:
    """``(K+1, len(x))`` matrix of ``T_k(x)`` for ``k = 0..K``."""
    x = np.asarray(x, dtype=float)
    T = np.zeros((K + 1, x.size), dtype=float)
    T[0] = 1.0
    if K >= 1:
        T[1] = x
    for k in range(2, K + 1):
        T[k] = 2.0 * x * T[k - 1] - T[k - 2]
    return T


def target_filter(name: str, x: np.ndarray) -> np.ndarray:
    """Target spectral response evaluated at points ``x`` in ``[-1, 1]``.

    The domain is the rescaled eigenvalue axis of the symmetric normalized
    Laplacian: ``x = -1`` is the low-frequency (constant / smooth) end and
    ``x = +1`` the high-frequency end. A *low-pass* filter must therefore be
    large at ``-1``; a *high-pass* filter large at ``+1``.
    """
    x = np.asarray(x, dtype=float)
    if name == "low_pass":
        return (1.0 - x) / 2.0
    if name == "high_pass":
        return (1.0 + x) / 2.0
    if name == "band_pass":
        return np.exp(-4.0 * x ** 2)
    if name == "band_reject":
        return 1.0 - np.exp(-4.0 * x ** 2)
    if name == "comb":
        return 0.5 + 0.5 * np.cos(2.0 * np.pi * x)
    if name == "constant":
        return np.ones_like(x)
    raise ValueError(f"unknown filter type {name!r}")


class ChebNetIIFilter(nn.Module):
    """Learnable ChebNetII filter with Chebyshev-interpolation evaluation.

    ``gamma`` are the ``K+1`` free coefficients of the Gauss-Chebyshev
    interpolation form

        f(L) = sum_k h_k T_k(L),   h_k = (2/(K+1)) sum_j gamma_j T_k(x_j)

    with the ``k = 0`` coefficient halved (``h_0 = (1/(K+1)) sum_j gamma_j``),
    which is the standard Gauss-Chebyshev convention. Under it, initialising
    ``gamma = g(nodes)`` makes ``f`` the *exact* interpolant of the target shape
    ``g`` at the nodes -- which is precisely how Node-MoE's differentiated
    initialisation (low-pass / constant / high-pass experts) is realised.
    """

    def __init__(self, K: int, filter_type: str = "low_pass") -> None:
        super().__init__()
        self.K = int(K)
        nodes = chebyshev_nodes(self.K)
        self.register_buffer("nodes", torch.tensor(nodes, dtype=torch.float32))
        self.register_buffer(
            "T_nodes", torch.tensor(chebyshev_matrix(nodes, self.K), dtype=torch.float32)
        )
        init = target_filter(filter_type, nodes)
        self.gamma = nn.Parameter(torch.tensor(init, dtype=torch.float32))
        self.filter_type = filter_type

    def poly_coeffs(self) -> torch.Tensor:
        """Polynomial coefficients ``h_k`` for ``k = 0..K`` (shape ``(K+1,)``).

        Gauss-Chebyshev interpolation: full weight ``2/(K+1)`` on ``k >= 1`` and
        half weight ``1/(K+1)`` on ``k = 0``.
        """
        raw = self.T_nodes @ self.gamma
        weight = torch.ones_like(raw)
        weight[0] = 0.5
        return (2.0 / (self.K + 1)) * weight * raw

    def response(self) -> torch.Tensor:
        """Filter values ``f`` at the Chebyshev nodes (shape ``(K+1,)``)."""
        return self.T_nodes.t() @ self.poly_coeffs()

    def forward(self, basis: list[torch.Tensor]) -> torch.Tensor:
        """Apply ``f(L)`` to precomputed basis ``[T_0(L)X, .., T_K(L)X]``."""
        coeffs = self.poly_coeffs()
        out = torch.zeros_like(basis[0])
        for k, Tk in enumerate(basis):
            out = out + coeffs[k] * Tk
        return out


def chebnetii_basis(
    Wt: torch.Tensor, X: torch.Tensor, K: int
) -> list[torch.Tensor]:
    """Chebyshev basis ``[T_0(L)X, .., T_K(L)X]`` for the symmetric normalised
    Laplacian ``L = I - A_hat``, so ``L_scaled = L - I = -A_hat``.

    ``Wt`` is the cached (dense or sparse-COO) normalised adjacency ``A_hat``.
    The basis does not depend on any learnable parameter, so it is computed once
    and shared by every expert and every training step.
    """
    def prop(h: torch.Tensor) -> torch.Tensor:
        if Wt.layout == torch.sparse_coo:
            return torch.sparse.mm(Wt, h)
        return torch.mm(Wt, h)

    basis = [X]
    if K >= 1:
        basis.append(-prop(X))  # T_1 on L_scaled = -A_hat
    for _ in range(2, K + 1):
        # T_k = 2 * L_scaled * T_{k-1} - T_{k-2}, with L_scaled = -A_hat
        basis.append(-2.0 * prop(basis[-1]) - basis[-2])
    return basis


def gate_features(Wt: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """Node-MoE gate input ``[X, |AX-X|, |A^2X-X|]`` (label-free)."""
    def prop(h: torch.Tensor) -> torch.Tensor:
        if Wt.layout == torch.sparse_coo:
            return torch.sparse.mm(Wt, h)
        return torch.mm(Wt, h)

    AX = prop(X)
    A2X = prop(AX)
    return torch.cat([X, torch.abs(AX - X), torch.abs(A2X - X)], dim=-1)


class ChebNetIIModule(nn.Module):
    """One ChebNetII expert head: ``readout([X, f(L)X])``.

    The filter term ``f(L)X`` is the spectral (structural) component; concatenating
    the raw features keeps the expert strong on feature-driven labels, so a weak
    expert cannot be blamed for a routing null.
    """

    def __init__(
        self,
        n_features: int,
        K: int,
        filter_type: str,
        skip: bool = True,
        hidden_dim: int = 0,
    ) -> None:
        super().__init__()
        self.filter = ChebNetIIFilter(K, filter_type)
        self.skip = skip
        in_dim = 2 * n_features if skip else n_features
        self.hidden_dim = hidden_dim
        if hidden_dim > 0:
            self.lin = nn.Linear(in_dim, hidden_dim)
            self.out = nn.Linear(hidden_dim, 1)
        else:
            self.out = nn.Linear(in_dim, 1)

    def forward(self, basis: list[torch.Tensor]) -> torch.Tensor:
        filtered = self.filter(basis)
        x = basis[0]
        h = torch.cat([x, filtered], dim=-1) if self.skip else filtered
        if self.hidden_dim > 0:
            h = F.relu(self.lin(h))
        return self.out(h).squeeze(-1)


class ChebNetIIHead:
    """A single ChebNetII expert exposing the ``fit_graph``/``predict_nodes`` hooks.

    Subset-restricted loss (``train_idx``) mirrors
    :class:`~spectral_distillation.src.gnn_expert.GNNExpertHead`, so it drops
    straight into :func:`~spectral_distillation.src.mixture.fit_experts` and the
    Protocol B runner. Message passing is transductive over the full graph (test
    *features* participate, test *labels* never do).
    """

    def __init__(
        self,
        W_norm: np.ndarray,
        n_features: int,
        train_idx: np.ndarray,
        K: int = 2,
        filter_type: str = "low_pass",
        hidden_dim: int = 0,
        epochs: int = 300,
        lr: float = 1e-2,
        weight_decay: float = 1e-3,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        self.W_norm = W_norm
        self.n_features = n_features
        train_idx = np.asarray(train_idx)
        self.train_idx = (
            np.flatnonzero(train_idx) if train_idx.dtype == bool
            else train_idx.astype(int)
        )
        self.K = K
        self.filter_type = filter_type
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        self.device = device
        self._Wt: torch.Tensor | None = None
        self._basis: list[torch.Tensor] | None = None
        self._model: ChebNetIIModule | None = None
        self._fallback: DummyClassifier | None = None

    def _W(self) -> torch.Tensor:
        if self._Wt is None:
            self._Wt = _as_torch_W(self.W_norm).to(self.device)
        return self._Wt

    def _basis_features(self, X: np.ndarray) -> list[torch.Tensor]:
        if self._basis is None:
            xt = torch.as_tensor(np.asarray(X, dtype=np.float32), device=self.device)
            self._basis = chebnetii_basis(self._W(), xt, self.K)
        return self._basis

    def fit_graph(self, X: np.ndarray, y: np.ndarray) -> "ChebNetIIHead":
        classes = np.unique(y[self.train_idx])
        if classes.size < 2:
            self._fallback = DummyClassifier(strategy="most_frequent")
            self._fallback.fit(X[self.train_idx], y[self.train_idx])
            return self

        torch.manual_seed(self.seed)
        self._model = ChebNetIIModule(
            self.n_features, self.K, self.filter_type, hidden_dim=self.hidden_dim
        ).to(self.device)
        basis = self._basis_features(X)
        yt = torch.as_tensor(np.asarray(y, dtype=np.float32), device=self.device)
        idx = torch.as_tensor(self.train_idx, dtype=torch.long, device=self.device)

        pos = float(yt[idx].sum().item())
        neg = float(idx.numel() - pos)
        pos_weight = torch.tensor(
            [neg / pos if pos > 0 else 1.0], device=self.device, dtype=torch.float32
        )
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.Adam(
            self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self._model.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = loss_fn(self._model(basis)[idx], yt[idx])
            loss.backward()
            opt.step()
        self._model.eval()
        return self

    def _logits_nodes(self, X: np.ndarray, idx: np.ndarray) -> np.ndarray:
        assert self._model is not None, "fit_graph must be called before prediction"
        with torch.no_grad():
            logits = self._model(self._basis_features(X))
            sel = torch.as_tensor(np.asarray(idx, dtype=np.int64), device=self.device)
            return logits[sel].cpu().numpy().astype(float)

    def predict_nodes(self, X: np.ndarray, idx: np.ndarray) -> np.ndarray:
        if self._fallback is not None:
            return self._fallback.predict(np.asarray(X)[idx])
        return (self._logits_nodes(X, idx) > 0).astype(np.int64)

    def predict_proba_nodes(self, X: np.ndarray, idx: np.ndarray) -> np.ndarray:
        if self._fallback is not None:
            return np.asarray(self._fallback.predict_proba(np.asarray(X)[idx]), dtype=float)
        return 1.0 / (1.0 + np.exp(-self._logits_nodes(X, idx)))[:, None]


def cheb_head_factory(
    W_norm: np.ndarray,
    n_features: int,
    filter_type: str = "low_pass",
    K: int = 2,
    hidden_dim: int = 0,
    epochs: int = 300,
    lr: float = 1e-2,
    weight_decay: float = 1e-3,
    seed: int = 0,
    device: str = "cpu",
) -> Callable[[np.ndarray], ChebNetIIHead]:
    """Build an ``expert_factory`` for :func:`fit_experts` (fixed filter type)."""

    def factory(train_idx: np.ndarray) -> ChebNetIIHead:
        return ChebNetIIHead(
            W_norm=W_norm, n_features=n_features, train_idx=train_idx,
            K=K, filter_type=filter_type, hidden_dim=hidden_dim, epochs=epochs,
            lr=lr, weight_decay=weight_decay, seed=seed, device=device,
        )

    return factory


class NodeMoEGate(nn.Module):
    """One-hop low-pass GNN gate over the Node-MoE descriptor ``[X, |AX-X|, |A^2X-X|]``.

    ``g = softmax(MLP([G, A_hat G]))`` where ``G`` is the descriptor. The
    propagation is what makes neighbouring nodes pick similar experts, as in the
    paper's GIN gate.
    """

    def __init__(
        self, n_gate_features: int, n_experts: int, hidden_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * n_gate_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_experts),
        )

    def forward(self, gate_feats: torch.Tensor, Wt: torch.Tensor) -> torch.Tensor:
        if Wt.layout == torch.sparse_coo:
            prop = torch.sparse.mm(Wt, gate_feats)
        else:
            prop = torch.mm(Wt, gate_feats)
        return F.softmax(self.net(torch.cat([gate_feats, prop], dim=-1)), dim=-1)


def _smoothing_loss(filters: list[ChebNetIIFilter]) -> torch.Tensor:
    """Node-MoE filter-smoothing loss ``sum_o sum_i (f_o(x_i) - f_o(x_{i-1}))^2``.

    The nodes come out of :func:`chebyshev_nodes` in decreasing order; the paper's
    penalty is over adjacent interpolation points, so sort them ascending first.
    """
    loss = torch.zeros((), device=filters[0].gamma.device)
    for flt in filters:
        # response() is evaluated at flt.nodes; sort by node so "adjacent" means
        # adjacent on the eigenvalue axis, which is what the paper penalises.
        order = torch.argsort(flt.nodes)
        f_sorted = flt.response()[order]
        loss = loss + torch.sum((f_sorted[1:] - f_sorted[:-1]) ** 2)
    return loss


class NodeMoEHead:
    """The faithful joint Node-MoE model (gate + differentiated ChebNetII experts).

    Trained end-to-end on the training split with the task loss plus the filter
    smoothing loss, matching Node-MoE's own training. Exposes the shared expert
    hooks so it can also be scored by :func:`mixture_accuracy`.

    ``routing="uniform"`` removes the gate entirely and combines the same
    differentiated experts by a fixed equal-weight mean (``1/K`` per expert).
    Every other aspect of training is identical -- same experts, same loss, same
    epochs -- so ``gate`` vs ``uniform`` isolates exactly what the learned gate
    contributes over an unrouted multi-expert ensemble with the same expert
    capacity. If they tie, the routing signal contributed nothing.
    """

    def __init__(
        self,
        W_norm: np.ndarray,
        n_features: int,
        train_idx: np.ndarray,
        filter_types: list[str],
        K: int = 2,
        routing: str = "gate",
        gate_hidden: int = 32,
        gate_dropout: float = 0.1,
        epochs: int = 300,
        lr: float = 1e-2,
        weight_decay: float = 1e-3,
        gamma_smooth: float = 1e-3,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        if routing not in ("gate", "uniform"):
            raise ValueError(f"routing must be 'gate' or 'uniform', got {routing!r}")
        self.W_norm = W_norm
        self.n_features = n_features
        train_idx = np.asarray(train_idx)
        self.train_idx = (
            np.flatnonzero(train_idx) if train_idx.dtype == bool
            else train_idx.astype(int)
        )
        self.filter_types = list(filter_types)
        self.K = K
        self.routing = routing
        self.gate_hidden = gate_hidden
        self.gate_dropout = gate_dropout
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.gamma_smooth = gamma_smooth
        self.seed = seed
        self.device = device
        self._Wt: torch.Tensor | None = None
        self._basis: list[torch.Tensor] | None = None
        self._gate_feats: torch.Tensor | None = None
        self._model: nn.Module | None = None
        self._fallback: DummyClassifier | None = None

    def _W(self) -> torch.Tensor:
        if self._Wt is None:
            self._Wt = _as_torch_W(self.W_norm).to(self.device)
        return self._Wt

    def _features(self, X: np.ndarray) -> tuple[list[torch.Tensor], torch.Tensor]:
        if self._basis is None:
            xt = torch.as_tensor(np.asarray(X, dtype=np.float32), device=self.device)
            self._basis = chebnetii_basis(self._W(), xt, self.K)
            self._gate_feats = gate_features(self._W(), xt)
        return self._basis, self._gate_feats

    def _build_model(self) -> nn.Module:
        experts = nn.ModuleList(
            [
                ChebNetIIModule(self.n_features, self.K, ft)
                for ft in self.filter_types
            ]
        )
        holder = nn.Module()
        holder.experts = experts
        if self.routing == "gate":
            holder.gate = NodeMoEGate(
                3 * self.n_features, len(self.filter_types),
                hidden_dim=self.gate_hidden, dropout=self.gate_dropout,
            )
        return holder

    def _mix(self, expert_logits: torch.Tensor, gfeats, Wt):
        """Combine expert logits by the learned gate or a fixed 1/K mean."""
        if self.routing == "gate":
            g = self._model.gate(gfeats, Wt)
            return (g * expert_logits).sum(dim=1), g
        return expert_logits.mean(dim=1), None

    def fit_graph(self, X: np.ndarray, y: np.ndarray) -> "NodeMoEHead":
        classes = np.unique(y[self.train_idx])
        if classes.size < 2:
            self._fallback = DummyClassifier(strategy="most_frequent")
            self._fallback.fit(X[self.train_idx], y[self.train_idx])
            return self

        torch.manual_seed(self.seed)
        self._model = self._build_model().to(self.device)
        basis, gfeats = self._features(X)
        Wt = self._W()
        yt = torch.as_tensor(np.asarray(y, dtype=np.float32), device=self.device)
        idx = torch.as_tensor(self.train_idx, dtype=torch.long, device=self.device)

        pos = float(yt[idx].sum().item())
        neg = float(idx.numel() - pos)
        pos_weight = torch.tensor(
            [neg / pos if pos > 0 else 1.0], device=self.device, dtype=torch.float32
        )
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.Adam(
            self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self._model.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            expert_logits = torch.stack(
                [e(basis) for e in self._model.experts], dim=1
            )
            mix, _ = self._mix(expert_logits, gfeats, Wt)
            loss = loss_fn(mix[idx], yt[idx])
            if self.gamma_smooth > 0:
                loss = loss + self.gamma_smooth * _smoothing_loss(
                    [e.filter for e in self._model.experts]
                )
            loss.backward()
            opt.step()
        self._model.eval()
        return self

    def _forward(self, X: np.ndarray) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        assert self._model is not None, "fit_graph must be called before prediction"
        basis, gfeats = self._features(X)
        Wt = self._W()
        expert_logits = torch.stack([e(basis) for e in self._model.experts], dim=1)
        mix, g = self._mix(expert_logits, gfeats, Wt)
        return mix, g, expert_logits

    def _mix_logits(self, X: np.ndarray, idx: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            mix, _, _ = self._forward(X)
            sel = torch.as_tensor(np.asarray(idx, dtype=np.int64), device=self.device)
            return mix[sel].cpu().numpy().astype(float)

    def predict_nodes(self, X: np.ndarray, idx: np.ndarray) -> np.ndarray:
        if self._fallback is not None:
            return self._fallback.predict(np.asarray(X)[idx])
        return (self._mix_logits(X, idx) > 0).astype(np.int64)

    def predict_proba_nodes(self, X: np.ndarray, idx: np.ndarray) -> np.ndarray:
        if self._fallback is not None:
            return np.asarray(self._fallback.predict_proba(np.asarray(X)[idx]), dtype=float)
        return 1.0 / (1.0 + np.exp(-self._mix_logits(X, idx)))[:, None]

    def gate_weights(self, X: np.ndarray) -> np.ndarray:
        if self.routing != "gate":
            raise ValueError("gate_weights is only defined for routing='gate'")
        with torch.no_grad():
            _, g, _ = self._forward(X)
            return g.cpu().numpy()

    def expert_logits_nodes(self, X: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """Per-expert raw logits for ``idx``: shape ``(len(idx), n_experts)``."""
        with torch.no_grad():
            _, _, expert_logits = self._forward(X)
            sel = torch.as_tensor(np.asarray(idx, dtype=np.int64), device=self.device)
            return expert_logits[sel].cpu().numpy()


def train_gate(
    gate_feats_t: torch.Tensor,
    Wt: torch.Tensor,
    targets: np.ndarray,
    router_idx: np.ndarray,
    n_experts: int,
    hidden_dim: int = 32,
    dropout: float = 0.1,
    epochs: int = 200,
    lr: float = 1e-2,
    seed: int = 0,
    device: str = "cpu",
) -> NodeMoEGate:
    """Fit the Node-MoE gate on ``router_idx`` only (cross-fit, no test leakage)."""
    torch.manual_seed(seed)
    model = NodeMoEGate(
        gate_feats_t.shape[1], n_experts, hidden_dim=hidden_dim, dropout=dropout
    ).to(device)
    yt = torch.as_tensor(np.asarray(targets, dtype=np.float32), device=device)
    rt = torch.as_tensor(np.asarray(router_idx, dtype=np.int64), device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        g = model(gate_feats_t, Wt)[rt]
        loss = -(yt * torch.log(g + 1e-8)).sum(dim=1).mean()
        loss.backward()
        opt.step()
    model.eval()
    return model


def gate_assignment(model: NodeMoEGate, gate_feats_t: torch.Tensor, Wt: torch.Tensor) -> np.ndarray:
    """Hard one-hot routing from a trained gate's argmax."""
    model.eval()
    with torch.no_grad():
        g = model(gate_feats_t, Wt).cpu().numpy()
    out = np.zeros_like(g)
    out[np.arange(g.shape[0]), g.argmax(axis=1)] = 1.0
    return out


def build_differentiated_pool(
    W_norm: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    fit_mask: np.ndarray,
    filter_types: list[str],
    K: int = 2,
    epochs: int = 300,
    lr: float = 1e-2,
    weight_decay: float = 1e-3,
    seed: int = 0,
    device: str = "cpu",
) -> list[ChebNetIIHead]:
    """Routing-blind pool: one ChebNetII expert per filter type, each on *all*
    fit nodes. Routing plays no role in fitting -- the diversity comes from the
    differentiated filter initialisation, exactly Node-MoE's design.
    """
    fit_idx = np.flatnonzero(np.asarray(fit_mask))
    pool = []
    for ft in filter_types:
        head = ChebNetIIHead(
            W_norm=W_norm, n_features=X.shape[1], train_idx=fit_idx,
            K=K, filter_type=ft, epochs=epochs, lr=lr,
            weight_decay=weight_decay, seed=seed, device=device,
        )
        head.fit_graph(X, y)
        pool.append(head)
    return pool


def evaluate_node_moe_protocol_b(
    X: np.ndarray,
    y: np.ndarray,
    W_norm: np.ndarray,
    fit_mask: np.ndarray,
    router_mask: np.ndarray,
    test_mask: np.ndarray,
    filter_types: list[str],
    oracle: np.ndarray,
    K: int = 2,
    epochs: int = 300,
    gate_epochs: int = 200,
    lr: float = 1e-2,
    weight_decay: float = 1e-3,
    gate_lr: float = 1e-2,
    seed: int = 0,
    device: str = "cpu",
) -> dict:
    """Score Node-MoE routing under the honest fixed-expert (Protocol B) protocol.

    The differentiated ChebNetII pool is fitted routing-blind on ``fit_mask`` and
    frozen. The gate is supervised only on ``router_mask`` (disjoint from
    ``fit_mask``) against "which frozen expert is correct here", so it cannot
    memorise the training nodes. The no-routing reference is a single global
    ChebNetII fitted on ``fit_mask``. Returns per-condition accuracy plus paired
    gap summaries; the caller aggregates across splits.
    """
    from spectral_distillation.src.mixture import mixture_accuracy
    from spectral_distillation.src.router_protocol import random_assignment

    n_experts = len(filter_types)
    pool = build_differentiated_pool(
        W_norm, X, y, fit_mask, filter_types, K=K, epochs=epochs, lr=lr,
        weight_decay=weight_decay, seed=seed, device=device,
    )
    global_head = ChebNetIIHead(
        W_norm=W_norm, n_features=X.shape[1], train_idx=np.flatnonzero(fit_mask),
        K=K, filter_type="low_pass", epochs=epochs, lr=lr,
        weight_decay=weight_decay, seed=seed, device=device,
    )
    global_head.fit_graph(X, y)

    targets = build_router_targets(y, pool, X, np.flatnonzero(router_mask))
    Wt = _as_torch_W(W_norm).to(device)
    xt = torch.as_tensor(np.asarray(X, dtype=np.float32), device=device)
    gf = gate_features(Wt, xt)
    gate = train_gate(
        gf, Wt, targets, np.flatnonzero(router_mask), n_experts,
        epochs=gate_epochs, lr=gate_lr, seed=seed, device=device,
    )
    learned = gate_assignment(gate, gf, Wt)

    conditions = {
        "single_global": np.ones((X.shape[0], 1)),
        "node_moe_gate": learned,
        "random": random_assignment(oracle, rng=seed),
        "oracle": oracle,
    }
    acc = {}
    for name, assignment in conditions.items():
        heads = [global_head] if name == "single_global" else pool
        acc[name] = mixture_accuracy(X, y, assignment, heads, test_mask)[0]
    acc["ensemble_uniform"] = uniform_ensemble_accuracy(X, y, pool, test_mask)

    return {
        "accuracy": acc,
        "gate_minus_no_routing": acc["node_moe_gate"] - acc["single_global"],
        "gate_minus_random": acc["node_moe_gate"] - acc["random"],
        "gate_minus_ensemble": acc["node_moe_gate"] - acc["ensemble_uniform"],
        "ensemble_minus_no_routing": acc["ensemble_uniform"] - acc["single_global"],
        "oracle_minus_no_routing": acc["oracle"] - acc["single_global"],
        "n_fit_nodes": int(np.asarray(fit_mask).sum()),
        "n_router_nodes": int(np.asarray(router_mask).sum()),
    }

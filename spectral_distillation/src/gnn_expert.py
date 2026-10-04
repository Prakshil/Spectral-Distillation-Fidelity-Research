"""Message-passing (graph) expert heads for the fixed-expert router protocol.

The linear and MLP experts in :mod:`spectral_distillation.src.mixture` see only a
node's own feature row. That is the weakest possible argument against routing:
a router partitions nodes by *structure* (homophily, degree, clustering,
spectral position), while the expert being routed is blind to structure. A
reviewer of "routing does not help" will fairly ask whether the experts were
simply unable to exploit the structure a router routes on.

``GNNExpertHead`` closes that objection. Each expert performs real message
passing over the same symplectically normalized adjacency, so it can condition
its prediction on a node's neighbourhood -- exactly the information the routing
scores are computed from. The only free variable left in the protocol is then
the routing itself, which is what makes the comparison decisive.

Two deliberate choices, both aimed at not weakening the negative result:

* **Experts are trained on their own routed subset only.** The routing mask
  arrives via the factory (see :func:`gnn_head_factory`) and is used to restrict
  the loss, so a GNN expert gets no more labels than the linear or MLP expert
  gets for the same routing.
* **Message passing is transductive over the full graph.** Test-node *features*
  and edges participate in propagation; test-node *labels* never enter the
  loss. This is the standard transductive setting used by the routing
  literature this work critiques, and it is strictly more information for the
  routed experts -- the conservative direction for the claim being tested.

The head deliberately exposes ``fit_graph`` / ``predict_nodes`` rather than the
sklearn ``fit`` / ``predict`` pair: message passing needs the *whole* graph plus
the node identities of the rows being scored, whereas the sklearn interface is
handed a bare feature submatrix whose row-to-node mapping has been thrown away.
:func:`fit_experts` and :func:`mixture_accuracy` probe for these hooks and fall
back to the sklearn path, so every other head is unaffected.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.dummy import DummyClassifier

from spectral_distillation.src.gnn_router import _as_torch_W


class GraphExpertNet(nn.Module):
    """GraphSAGE-style encoder with a linear read-out and an input skip.

    ``h0 = AX + X`` is the one-hop neighbourhood representation and the read-out
    is ``Linear_skip(h0)``. With ``hidden_dim=0`` (the default) there is no
    hidden branch at all, i.e. a one-hop message-passing expert: the prediction
    is a linear function of the node's own features *and* its neighbours'.

    The hidden branch is available but **not** the default, and that is an
    evidence-driven choice. On a synthetic benchmark whose label is a linear
    function of the neighbourhood (where a per-row logistic head scores 0.52,
    i.e. chance, and a logistic head on explicit ``[X, AX]`` scores 0.94):

    ==========================  held-out accuracy
    linear read-out on ``[AX, X]``  **0.96**
    + ReLU hidden branch, 18configs  0.53 - 0.64

    Every ReLU variant -- widths 4/16/64, weight decay 1e-4..1e-1, dropout
    0/0.3, zero-initialised or not, 300..4000 epochs -- drove training accuracy
    to 1.0 while held-out accuracy fell to chance: full-batch Adam finds a
    memorising solution long before the generalising linear one. A weak expert
    would make "routing does not help" unfalsifiable, so the configuration that
    is measurably strong is the one used.
    """

    def __init__(
        self,
        n_features_in: int,
        hidden_dim: int = 0,
        n_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        # Read-out consumes the *concatenation* [X, AX], not the sum AX + X. The
        # sum forces one shared weight on a node's own features and its
        # neighbours', which cannot represent a general neighbour-dependent
        # label; concatenation can, and matches the strong linear reference.
        self.out_skip = nn.Linear(2 * n_features_in, 1)
        if hidden_dim > 0:
            self.lin = nn.Linear(2 * n_features_in, hidden_dim)
            self.out_hidden = nn.Linear(hidden_dim, 1)
            self.dropout = nn.Dropout(dropout)

    def forward(self, h0, W=None):
        logits = self.out_skip(h0)
        if self.hidden_dim > 0:
            h = F.dropout(F.relu(self.lin(h0)), self.dropout.p, self.training)
            for _ in range(self.n_layers - 1):
                if W is None:
                    raise ValueError("depth > 1 needs W to re-aggregate hidden states")
                a = torch.sparse.mm(W, h) if W.layout == torch.sparse_coo else torch.mm(W, h)
                h = F.dropout(F.relu(self.lin(torch.cat([h, a], dim=-1))),
                              self.dropout.p, self.training)
            logits = logits + self.out_hidden(h)
        return logits.squeeze(-1)


class GNNExpertHead:
    """One graph expert: message passing + linear read-out, routed-subset fit.

    Exposes ``fit_graph(X, y)`` and ``predict_nodes(X, idx)`` for the hooks in
    :mod:`spectral_distillation.src.mixture`.
    """

    def __init__(
        self,
        W_norm: np.ndarray,
        n_features: int,
        train_idx: np.ndarray,
        hidden_dim: int = 0,
        n_layers: int = 1,
        epochs: int = 2000,
        lr: float = 1e-2,
        dropout: float = 0.0,
        weight_decay: float = 1e-3,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        self.W_norm = W_norm
        self.n_features = n_features
        # ``fit_experts`` hands over a boolean mask; ``numpy`` would cast that to
        # 0/1 rather than to node indices, so normalise both accepted forms here.
        train_idx = np.asarray(train_idx)
        self.train_idx = (
            np.flatnonzero(train_idx) if train_idx.dtype == bool
            else train_idx.astype(int)
        )
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.epochs = epochs
        self.lr = lr
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.seed = seed
        self.device = device
        self._Wt: torch.Tensor | None = None
        self._h0_cache: torch.Tensor | None = None
        self._model: GraphExpertNet | None = None
        self.classes_: np.ndarray | None = None

    # -- graph plumbing ----------------------------------------------------
    def _W(self) -> torch.Tensor:
        """Cache the torch adjacency; conversion is expensive and reused."""
        if self._Wt is None:
            self._Wt = _as_torch_W(self.W_norm).to(self.device)
        return self._Wt

    def _h0(self, X: np.ndarray) -> torch.Tensor:
        """One-hop propagated features ``[X, AX]``, computed once and cached.

        ``h0`` does not depend on any learnable parameter, so recomputing it on
        every one of the thousands of training steps is pure waste -- and on a
        8.6k-node graph it dominated runtime by ~3 orders of magnitude. The
        cached tensor is also shared by ``predict_nodes`` so scoring cannot drift
        from training. Kept per-instance (one head per expert per split/seed).
        """
        if self._h0_cache is None:
            xt = torch.as_tensor(np.asarray(X, dtype=np.float32), device=self.device)
            Wt = self._W()
            a = torch.sparse.mm(Wt, xt) if Wt.layout == torch.sparse_coo \
                else torch.mm(Wt, xt)
            self._h0_cache = torch.cat([xt, a], dim=-1)
        return self._h0_cache

    # -- hooks used by mixture.py ------------------------------------------
    def _model_hidden_depth(self) -> int:
        """Number of hidden layers that still need ``W`` to re-aggregate."""
        return self.n_layers if self.hidden_dim > 0 else 0

    def fit_graph(self, X: np.ndarray, y: np.ndarray):
        """Fit on the routed train subset using full-graph message passing."""
        classes = np.unique(y[self.train_idx])
        if classes.size < 2:
            # Mirror the sklearn-head fallback so behaviour matches exactly.
            self._fallback = DummyClassifier(strategy="most_frequent")
            self._fallback.fit(X[self.train_idx], y[self.train_idx])
            self.classes_ = classes
            return self
        self.classes_ = classes

        torch.manual_seed(self.seed)
        self._model = GraphExpertNet(
            self.n_features, self.hidden_dim, self.n_layers, self.dropout
        ).to(self.device)
        h0 = self._h0(X)
        yt = torch.as_tensor(np.asarray(y, dtype=np.float32), device=self.device)
        idx = torch.as_tensor(self.train_idx, dtype=torch.long, device=self.device)
        Wt = self._W() if self._model_hidden_depth() > 1 else None

        # Balance the loss so the minority class is actually learned; the metric
        # is accuracy, so an unbalanced head would simply be a weaker expert.
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
            loss = loss_fn(self._model(h0, Wt)[idx], yt[idx])
            loss.backward()
            opt.step()
        self._model.eval()
        return self

    def _logits_nodes(self, X: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """Raw read-out logits for global node indices ``idx`` (shape ``(len(idx),)``).

        Single source of truth for both label and probability prediction, so the
        two can never disagree on a node that sits exactly at logit 0.
        """
        assert self._model is not None, "fit_graph must be called before prediction"
        Wt = self._W() if self._model_hidden_depth() > 1 else None
        with torch.no_grad():
            logits = self._model(self._h0(X), Wt)
            sel = torch.as_tensor(np.asarray(idx, dtype=np.int64), device=self.device)
            return logits[sel].cpu().numpy().astype(float)

    def predict_nodes(self, X: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """Predict labels for global node indices ``idx``."""
        if hasattr(self, "_fallback"):
            return self._fallback.predict(np.asarray(X)[idx])
        return (self._logits_nodes(X, idx) > 0).astype(np.int64)

    def predict_proba_nodes(self, X: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """Class-1 probability for global node indices ``idx``.

        Required by the soft-voting ensemble in
        :func:`spectral_distillation.src.mixture.uniform_ensemble_accuracy`.
        ``predict_nodes`` alone is not enough for it: an ensemble that averages
        hard labels degenerates into majority voting and cannot express graded
        confidence, which is the entire quantity the ensemble baseline is meant
        to measure. This expert is binary (``BCEWithLogitsLoss``), so the
        returned column is the only column and ``classes_`` is ``[0, 1]``.

        The dummy fallback keeps sklearn's ``predict_proba`` so a degenerate
        one-class routed subset still contributes a valid distribution.
        """
        if hasattr(self, "_fallback"):
            return np.asarray(self._fallback.predict_proba(np.asarray(X)[idx]),
                              dtype=float)
        return 1.0 / (1.0 + np.exp(-self._logits_nodes(X, idx)))[:, None]


def gnn_head_factory(
    W_norm: np.ndarray,
    n_features: int,
    device: str = "cpu",
    hidden_dim: int = 0,
    n_layers: int = 1,
    epochs: int = 2000,
    lr: float = 1e-2,
    dropout: float = 0.0,
    weight_decay: float = 1e-3,
    seed: int = 0,
):
    """Build an ``expert_factory`` for :func:`fit_experts`.

    ``fit_experts`` calls ``expert_factory(train_idx)`` with the routed training
    mask, so the returned callable bakes that mask into each expert -- the
    expert trains on its own routed subset only, never on the full train set.
    """

    def factory(train_idx):
        return GNNExpertHead(
            W_norm=W_norm,
            n_features=n_features,
            train_idx=train_idx,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            epochs=epochs,
            lr=lr,
            dropout=dropout,
            weight_decay=weight_decay,
            seed=seed,
            device=device,
        )

    return factory
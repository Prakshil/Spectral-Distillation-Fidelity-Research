"""Expert mixture: identical per-expert heads + routing to combine them.

Each expert is a logistic-regression head trained only on the nodes routed to
it (guide 6.4: identical experts, training, and budget across conditions; only
the assignment varies). Prediction applies hard routing: a test node is scored
by the single expert it is assigned to. This is what makes routing matters --
a node's assigned expert must have been trained on nodes from the same regime
basis to predict it well.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def expert_head(train_idx: np.ndarray | None = None) -> LogisticRegression:
    """Build a single expert head (identical for every expert/condition).

    ``train_idx`` is accepted (and ignored) so every expert factory shares one
    signature; :func:`fit_experts` always passes the routed training mask.
    """
    return LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs", n_jobs=1)


def mlp_head(train_idx: np.ndarray | None = None) -> MLPClassifier:
    """A small non-linear expert head.

    The linear head may simply be too weak to specialise, so a router can never
    win. This gives each expert genuine non-linear capacity. Deterministic
    (fixed ``random_state``) so protocol cells reproduce.
    """
    return MLPClassifier(
        hidden_layer_sizes=(64,),
        activation="relu",
        alpha=1e-3,
        max_iter=400,
        random_state=0,
        early_stopping=False,
    )


EXPERT_FACTORIES = {"logistic": expert_head, "mlp": mlp_head}


def fit_experts(
    X: np.ndarray,
    y: np.ndarray,
    assignment: np.ndarray,
    train_mask: np.ndarray,
    expert_factory: Callable[[np.ndarray], object] = expert_head,
) -> list[object | None]:
    """Fit K expert heads, each on its own routed train subset.

    A shortage of routes yields a majority-class dummy head (never ``None``
    unless the masked set is empty) so the mixture stays well defined even
    under heavy dilution. ``expert_factory`` builds each head; it defaults to
    the linear ``expert_head`` so existing callers reproduce exactly.

    Each factory is called as ``expert_factory(train_mask_k)`` so a head that
    needs the full graph (see :mod:`spectral_distillation.src.gnn_expert`) can
    learn which nodes it owns. Heads exposing ``fit_graph`` receive the whole
    feature matrix -- message passing needs every node, while the loss is still
    restricted to ``train_mask_k``. All other heads get the usual sklearn
    ``fit(X[masked], y[masked])`` call.
    """
    assignment = np.asarray(assignment, dtype=float)
    K = assignment.shape[1]
    experts: list[object | None] = []
    for k in range(K):
        masked = train_mask & (assignment[:, k] >= 0.01)
        if masked.sum() < 10:
            experts.append(None)
            continue
        unique = np.unique(y[masked])
        if unique.size < 2:
            dummy = DummyClassifier(strategy="most_frequent")
            dummy.fit(X[masked], y[masked])
            experts.append(dummy)
            continue
        head = expert_factory(masked)
        if hasattr(head, "fit_graph"):
            head.fit_graph(X, y)
        else:
            head.fit(X[masked], y[masked])
        experts.append(head)
    return experts


def mixture_accuracy(
    X: np.ndarray,
    y: np.ndarray,
    assignment: np.ndarray,
    experts: list[object | None],
    test_mask: np.ndarray,
) -> tuple[float, int]:
    """Accuracy over test nodes using hard routing (argmax assignment).

    Returns (accuracy, n_evaluated). Nodes routed to experts that had no
    training data are skipped and reported neither correct nor wrong.

    Heads exposing ``predict_nodes`` are handed the global node indices as well
    as the full feature matrix, because a message-passing expert cannot recover
    node identity from a feature submatrix alone. Every other head gets the
    ordinary ``predict(X[block])`` call.
    """
    assignment = np.asarray(assignment, dtype=float)
    hard = np.argmax(assignment, axis=1)
    idx = np.flatnonzero(test_mask)
    if idx.size == 0:
        return 0.0, 0
    correct = 0
    evaluated = 0
    for k, head in enumerate(experts):
        if head is None:
            continue
        block = idx[hard[idx] == k]
        if block.size == 0:
            continue
        if hasattr(head, "predict_nodes"):
            pred = head.predict_nodes(X, block)
        else:
            pred = head.predict(X[block])
        correct += int(np.count_nonzero(pred == y[block]))
        evaluated += int(block.size)
    if evaluated == 0:
        return 0.0, 0
    return float(correct / evaluated), evaluated


def train_test_split(
    y: np.ndarray,
    split: int,
    seed: int,
    train_frac: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Class-stratified train/test split derived deterministically from
    (split, seed) so protocol cells are reproducible and disjoint."""
    rng = np.random.default_rng(1000 * (split + 1) + seed)
    n = y.size
    train = np.zeros(n, dtype=bool)
    classes = np.unique(y)
    for c in classes:
        members = np.flatnonzero(y == c)
        rng.shuffle(members)
        n_tr = max(1, int(np.rint(train_frac * members.size)))
        train[members[:n_tr]] = True
    return train, ~train
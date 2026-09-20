"""Expert mixture: identical per-expert heads + routing to combine them.

Each expert is a logistic-regression head trained only on the nodes routed to
it (guide 6.4: identical experts, training, and budget across conditions; only
the assignment varies). Prediction applies hard routing: a test node is scored
by the single expert it is assigned to. This is what makes routing matters --
a node's assigned expert must have been trained on nodes from the same regime
basis to predict it well.
"""

from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression


def expert_head() -> LogisticRegression:
    """Build a single expert head (identical for every expert/condition)."""
    return LogisticRegression(C=1.0, max_iter=2000)


def fit_experts(
    X: np.ndarray,
    y: np.ndarray,
    assignment: np.ndarray,
    train_mask: np.ndarray,
) -> list[LogisticRegression | DummyClassifier | None]:
    """Fit K expert heads, each on its own routed train subset.

    A shortage of routes yields a majority-class dummy head (never ``None``
    unless the masked set is empty) so the mixture stays well defined even
    under heavy dilution.
    """
    assignment = np.asarray(assignment, dtype=float)
    K = assignment.shape[1]
    experts: list[LogisticRegression | DummyClassifier | None] = []
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
        head = expert_head()
        head.fit(X[masked], y[masked])
        experts.append(head)
    return experts


def mixture_accuracy(
    X: np.ndarray,
    y: np.ndarray,
    assignment: np.ndarray,
    experts: list[LogisticRegression | None],
    test_mask: np.ndarray,
) -> tuple[float, int]:
    """Accuracy over test nodes using hard routing (argmax assignment).

    Returns (accuracy, n_evaluated). Nodes routed to experts that had no
    training data are skipped and reported neither correct nor wrong.
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
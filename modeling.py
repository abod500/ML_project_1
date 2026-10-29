"""Training helpers, predictions, data splits and evaluation metrics."""

from __future__ import annotations

from typing import Callable

import numpy as np

from implementations import (
    _sigmoid,
    logistic_regression,
    reg_logistic_regression,
    ridge_regression,
)

def predict_logistic(x, weights, threshold: float = 0.5):
    """Return logistic probabilities and {-1, 1} predictions."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    probabilities = _sigmoid(np.asarray(x).dot(weights))
    return probabilities, np.where(probabilities >= threshold, 1, -1)


def predict_ridge(x, weights, threshold: float = 0.0):
    """Return linear scores and {-1, 1} predictions."""
    scores = np.asarray(x).dot(weights)
    return scores, np.where(scores >= threshold, 1, -1)


# --------------------------------------------------------------------------- #
# Metrics (labels in {-1, 1} or {0, 1}; positive class = 1)
# --------------------------------------------------------------------------- #


def roc_auc(y_true, scores) -> float:
    """Area under the ROC curve via the rank (Mann-Whitney) formula."""
    positive = np.asarray(y_true) == 1
    n_pos, n_neg = positive.sum(), (~positive).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = np.empty(len(scores))
    ranks[np.argsort(scores, kind="mergesort")] = np.arange(1, len(scores) + 1)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def classification_metrics(y_true, y_pred, scores=None) -> dict:
    """Accuracy, balanced accuracy, precision, recall, F1 (and AUC if scores given)."""
    actual, predicted = np.asarray(y_true) == 1, np.asarray(y_pred) == 1
    tp = int(np.sum(actual & predicted))
    tn = int(np.sum(~actual & ~predicted))
    fp = int(np.sum(~actual & predicted))
    fn = int(np.sum(actual & ~predicted))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    metrics = {
        "accuracy": (tp + tn) / len(actual),
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "confusion_matrix": np.array([[tn, fp], [fn, tp]]),  # [[TN, FP], [FN, TP]]
    }
    if scores is not None:
        metrics["roc_auc"] = roc_auc(y_true, scores)
    return metrics


def best_f1_threshold(y_true, scores, n_steps: int = 200) -> tuple[float, float]:
    """Return (threshold, F1) maximising F1 on a validation set."""
    candidates = np.quantile(scores, np.linspace(0.01, 0.99, n_steps))
    best = (0.5, -1.0)
    for t in candidates:
        f1 = classification_metrics(y_true, (scores >= t).astype(int))["f1"]
        if f1 > best[1]:
            best = (float(t), f1)
    return best


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #


def stratified_split(y, validation_fraction: float = 0.2, seed: int = 42):
    """Return (train_idx, validation_idx) with the same positive rate in both."""
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(np.asarray(y) == 1)
    negative = np.flatnonzero(np.asarray(y) != 1)
    rng.shuffle(positive)
    rng.shuffle(negative)
    n_pos_val = int(round(validation_fraction * len(positive)))
    n_neg_val = int(round(validation_fraction * len(negative)))
    validation = np.concatenate((positive[:n_pos_val], negative[:n_neg_val]))
    train = np.concatenate((positive[n_pos_val:], negative[n_neg_val:]))
    rng.shuffle(validation)
    rng.shuffle(train)
    return train, validation


def undersample(indices, y, seed: int = 42):
    """Keep all positives in ``indices`` and as many randomly chosen negatives."""
    rng = np.random.default_rng(seed)
    indices = np.asarray(indices)
    positive = indices[np.asarray(y)[indices] == 1]
    negative = indices[np.asarray(y)[indices] != 1]
    chosen = rng.choice(negative, size=min(len(positive), len(negative)), replace=False)
    balanced = np.concatenate((positive, chosen))
    rng.shuffle(balanced)
    return balanced


def k_fold_indices(n_samples: int, k: int = 5, seed: int = 42) -> list[np.ndarray]:
    """Split ``range(n_samples)`` into ``k`` shuffled folds."""
    rng = np.random.default_rng(seed)
    return np.array_split(rng.permutation(n_samples), k)


def stratified_k_fold(y, k: int = 5, seed: int = 42) -> list[np.ndarray]:
    """Split the rows into ``k`` folds with the same positive rate in each."""
    rng = np.random.default_rng(seed)
    positive = rng.permutation(np.flatnonzero(np.asarray(y) == 1))
    negative = rng.permutation(np.flatnonzero(np.asarray(y) != 1))
    return [
        np.sort(np.concatenate((p, n)))
        for p, n in zip(np.array_split(positive, k), np.array_split(negative, k))
    ]


# --------------------------------------------------------------------------- #
# Our modification: class-weighted (regularized) logistic regression
# --------------------------------------------------------------------------- #


def weighted_logistic_loss(y, tx, w, sample_weights):
    """Weighted average binary cross-entropy (no penalty term)."""
    logits = tx.dot(w)
    losses = np.logaddexp(0.0, logits) - y * logits
    return float(np.sum(sample_weights * losses) / np.sum(sample_weights))


def weighted_reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma,
                                     positive_weight=None):
    """L2-regularized logistic regression where positive samples get more weight.

    Instead of throwing away negatives (undersampling), every sample is kept and
    each positive counts ``positive_weight`` times (default n_negative/n_positive,
    so both classes have the same total weight). The gradient is the weighted
    version of the course gradient:
        grad = sum_i a_i (sigmoid(x_i w) - y_i) x_i / sum_i a_i + 2 lambda w
    Labels must be in {0, 1}. Returns (w, unregularized weighted loss).
    """
    y = np.asarray(y, dtype=float)
    if positive_weight is None:
        positive_weight = np.sum(y == 0) / max(np.sum(y == 1), 1)
    sample_weights = np.where(y == 1, positive_weight, 1.0)
    normalised = sample_weights / sample_weights.sum()
    w = np.array(initial_w, dtype=float, copy=True)
    for _ in range(max_iters):
        residual = _sigmoid(tx.dot(w)) - y
        gradient = tx.T.dot(normalised * residual) + 2.0 * lambda_ * w
        w -= gamma * gradient
    return w, weighted_logistic_loss(y, tx, w, sample_weights)


# --------------------------------------------------------------------------- #
# Training wrappers around implementations.py (labels must be 0/1)
# --------------------------------------------------------------------------- #


def fit_ridge(y, x, lambda_: float = 1e-3, trainer: Callable = ridge_regression):
    """Train ridge regression and return (weights, training loss)."""
    if lambda_ < 0:
        raise ValueError("lambda_ must be non-negative")
    return trainer(y, x, lambda_)


def fit_logistic(y, x, max_iters=100, gamma=1e-2, initial_w=None,
                 trainer: Callable = logistic_regression):
    """Train logistic regression with gradient descent."""
    initial = np.zeros(x.shape[1]) if initial_w is None else initial_w
    return trainer(y, x, initial, max_iters, gamma)


def fit_regularized_logistic(y, x, lambda_=1e-3, max_iters=100, gamma=1e-2,
                             initial_w=None, trainer: Callable = reg_logistic_regression):
    """Train L2-regularized logistic regression with gradient descent."""
    if lambda_ < 0:
        raise ValueError("lambda_ must be non-negative")
    initial = np.zeros(x.shape[1]) if initial_w is None else initial_w
    return trainer(y, x, lambda_, initial, max_iters, gamma)

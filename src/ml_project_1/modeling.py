import numpy as np
from typing import Callable

from basic_functions import _as_matrix, _sigmoid

def predict_logistic(
    x: np.ndarray, weights: np.ndarray, threshold: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Return logistic probabilities and {-1, 1} predictions."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    probabilities = _sigmoid(_as_matrix(x).dot(weights))
    return probabilities, np.where(probabilities >= threshold, 1, -1)


def predict_ridge(x: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ridge scores and predictions using zero as the decision boundary."""
    scores = _as_matrix(x).dot(weights)
    return scores, np.where(scores >= 0, 1, -1)


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float | np.ndarray]:
    """Calculate accuracy, balanced accuracy, and a {-1, 1} confusion matrix."""
    labels = np.array([-1, 1])
    confusion = np.zeros((2, 2), dtype=int)
    for row, actual in enumerate(labels):
        for column, predicted in enumerate(labels):
            confusion[row, column] = np.sum((y_true == actual) & (y_pred == predicted))
    recalls = np.diag(confusion) / np.maximum(confusion.sum(axis=1), 1)
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "balanced_accuracy": float(np.mean(recalls)),
        "confusion_matrix": confusion,
    }


def fit_ridge(
    y: np.ndarray,
    x: np.ndarray,
    lambda_: float = 1e-3,
    trainer: Callable | None = None,
) -> tuple[np.ndarray, float]:
    """Train the course ridge model and return weights and training loss."""
    if lambda_ < 0:
        raise ValueError("lambda_ must be non-negative")
    if trainer is None:
        from implementations import ridge_regression

        trainer = ridge_regression
    return trainer(y, x, lambda_)


def fit_logistic(
    y: np.ndarray,
    x: np.ndarray,
    max_iters: int = 100,
    gamma: float = 1e-2,
    initial_w: np.ndarray | None = None,
    trainer: Callable | None = None,
) -> tuple[np.ndarray, float]:
    """Train course logistic regression for labels encoded as 0 and 1."""
    initial = np.zeros(x.shape[1]) if initial_w is None else initial_w
    if trainer is None:
        from implementations import logistic_regression

        trainer = logistic_regression
    return trainer(y, x, initial, max_iters, gamma)


def fit_regularized_logistic(
    y: np.ndarray,
    x: np.ndarray,
    lambda_: float = 1e-3,
    max_iters: int = 100,
    gamma: float = 1e-2,
    initial_w: np.ndarray | None = None,
    trainer: Callable | None = None,
) -> tuple[np.ndarray, float]:
    """Train course L2-regularized logistic regression for labels 0 and 1."""
    if lambda_ < 0:
        raise ValueError("lambda_ must be non-negative")
    initial = np.zeros(x.shape[1]) if initial_w is None else initial_w
    if trainer is None:
        from implementations import reg_logistic_regression

        trainer = reg_logistic_regression
    return trainer(y, x, lambda_, initial, max_iters, gamma)
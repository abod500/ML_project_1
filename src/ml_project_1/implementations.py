import numpy as np


def _mse_loss(y, tx, w):
    """Return the mean squared error with the course's one-half factor."""
    errors = y - tx.dot(w)
    return 0.5 * np.mean(errors**2)


def _sigmoid(t):
    """Compute sigmoid values without overflow for large-magnitude inputs."""
    result = np.empty_like(t, dtype=float)
    positive = t >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-t[positive]))
    exp_t = np.exp(t[~positive])
    result[~positive] = exp_t / (1.0 + exp_t)
    return result


def _logistic_loss(y, tx, w):
    """Return the average binary cross-entropy loss."""
    logits = tx.dot(w)
    return np.mean(np.logaddexp(0.0, logits) - y * logits)


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Compute linear regression with batch gradient descent.

    Returns the final weight vector and the corresponding mean squared error.
    """
    w = np.array(initial_w, dtype=float, copy=True)
    n_samples = y.shape[0]
    for _ in range(max_iters):
        errors = y - tx.dot(w)
        gradient = -tx.T.dot(errors) / n_samples
        w -= gamma * gradient
    return w, _mse_loss(y, tx, w)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Compute linear regression with stochastic gradient descent.

    Stochastic updates use one sample at a time, cycling through the data.
    """
    w = np.array(initial_w, dtype=float, copy=True)
    n_samples = y.shape[0]
    for iteration in range(max_iters):
        index = iteration % n_samples
        error = y[index] - tx[index].dot(w)
        gradient = -tx[index] * error
        w -= gamma * gradient
    return w, _mse_loss(y, tx, w)


def least_squares(y, tx):
    """Compute least-squares regression using the normal equations."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    return w, _mse_loss(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """Compute ridge regression using the normal equations."""
    n_features = tx.shape[1]
    a = tx.T.dot(tx) + 2.0 * tx.shape[0] * lambda_ * np.eye(n_features)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    return w, _mse_loss(y, tx, w)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Compute logistic regression with batch gradient descent for y in {0, 1}."""
    w = np.array(initial_w, dtype=float, copy=True)
    n_samples = y.shape[0]
    for _ in range(max_iters):
        probabilities = _sigmoid(tx.dot(w))
        gradient = tx.T.dot(probabilities - y) / n_samples
        w -= gamma * gradient
    return w, _logistic_loss(y, tx, w)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Compute L2-regularized logistic regression for y in {0, 1}.

    The returned loss is the unregularized logistic loss, as required by the
    project specification.
    """
    w = np.array(initial_w, dtype=float, copy=True)
    n_samples = y.shape[0]
    for _ in range(max_iters):
        probabilities = _sigmoid(tx.dot(w))
        gradient = tx.T.dot(probabilities - y) / n_samples + 2.0 * lambda_ * w
        w -= gamma * gradient
    return w, _logistic_loss(y, tx, w)
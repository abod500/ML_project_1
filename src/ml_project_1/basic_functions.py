import numpy as np

def _as_matrix(x: np.ndarray) -> np.ndarray:
    matrix = np.asarray(x)
    if matrix.ndim != 2:
        raise ValueError("x must be a two-dimensional array")
    return matrix


def _select_columns(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    matrix = _as_matrix(x)
    if matrix.shape[1] != mask.size:
        raise ValueError("all arrays must have the same number of columns")
    return matrix[:, mask]


def _validate_column(column: int, n_columns: int) -> None:
    if not 0 <= column < n_columns:
        raise ValueError(f"code mapping refers to unavailable column {column}")


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -700, 700)
    return 1.0 / (1.0 + np.exp(-clipped))

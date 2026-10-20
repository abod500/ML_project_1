"""Reusable data-preparation and model functions for project 1.

The cleaning functions do not modify their inputs.  Fit-dependent operations
return parameters that must be reused for validation and test data.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np

from basic_functions import _as_matrix, _select_columns, _sigmoid, _validate_column

DEFAULT_MISSING_CODES: dict[int, tuple[float, ...]] = {
    **{
        column: (7, 9)
        for column in (
            14,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            38,
            39,
            40,
            41,
            45,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            64,
            70,
            71,
            72,
            73,
        )
    },
    36: (9,),
    37: (9,),
    42: (9,),
    **{column: (77, 99) for column in (11, 15, 16, 17, 44, 65, 68)},
    **{column: (777, 999) for column in (57, 58, 59, 60, 61, 62, 63, 66, 67, 69)},
    46: (7777, 9999),
    47: (7777, 9999),
}

DEFAULT_ZERO_CODES: dict[int, tuple[float, ...]] = {
    **{column: (88,) for column in (15, 16, 17, 43)},
    57: (888,),
    69: (888,),
    **{column: (555,) for column in (58, 59, 60, 61, 62, 63)},
}


def audit_matrix(x, name):
    finite = np.isfinite(x)
    safe = np.where(finite, x, np.nan)
    print(name, x.shape)
    print('  non-finite entries:', int(np.sum(~finite)))
    print('  constant columns:', int(np.sum(np.nanstd(safe, axis=0) == 0)))
    print('  min/max:', np.nanmin(safe), np.nanmax(safe))


def drop_constant_columns(x: np.ndarray, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """Remove columns with zero variance, returning the filtered arrays."""
    matrix = _as_matrix(x)
    variances = np.nanvar(matrix, axis=0)
    mask = variances > 0
    return tuple(_select_columns(array, mask) for array in (matrix, *arrays))


def drop_columns_with_nan_fraction(
    x: np.ndarray, *arrays: np.ndarray, max_fraction: float = 0.50
) -> tuple[np.ndarray, ...]:
    """Remove columns whose NaN fraction is greater than a limit."""
    if not 0 <= max_fraction <= 1:
        raise ValueError("max_fraction must be between 0 and 1")
    matrix = _as_matrix(x)
    mask = np.isnan(matrix).mean(axis=0) <= max_fraction
    return tuple(_select_columns(array, mask) for array in (matrix, *arrays))


def replace_survey_codes(
    x: np.ndarray,
    missing_codes: Mapping[int, Sequence[float]] = DEFAULT_MISSING_CODES,
    zero_codes: Mapping[int, Sequence[float]] = DEFAULT_ZERO_CODES,
) -> np.ndarray:
    """Convert column-specific unknown codes to NaN and zero codes to 0."""
    cleaned = _as_matrix(x).astype(float, copy=True)
    for column, codes in missing_codes.items():
        _validate_column(column, cleaned.shape[1])
        for code in codes:
            cleaned[cleaned[:, column] == code, column] = np.nan
    for column, codes in zero_codes.items():
        _validate_column(column, cleaned.shape[1])
        for code in codes:
            cleaned[cleaned[:, column] == code, column] = 0.0
    return cleaned


def fit_preprocessor(x: np.ndarray) -> dict[str, np.ndarray]:
    """Fit median imputation and standardization parameters on training data."""
    matrix = _as_matrix(x)
    median = np.nanmedian(matrix, axis=0)
    if np.isnan(median).any():
        raise ValueError("cannot fit preprocessing: a column contains only NaNs")
    imputed = np.where(np.isnan(matrix), median, matrix)
    mean = np.mean(imputed, axis=0)
    std = np.std(imputed, axis=0)
    std[std == 0] = 1.0
    return {"median": median, "mean": mean, "std": std}


def transform_features(x: np.ndarray, params: Mapping[str, np.ndarray]) -> np.ndarray:
    """Impute and standardize features using parameters fitted on training data."""
    matrix = _as_matrix(x)
    for key in ("median", "mean", "std"):
        if key not in params:
            raise KeyError(f"missing preprocessing parameter: {key}")
    if matrix.shape[1] != len(params["median"]):
        raise ValueError("feature count does not match preprocessing parameters")
    imputed = np.where(np.isnan(matrix), params["median"], matrix)
    return (imputed - params["mean"]) / params["std"]


def add_bias_column(x: np.ndarray) -> np.ndarray:
    """Prepend an intercept column of ones to a feature matrix."""
    matrix = _as_matrix(x)
    return np.c_[np.ones(matrix.shape[0]), matrix]

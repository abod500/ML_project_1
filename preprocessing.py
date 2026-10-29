"""Data cleaning and feature preparation for the BRFSS heart-disease data.

Columns are always identified by their *name* (read from the CSV header), so
the cleaning rules in ``brfss_codebook.py`` cannot drift out of sync with the
column order.

Typical use::

    names = load_feature_names(DATA_DIR)
    x_tr, feature_names, params = fit_transform_pipeline(x_train, names)
    x_te, _ = transform_pipeline(x_test, names, params)

The pipeline has two kinds of steps:

* Fixed steps (no learning): drop useless columns, convert survey codes such as
  7/9/77/99 to NaN and 88 to 0, fill questions that were skipped for a logical
  reason, recode a few variables to be ordinal.
* Fitted steps (learned on the training set only, then reused on validation /
  test data): drop mostly-missing and constant columns, add missing-value
  indicators, encode yes/no as 1/0, one-hot nominal variables, clip outliers,
  drop near-duplicate columns, median imputation and standardisation.
"""

from __future__ import annotations

import os
from typing import Mapping, Sequence

import numpy as np

import brfss_codebook as cb

# --------------------------------------------------------------------------- #
# Small utilities (previously in basic_functions.py)
# --------------------------------------------------------------------------- #


def _as_matrix(x: np.ndarray) -> np.ndarray:
    """Return ``x`` as a 2-D array or raise a clear error."""
    matrix = np.asarray(x)
    if matrix.ndim != 2:
        raise ValueError("x must be a two-dimensional array")
    return matrix


def _check_names(x: np.ndarray, names: Sequence[str]) -> list[str]:
    """Make sure there is exactly one name per column."""
    names = list(names)
    if _as_matrix(x).shape[1] != len(names):
        raise ValueError(
            f"x has {x.shape[1]} columns but {len(names)} names were given"
        )
    return names


def _keep(x: np.ndarray, names: list[str], keep: Sequence[str]):
    """Keep the columns listed in ``keep`` (in that order)."""
    index = {name: i for i, name in enumerate(names)}
    missing = [name for name in keep if name not in index]
    if missing:
        raise ValueError(f"columns not found: {missing[:5]}")
    columns = [index[name] for name in keep]
    return x[:, columns], list(keep)


def load_feature_names(data_dir: str, file_name: str = "x_train.csv") -> list[str]:
    """Read the feature names from the CSV header, without the leading Id column.

    ``helpers.load_csv_data`` skips the header, so this gives us the matching
    list of names for the columns of ``x_train`` / ``x_test``.
    """
    with open(os.path.join(data_dir, file_name), "r") as handle:
        header = handle.readline().strip().split(",")
    return [name.strip().strip('"') for name in header[1:]]


def audit_matrix(x: np.ndarray, name: str) -> None:
    """Print a quick summary of a feature matrix."""
    finite = np.isfinite(x)
    safe = np.where(finite, x, np.nan)
    print(name, x.shape)
    print("  non-finite entries:", int(np.sum(~finite)))
    print("  constant columns:", int(np.sum(np.nanstd(safe, axis=0) == 0)))
    print("  min/max:", np.nanmin(safe), np.nanmax(safe))


def describe_columns(x: np.ndarray, names: Sequence[str], top: int = 5) -> None:
    """Print the NaN fraction and most frequent values of each column (EDA)."""
    names = _check_names(x, names)
    for j, name in enumerate(names):
        column = x[:, j]
        nan_fraction = np.isnan(column).mean()
        values, counts = np.unique(column[~np.isnan(column)], return_counts=True)
        order = np.argsort(counts)[::-1][:top]
        common = ", ".join(f"{values[k]:g}:{counts[k]}" for k in order)
        print(f"{j:3d} {name:<10} nan={nan_fraction:5.1%}  {common}")


# --------------------------------------------------------------------------- #
# Fixed cleaning steps (no parameters learned from data)
# --------------------------------------------------------------------------- #


def drop_named_columns(x, names, to_drop: Sequence[str]):
    """Remove the columns whose name is in ``to_drop`` (absent names are ignored)."""
    names = _check_names(x, names)
    drop = set(to_drop)
    keep = [name for name in names if name not in drop]
    return _keep(x, names, keep)


def replace_survey_codes(
    x: np.ndarray,
    names: Sequence[str],
    missing_codes: Mapping[str, Sequence[float]] = cb.MISSING_CODES,
    zero_codes: Mapping[str, Sequence[float]] = cb.ZERO_CODES,
) -> np.ndarray:
    """Convert "don't know / refused" codes to NaN and "none" codes to 0."""
    names = _check_names(x, names)
    cleaned = _as_matrix(x).astype(float, copy=True)
    index = {name: i for i, name in enumerate(names)}
    for name, codes in missing_codes.items():
        if name in index:
            column = cleaned[:, index[name]]
            column[np.isin(column, codes)] = np.nan
    for name, codes in zero_codes.items():
        if name in index:
            column = cleaned[:, index[name]]
            column[np.isin(column, codes)] = 0.0
    return cleaned


def frequency_code_to_per_month(values: np.ndarray) -> np.ndarray:
    """Convert BRFSS 3-digit frequency codes to "times per month".

    1xx = xx times per day, 2xx = per week, 3xx = per month, 4xx = per year,
    888 = never (0). Anything else becomes NaN.
    """
    values = np.asarray(values, dtype=float)
    result = np.full(values.shape, np.nan)
    unit = np.floor(values / 100)
    amount = values - 100 * unit
    scale = {1: 30.0, 2: 30.0 / 7.0, 3: 1.0, 4: 1.0 / 12.0}
    for code, factor in scale.items():
        mask = unit == code
        result[mask] = amount[mask] * factor
    result[values == 888] = 0.0
    return result


def fill_skip_patterns(x: np.ndarray, names: Sequence[str]) -> np.ndarray:
    """Fill answers that are missing only because the question did not apply.

    BRFSS skips follow-up questions based on earlier answers. Those NaNs have
    a known meaning, so we fill them with that value instead of the median.
    Uses the raw codes, so call it after ``replace_survey_codes`` and before
    ``apply_value_remaps``.
    """
    names = _check_names(x, names)
    out = x.copy()
    index = {name: i for i, name in enumerate(names)}

    def col(name):
        return out[:, index[name]] if name in index else None

    def fill(target, condition, value):
        column = col(target)
        if column is not None and condition is not None:
            column[np.isnan(column) & condition] = value

    sex, drinks = col("SEX"), col("DRNKANY5")
    bp, cholesterol_test = col("BPHIGH4"), col("BLOODCHO")
    smoked, asthma = col("SMOKE100"), col("ASTHMA3")

    if drinks is not None:  # 2 = no alcohol in the past 30 days
        for name in ("AVEDRNK2", "DRNK3GE5", "MAXDRNKS"):
            fill(name, drinks == 2, 0.0)
    if bp is not None:  # not hypertensive -> not on blood-pressure meds (2 = no)
        fill("BPMEDS", np.isin(bp, (2, 3, 4)), 2.0)
    if sex is not None:  # 1 = male -> not pregnant (2 = no)
        fill("PREGNANT", sex == 1, 2.0)
    if smoked is not None:  # never smoked 100 cigarettes -> smokes "not at all"
        fill("SMOKDAY2", smoked == 2, 3.0)
    if cholesterol_test is not None:  # never had cholesterol checked
        fill("CHOLCHK", cholesterol_test == 2, 5.0)
    if asthma is not None:  # never had asthma -> doesn't have it now (2 = no)
        fill("ASTHNOW", asthma == 2, 2.0)
    return out


def apply_value_remaps(
    x: np.ndarray,
    names: Sequence[str],
    remaps: Mapping[str, Mapping[float, float]] = cb.VALUE_REMAPS,
    frequency_columns: Sequence[str] = cb.FREQUENCY_COLUMNS,
) -> np.ndarray:
    """Make a few variables ordinal and convert frequency codes to amounts."""
    names = _check_names(x, names)
    out = x.copy()
    index = {name: i for i, name in enumerate(names)}
    for name, mapping in remaps.items():
        if name in index:
            original = out[:, index[name]].copy()
            for old, new in mapping.items():
                out[original == old, index[name]] = new
    for name in frequency_columns:
        if name in index:
            out[:, index[name]] = frequency_code_to_per_month(out[:, index[name]])
    return out


def basic_clean(x: np.ndarray, names: Sequence[str], keep_state: bool = False):
    """Run all fixed cleaning steps. Returns the cleaned matrix and its names."""
    names = _check_names(x, names)
    to_drop = (
        cb.ADMIN_COLUMNS
        + cb.UNUSABLE_COLUMNS
        + cb.REDUNDANT_COLUMNS
        + (() if keep_state else (cb.STATE_COLUMN,))
    )
    x, names = drop_named_columns(x, names, to_drop)
    x = replace_survey_codes(x, names)
    x = fill_skip_patterns(x, names)
    x = apply_value_remaps(x, names)
    return x, names


# --------------------------------------------------------------------------- #
# Fitted steps (generic, work on any matrix)
# --------------------------------------------------------------------------- #


def drop_constant_columns(x: np.ndarray, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """Remove columns with zero variance in ``x`` from ``x`` and all ``arrays``."""
    matrix = _as_matrix(x)
    mask = np.nanvar(matrix, axis=0) > 0
    return tuple(_as_matrix(a)[:, mask] for a in (matrix, *arrays))


def drop_columns_with_nan_fraction(
    x: np.ndarray, *arrays: np.ndarray, max_fraction: float = 0.50
) -> tuple[np.ndarray, ...]:
    """Remove columns whose NaN fraction in ``x`` is greater than a limit."""
    if not 0 <= max_fraction <= 1:
        raise ValueError("max_fraction must be between 0 and 1")
    matrix = _as_matrix(x)
    mask = np.isnan(matrix).mean(axis=0) <= max_fraction
    return tuple(_as_matrix(a)[:, mask] for a in (matrix, *arrays))


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
    if matrix.shape[1] != len(params["median"]):
        raise ValueError("feature count does not match preprocessing parameters")
    imputed = np.where(np.isnan(matrix), params["median"], matrix)
    return (imputed - params["mean"]) / params["std"]


def add_bias_column(x: np.ndarray) -> np.ndarray:
    """Prepend an intercept column of ones to a feature matrix."""
    matrix = _as_matrix(x)
    return np.c_[np.ones(matrix.shape[0]), matrix]


# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #


def _pipeline(x, names, params, fit, options):
    """Shared code for fit and transform. When ``fit`` is True, ``params`` is filled."""
    x, names = basic_clean(x, names, keep_state=options["keep_state"])

    # 1. Drop columns that are almost always missing (decided on train).
    if fit:
        nan_fraction = np.isnan(x).mean(axis=0)
        params["kept_after_nan"] = [
            n for n, f in zip(names, nan_fraction) if f <= options["max_nan_fraction"]
        ]
    x, names = _keep(x, names, params["kept_after_nan"])

    # 2. Missing indicators: "was this question answered?" is itself
    #    informative (e.g. a question only asked to diabetics). Identical
    #    indicator columns (same skip pattern) are only kept once.
    if fit:
        indicator_sources, seen = [], set()
        for j, name in enumerate(names):
            missing = np.isnan(x[:, j])
            fraction = missing.mean()
            if options["indicator_min_fraction"] <= fraction < 1:
                signature = np.packbits(missing).tobytes()
                if signature not in seen:
                    seen.add(signature)
                    indicator_sources.append(name)
        params["indicator_sources"] = indicator_sources
    index = {n: i for i, n in enumerate(names)}
    indicators = [np.isnan(x[:, index[n]]).astype(float) for n in params["indicator_sources"]]

    # 3. Yes/No questions (values 1/2) -> 1/0, so weights are easy to read.
    if fit:
        params["binary_columns"] = []
        for j, name in enumerate(names):
            values = np.unique(x[~np.isnan(x[:, j]), j])
            if values.size and set(values.tolist()) <= {1.0, 2.0}:
                params["binary_columns"].append(name)
    for name in params["binary_columns"]:
        column = x[:, index[name]]
        x[:, index[name]] = np.where(column == 2, 0.0, column)

    # 4. One-hot encode nominal variables (categories seen in training only).
    nominal = list(cb.NOMINAL_COLUMNS) + ([cb.STATE_COLUMN] if options["keep_state"] else [])
    nominal = [n for n in nominal if n in index]
    if fit:
        params["categories"] = {
            n: np.unique(x[~np.isnan(x[:, index[n]]), index[n]]).tolist() for n in nominal
        }
    one_hot, one_hot_names = [], []
    for name in nominal:
        column = x[:, index[name]]
        for value in params["categories"][name]:
            one_hot.append((column == value).astype(float))
            one_hot_names.append(f"{name}={value:g}")
    x, names = drop_named_columns(x, names, nominal)

    # 5. Put everything together.
    extra = indicators + one_hot
    if extra:
        x = np.column_stack([x] + extra)
    names = names + [f"{n}_missing" for n in params["indicator_sources"]] + one_hot_names

    # 6. Drop constant columns (on train).
    if fit:
        params["kept_after_constant"] = [
            n for n, v in zip(names, np.nanvar(x, axis=0)) if v > 0
        ]
    x, names = _keep(x, names, params["kept_after_constant"])

    # 7. Clip outliers of continuous columns to training quantiles
    #    (e.g. 99 drinks per day, BMI of 90): linear models are sensitive to them.
    if fit:
        q = options["clip_quantile"]
        params["clip"] = {}
        if q is not None:
            for j, name in enumerate(names):
                column = x[~np.isnan(x[:, j]), j]
                if np.unique(column).size > options["continuous_min_unique"]:
                    low, high = np.quantile(column, [q, 1 - q])
                    params["clip"][name] = (low, high)
    index = {n: i for i, n in enumerate(names)}
    for name, (low, high) in params["clip"].items():
        x[:, index[name]] = np.clip(x[:, index[name]], low, high)

    # 8. Remove near-duplicate columns (|correlation| above a threshold).
    if fit:
        threshold = options["max_correlation"]
        keep = list(names)
        if threshold is not None:
            # A 50k-row sample estimates correlations well and saves memory.
            rows = np.random.default_rng(0).permutation(len(x))[:50_000]
            sample = x[rows]
            filled = np.where(np.isnan(sample), np.nanmedian(sample, axis=0), sample)
            with np.errstate(invalid="ignore", divide="ignore"):
                corr = np.nan_to_num(np.abs(np.corrcoef(filled, rowvar=False)))
            dropped = set()
            for i in range(len(names)):
                if i in dropped:
                    continue
                for j in range(i + 1, len(names)):
                    if j not in dropped and corr[i, j] > threshold:
                        dropped.add(j)
            keep = [n for k, n in enumerate(names) if k not in dropped]
        params["kept_after_correlation"] = keep
    x, names = _keep(x, names, params["kept_after_correlation"])

    # 9. Median imputation + standardisation (train statistics).
    if fit:
        params["scaler"] = fit_preprocessor(x)
    x = transform_features(x, params["scaler"])

    # 10. Optional degree-2 features on the standardized values: squares of the
    #     continuous columns and pairwise products of a few key risk factors.
    #     They let a linear model express e.g. "risk grows faster after 60".
    if options["polynomial"]:
        index = {n: i for i, n in enumerate(names)}
        if fit:
            squares = [n for n in params["clip"] if n in index]
            key = [n for n in POLY_KEY_FEATURES if n in index]
            pairs = [(a, b) for i, a in enumerate(key) for b in key[i + 1:]]
            params["poly_terms"] = [(n, n) for n in squares] + pairs
        new = np.column_stack(
            [x[:, index[a]] * x[:, index[b]] for a, b in params["poly_terms"]]
        )
        if fit:
            params["poly_scaler"] = fit_preprocessor(new)
        x = np.column_stack([x, transform_features(new, params["poly_scaler"])])
        names = names + [f"{a}*{b}" for a, b in params["poly_terms"]]

    if options["add_bias"]:
        x = add_bias_column(x)
        names = ["bias"] + names
    return x, names


# Strong, well-known heart-disease risk factors used for pairwise interactions.
POLY_KEY_FEATURES = (
    "_AGE80", "SEX", "GENHLTH", "_BMI5", "DIABETE3", "BPHIGH4", "TOLDHI2",
    "CVDSTRK3", "_SMOKER3", "CHCKIDNY", "CHCCOPD1", "DIFFWALK",
)

DEFAULT_OPTIONS = {
    "keep_state": False,  # one-hot encode the US state (+~50 columns)
    "max_nan_fraction": 0.80,  # drop columns missing more often than this
    "indicator_min_fraction": 0.05,  # add "_missing" flags above this NaN rate
    "clip_quantile": 0.005,  # clip continuous columns to [0.5%, 99.5%]; None = off
    "continuous_min_unique": 15,  # columns with more distinct values are "continuous"
    "max_correlation": 0.98,  # drop near-duplicate columns; None = off
    "polynomial": False,  # add squares + interactions of key risk factors
    "add_bias": True,  # prepend a column of ones
}


def fit_transform_pipeline(x_train: np.ndarray, names: Sequence[str], **options):
    """Fit the full cleaning pipeline on training data.

    Returns ``(x_clean, feature_names, params)``. Reuse ``params`` with
    ``transform_pipeline`` for validation and test data.
    """
    unknown = set(options) - set(DEFAULT_OPTIONS)
    if unknown:
        raise ValueError(f"unknown options: {sorted(unknown)}")
    params = {"options": {**DEFAULT_OPTIONS, **options}}
    x, out_names = _pipeline(x_train, names, params, fit=True, options=params["options"])
    return x, out_names, params


def transform_pipeline(x: np.ndarray, names: Sequence[str], params: dict):
    """Apply an already-fitted pipeline. Returns ``(x_clean, feature_names)``."""
    return _pipeline(x, names, params, fit=False, options=params["options"])

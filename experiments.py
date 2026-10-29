"""5-fold cross-validation of all models and preprocessing variants.

This produces the baseline comparison and the ablation table of the report:

    python experiments.py --data-dir ../../project1/data

For every fold, the cleaning pipeline is fitted on the 4 training folds only and
applied to the held-out fold, so no information from validation rows leaks
into the cleaning. Every model outputs a score per row; the decision threshold
is the one maximising F1 on the pooled out-of-fold scores, and the F1 of each
fold at that threshold is reported (mean +- std over the 5 folds). ROC-AUC does
not depend on the threshold.

Results are written to ``results/cv_results.md`` and ``results/cv_results.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from helpers import load_csv_data
from implementations import (
    least_squares,
    logistic_regression,
    reg_logistic_regression,
    ridge_regression,
)
from modeling import (
    best_f1_threshold,
    classification_metrics,
    roc_auc,
    stratified_k_fold,
    undersample,
    weighted_reg_logistic_regression,
)
from preprocessing import fit_transform_pipeline, load_feature_names, transform_pipeline

N_FOLDS = 5
SEED = 42
GAMMA = 0.5  # largest step size that converges (1.0 diverges)
MAX_ITERS = 300  # training loss no longer changes after ~300 steps with gamma 0.5


# --------------------------------------------------------------------------- #
# Models: each takes (y01, x_train, fold_seed) and returns a weight vector.
# --------------------------------------------------------------------------- #


def _zeros(x):
    return np.zeros(x.shape[1])


def m_least_squares(y, x, seed):
    return least_squares(y, x)[0]


def m_ridge(lambda_):
    return lambda y, x, seed: ridge_regression(y, x, lambda_)[0]


def m_logistic_undersampled(max_iters, gamma):
    def train(y, x, seed):
        rows = undersample(np.arange(len(y)), y, seed=seed)
        return logistic_regression(y[rows], x[rows], _zeros(x), max_iters, gamma)[0]
    return train


def m_reg_logistic_undersampled(lambda_):
    def train(y, x, seed):
        rows = undersample(np.arange(len(y)), y, seed=seed)
        return reg_logistic_regression(
            y[rows], x[rows], lambda_, _zeros(x), MAX_ITERS, GAMMA
        )[0]
    return train


def m_weighted_logistic(lambda_):
    def train(y, x, seed):
        return weighted_reg_logistic_regression(
            y, x, lambda_, _zeros(x), MAX_ITERS, GAMMA
        )[0]
    return train


# (name, group, pipeline options, model). Experiments sharing the same
# pipeline options reuse the same cleaned data within a fold.
DEFAULT = {}
EXPERIMENTS = [
    # Baselines (course methods)
    ("Least squares", "baseline", DEFAULT, m_least_squares),
    ("Ridge, lambda=1e-4", "baseline", DEFAULT, m_ridge(1e-4)),
    ("Ridge, lambda=1e-3", "baseline", DEFAULT, m_ridge(1e-3)),
    ("Ridge, lambda=1e-2", "baseline", DEFAULT, m_ridge(1e-2)),
    ("Logistic, undersampled, 100 it, gamma=0.05", "baseline", DEFAULT,
     m_logistic_undersampled(100, 0.05)),
    ("Logistic, undersampled, 300 it, gamma=0.5", "baseline", DEFAULT,
     m_logistic_undersampled(MAX_ITERS, GAMMA)),
    ("Reg. logistic, undersampled, lambda=1e-4", "baseline", DEFAULT,
     m_reg_logistic_undersampled(1e-4)),
    ("Reg. logistic, undersampled, lambda=1e-3", "baseline", DEFAULT,
     m_reg_logistic_undersampled(1e-3)),
    ("Reg. logistic, undersampled, lambda=1e-2", "baseline", DEFAULT,
     m_reg_logistic_undersampled(1e-2)),
    # Our modifications
    ("Weighted logistic (all rows)", "ours", DEFAULT, m_weighted_logistic(0.0)),
    ("Weighted logistic + polynomial", "ours", {"polynomial": True},
     m_weighted_logistic(0.0)),
    ("Weighted logistic + polynomial, lambda=1e-4", "ours", {"polynomial": True},
     m_weighted_logistic(1e-4)),
    # Cleaning ablations (each removes/adds one step, weighted logistic)
    ("  - missing indicators", "ablation", {"indicator_min_fraction": 1.1},
     m_weighted_logistic(0.0)),
    ("  - outlier clipping", "ablation", {"clip_quantile": None},
     m_weighted_logistic(0.0)),
    ("  - correlation filter", "ablation", {"max_correlation": None},
     m_weighted_logistic(0.0)),
    ("  + one-hot state", "ablation", {"keep_state": True},
     m_weighted_logistic(0.0)),
]


def run(data_dir: str, out_dir: str = "results", only: list[int] | None = None):    
    x, _, y_raw, _, _ = load_csv_data(data_dir)
    names = load_feature_names(data_dir)
    y = (y_raw == 1).astype(float)
    folds = stratified_k_fold(y, N_FOLDS, SEED)
    experiments = [e for i, e in enumerate(EXPERIMENTS) if only is None or i in only]

    oof_scores = {name: np.zeros(len(y)) for name, *_ in experiments}
    fold_auc = {name: [] for name, *_ in experiments}
    groups = {}
    for name, _, options, _ in experiments:
        groups.setdefault(json.dumps(options, sort_keys=True), []).append(name)
    models = {name: model for name, _, _, model in experiments}
    n_features = {}

    for k, val_idx in enumerate(folds):
        train_idx = np.setdiff1d(np.arange(len(y)), val_idx)
        for key, members in groups.items():
            start = time.time()
            x_tr, feat_names, params = fit_transform_pipeline(
                x[train_idx], names, **json.loads(key)
            )
            x_va, _ = transform_pipeline(x[val_idx], names, params)
            for name in members:
                w = models[name](y[train_idx], x_tr, SEED + k)
                scores = x_va.dot(w)
                oof_scores[name][val_idx] = scores
                fold_auc[name].append(roc_auc(y[val_idx], scores))
                n_features[name] = x_tr.shape[1]
            print(f"fold {k + 1}/{N_FOLDS}  options={key}  "
                  f"{len(members)} models  {time.time() - start:.0f}s", flush=True)
            del x_tr, x_va

    results = []
    for name, group, options, _ in experiments:
        threshold, _ = best_f1_threshold(y, oof_scores[name], n_steps=400)
        f1s, accs, bals = [], [], []
        for val_idx in folds:
            pred = (oof_scores[name][val_idx] >= threshold).astype(int)
            m = classification_metrics(y[val_idx], pred)
            f1s.append(m["f1"]); accs.append(m["accuracy"]); bals.append(m["balanced_accuracy"])
        results.append({
            "name": name, "group": group, "options": options,
            "n_features": n_features[name], "threshold": threshold,
            "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
            "auc_mean": float(np.mean(fold_auc[name])), "auc_std": float(np.std(fold_auc[name])),
            "accuracy_mean": float(np.mean(accs)), "balanced_accuracy_mean": float(np.mean(bals)),
        })

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "cv_results.json"), "w") as handle:
        json.dump(results, handle, indent=2)
    lines = [
        f"{N_FOLDS}-fold stratified CV. F1 at the threshold maximising pooled out-of-fold F1.",
        "",
        "| Experiment | Group | Features | F1 | ROC-AUC | Accuracy | Bal. acc. | Threshold |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['name']} | {r['group']} | {r['n_features']} | "
            f"{r['f1_mean']:.4f} ± {r['f1_std']:.4f} | {r['auc_mean']:.4f} ± {r['auc_std']:.4f} | "
            f"{r['accuracy_mean']:.4f} | {r['balanced_accuracy_mean']:.4f} | {r['threshold']:.4f} |"
        )
    table = "\n".join(lines)
    with open(os.path.join(out_dir, "cv_results.md"), "w") as handle:
        handle.write(table + "\n")
    print(table)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", default="../../project1/data")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--only", type=int, nargs="*", help="indices of experiments to run")
    args = parser.parse_args()
    run(args.data_dir, args.out_dir, args.only)

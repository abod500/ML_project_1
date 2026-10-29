"""Reproduce our best AIcrowd submission.

    python run.py --data-dir ../../project1/data

Steps (all described in the README and the report):
1. Load the data (``helpers.load_csv_data``) and the feature names.
2. Fit the cleaning pipeline on the full training set and apply it to the test
   set (``preprocessing.fit_transform_pipeline`` / ``transform_pipeline``).
3. Train class-weighted logistic regression on all training rows
   (``modeling.weighted_reg_logistic_regression``).
4. Predict +1 when the linear score x.w is above THRESHOLD, which was chosen by
   5-fold cross-validation in ``experiments.py`` to maximise F1.
5. Write ``submission.csv`` (columns Id, Prediction with values -1 / 1).

Everything is deterministic (no random steps), so the CSV is identical on
every run.
"""

from __future__ import annotations

import argparse

import numpy as np

from helpers import create_csv_submission, load_csv_data
from modeling import weighted_reg_logistic_regression
from preprocessing import fit_transform_pipeline, load_feature_names, transform_pipeline

# Best configuration from results/cv_results.md
PIPELINE_OPTIONS = {"polynomial": True}
LAMBDA = 0.0
GAMMA = 0.5
MAX_ITERS = 300
# Decision threshold on the score x.w (a logit, not a probability). It maximises
# F1 on the pooled out-of-fold scores of "Weighted logistic + polynomial" in
# results/cv_results.json (CV F1 = 0.430 +- 0.003, ROC-AUC = 0.862).
THRESHOLD = 1.0383


def main(data_dir: str, output: str) -> None:
    x_train, x_test, y_train, _, test_ids = load_csv_data(data_dir)
    names = load_feature_names(data_dir)
    print(f"train {x_train.shape}, test {x_test.shape}")

    x_tr, feature_names, params = fit_transform_pipeline(x_train, names, **PIPELINE_OPTIONS)
    x_te, _ = transform_pipeline(x_test, names, params)
    print(f"{len(feature_names)} features after cleaning")

    y01 = (y_train == 1).astype(float)
    w, loss = weighted_reg_logistic_regression(
        y01, x_tr, LAMBDA, np.zeros(x_tr.shape[1]), MAX_ITERS, GAMMA
    )
    print(f"training loss (weighted cross-entropy): {loss:.5f}")

    predictions = np.where(x_te.dot(w) >= THRESHOLD, 1, -1)
    create_csv_submission(test_ids, predictions, output)
    print(f"wrote {output}: {np.mean(predictions == 1):.2%} predicted positive")

    top = np.argsort(-np.abs(w))[:10]
    print("largest weights:", ", ".join(f"{feature_names[i]}={w[i]:+.2f}" for i in top))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the AIcrowd submission")
    parser.add_argument("--data-dir", default="../../project1/data")
    parser.add_argument("--output", default="submission.csv")
    args = parser.parse_args()
    main(args.data_dir, args.output)

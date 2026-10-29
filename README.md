# ML Project 1 – Predicting coronary heart disease (MICHD) from BRFSS 2015

Binary classification of whether a respondent of the CDC's 2015 Behavioral Risk
Factor Surveillance System (BRFSS) telephone survey has coronary heart disease
or has had a heart attack (MICHD). Only Python's standard library, NumPy
and matplotlib (plots only) are used.

## Reproducing the submission

Put `x_train.csv`, `y_train.csv` and `x_test.csv` in one folder, then:

```bash
python run.py --data-dir path/to/data          # writes submission.csv
python experiments.py --data-dir path/to/data  # 5-fold CV, writes results/
```

Each run first parses the CSVs with `helpers.load_csv_data` (about 2 min).
After that, `run.py` takes about 1 minute and `experiments.py` about 25.
Both are deterministic.

## Files

| File | Content |
|---|---|
| `implementations.py` | The 6 required methods (GD/SGD least squares, normal equations, ridge, logistic, regularized logistic). |
| `helpers.py` | Course helpers `load_csv_data` and `create_csv_submission`. |
| `brfss_codebook.py` | What the columns mean: columns to drop, survey answer codes for "don't know / refused" (→ NaN) and "none" (→ 0), ordinal recodes, nominal variables. Keyed by column **name**, with sources. |
| `preprocessing.py` | Cleaning and feature pipeline (`fit_transform_pipeline` on train, `transform_pipeline` on validation/test) and `load_feature_names`, which reads the column names from the CSV header. |
| `modeling.py` | Class-weighted logistic regression (our modification), predictions, metrics (F1, ROC-AUC, …), stratified splits and k-fold, threshold search. |
| `experiments.py` | 5-fold cross-validation of baselines, our model, and the cleaning ablations. |
| `run.py` | Trains the final model on all training data and writes `submission.csv`. |
| `test.ipynb` | Exploratory analysis and a first single-split experiment. |
| `results/` | Output of `experiments.py` (`cv_results.md`, `cv_results.json`). |

## Pipeline

The provided CSV already excludes the variables that define the label.

**Fixed cleaning** (`preprocessing.basic_clean`, no parameters learned):
1. Drop interview bookkeeping and survey weights, dates and free-text codes,
   raw answers that have a cleaned CDC "calculated" version (e.g. `WEIGHT2`
   mixes pounds and kilograms, so we keep `WTKG3`) and duplicate groupings of
   age/race. The state (`_STATE`) is dropped by default.
2. Survey codes: 7/9, 77/99, 777/999 ("don't know", "refused") → NaN;
   88/888 ("none") → 0. Some calculated variables appear with the implied
   decimals already applied in this CSV (e.g. 999 instead of 99900), which we
   checked on the data.
3. Skip patterns: questions skipped for a logical reason are filled
   (no alcohol in 30 days → 0 drinks; men → not pregnant; never smoked → does
   not smoke now; no high blood pressure → no BP medication; …).
4. Ordinal recodes (`BPHIGH4`, `DIABETE3`: "borderline" between "no" and "yes";
   `CHECKUP1`: "never" after "5+ years") and 3-digit frequency codes
   (per day/week/month/year) → times per month.

**Fitted on the training data** (`preprocessing.fit_transform_pipeline`):
5. Drop columns with more than 80% missing values.
6. Missing indicators for columns with ≥5% missing (deduplicated when several
   columns share the same skip pattern).
7. Yes/no (1/2) → 1/0; one-hot encoding of marital status, employment, home
   ownership and race.
8. Drop constant columns; clip continuous columns to the [0.5%, 99.5%]
   training quantiles; drop one of each pair of columns with |correlation| > 0.98.
9. Median imputation and standardization.
10. Optional degree-2 features: squares of continuous columns and pairwise
    products of 12 key risk factors (age, sex, general health, BMI, diabetes,
    blood pressure, cholesterol, stroke, smoking, kidney disease, COPD,
    difficulty walking).

**Model**: logistic regression trained on all rows with positive samples
weighted by n_neg / n_pos (≈ 10.3) instead of undersampling, by gradient
descent (γ = 0.5, 300 iterations; γ = 1 diverges). The decision threshold
maximizes F1 on the pooled out-of-fold scores of 5-fold stratified CV.

## Evaluation

5-fold stratified cross-validation. The cleaning pipeline is refitted inside
each fold on the training part only. F1 is measured at the threshold that
maximizes F1 on the pooled out-of-fold scores (mean ± std over folds). Full
table: `results/cv_results.md`.

| Model | Features | F1 | ROC-AUC |
|---|---|---|---|
| Least squares | 189 | 0.417 ± 0.003 | 0.852 |
| Ridge (λ = 1e-3) | 189 | 0.417 ± 0.003 | 0.853 |
| Logistic, undersampled, 100 it, γ = 0.05 | 189 | 0.418 ± 0.005 | 0.855 |
| Logistic, undersampled, 300 it, γ = 0.5 | 189 | 0.425 ± 0.004 | 0.859 |
| Reg. logistic, undersampled, λ = 1e-3 | 189 | 0.425 ± 0.003 | 0.860 |
| Weighted logistic, all rows | 189 | 0.426 ± 0.003 | 0.860 |
| **Weighted logistic + polynomial (submitted)** | 292 | **0.430 ± 0.003** | **0.862** |

Main findings: running gradient descent to convergence matters most
(+0.007 F1); degree-2 features add +0.004 F1; class weighting instead of
undersampling, L2 regularization and the individual cleaning steps (missing
indicators, clipping, correlation filter, one-hot state) each change F1 by
less than the fold-to-fold standard deviation.

The final model (`run.py`) is trained on all 328,135 training rows with
threshold 1.0383 on the logit x·w, and predicts 13.6% of the test rows as
positive.
# Initial-assessment model comparison

Target: existing angiographic disease presence (`num > 0`), not future events.
920 raw records; removed duplicate IDs [406, 908].
734 training and 184 held-out test records.
The 80/20 split is stratified by hospital and binary target, seed 42.

## Preprocessing and comparison design

Features: age, trestbps, chol, sex, cp, fbs, restecg. ID, hospital, and all exercise/imaging results are excluded from predictors.
Zero cholesterol and blood pressure are replaced with missing values. Numeric values use training-fold
median imputation, plus an unscaled missingness indicator for each numeric feature. Logistic regression
standardizes imputed numeric values; tree models do not scale. Categorical values use a missing category
and one-hot encoding (unseen categories map to all zeros). Every fit learns its own preprocessing.
No SMOTE or class weights: binary imbalance is mild. Hyperparameters were fixed before evaluation;
this is an initial comparison, with no hyperparameter search or post-hoc probability calibration.

Five-fold stratified CV runs on training data only. A candidate is selected by mean training CV ROC-AUC:
**Logistic regression**. All three models are reported on the identical test set.
An exploratory alternative cutoff is the largest training OOF cutoff achieving at least 80% sensitivity;
this target is not a clinical requirement and does not guarantee 80% on unseen data.

## Training cross-validation

| model | ROC-AUC mean | ROC-AUC SD |
| --- | --- | --- |
| Gradient boosting | 0.846 | 0.033 |
| Logistic regression | 0.858 | 0.036 |
| Random forest | 0.856 | 0.040 |

## Held-out test, cutoff 0.5

| model | roc_auc | average_precision | sensitivity | specificity | precision | f1 | balanced_accuracy | accuracy | brier | log_loss | fn | fp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic regression | 0.849 | 0.862 | 0.833 | 0.768 | 0.817 | 0.825 | 0.801 | 0.804 | 0.152 | 0.474 | 17 | 19 |
| Random forest | 0.851 | 0.851 | 0.814 | 0.744 | 0.798 | 0.806 | 0.779 | 0.783 | 0.149 | 0.466 | 19 | 21 |
| Gradient boosting | 0.843 | 0.842 | 0.853 | 0.720 | 0.791 | 0.821 | 0.786 | 0.793 | 0.154 | 0.476 | 15 | 23 |
| Constant prevalence reference | 0.500 | 0.554 | 1.000 | 0.000 | 0.554 | 0.713 | 0.500 | 0.554 | 0.247 | 0.687 | 0 | 82 |

Average precision summarizes precision–recall (not trapezoidal PR-AUC). Brier score and log loss
measure probabilistic prediction quality, including calibration; lower is better. The plot also shows
observed versus predicted probabilities. A constant training-prevalence reference supplies a useful
accuracy and probability baseline.

## Held-out test, training-selected sensitivity cutoff

| model | threshold | sensitivity | specificity | precision | fn | fp |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic regression | 0.564 | 0.794 | 0.780 | 0.818 | 21 | 18 |
| Random forest | 0.570 | 0.784 | 0.780 | 0.816 | 22 | 18 |
| Gradient boosting | 0.589 | 0.814 | 0.744 | 0.798 | 19 | 21 |

## Hospital transfer, training partition only, cutoff 0.5

For each hospital, fit on the other three hospitals' training records and evaluate on that hospital's
training records. The final test partition remains separate. No pooled thresholds are used here.

| model | held_out_hospital | n | positive_fraction | roc_auc | average_precision | sensitivity | specificity | brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic regression | Cleveland | 243 | 0.457 | 0.831 | 0.808 | 0.829 | 0.720 | 0.170 |
| Logistic regression | Hungary | 234 | 0.363 | 0.845 | 0.742 | 0.765 | 0.819 | 0.152 |
| Logistic regression | Switzerland | 98 | 0.939 | 0.761 | 0.965 | 0.902 | 0.667 | 0.107 |
| Logistic regression | VA Long Beach | 159 | 0.742 | 0.697 | 0.849 | 0.847 | 0.366 | 0.182 |
| Random forest | Cleveland | 243 | 0.457 | 0.838 | 0.800 | 0.838 | 0.697 | 0.170 |
| Random forest | Hungary | 234 | 0.363 | 0.839 | 0.699 | 0.788 | 0.785 | 0.162 |
| Random forest | Switzerland | 98 | 0.939 | 0.737 | 0.968 | 0.880 | 0.500 | 0.119 |
| Random forest | VA Long Beach | 159 | 0.742 | 0.688 | 0.834 | 0.797 | 0.488 | 0.184 |
| Gradient boosting | Cleveland | 243 | 0.457 | 0.826 | 0.800 | 0.820 | 0.674 | 0.174 |
| Gradient boosting | Hungary | 234 | 0.363 | 0.836 | 0.679 | 0.729 | 0.819 | 0.160 |
| Gradient boosting | Switzerland | 98 | 0.939 | 0.800 | 0.982 | 0.891 | 0.500 | 0.121 |
| Gradient boosting | VA Long Beach | 159 | 0.742 | 0.683 | 0.831 | 0.847 | 0.390 | 0.182 |

## Limits and interpretation

The test set is small; point-estimate differences do not establish superiority. Training-fold SD is
descriptive variability, not a confidence interval. Hospital and sex subgroup metrics are in
`test_subgroups.csv`; some subgroups are too small for stable estimates. Source-dependent missingness
can still encode hospital shortcuts without a hospital predictor. Switzerland has no usable cholesterol.
The random held-out test represents this hospital mixture, not prospective deployment.
Availability of baseline lab/ECG features at the intended prediction moment still needs confirmation.
Once inspected, this test set must not be used to tune features, hyperparameters, or cutoffs.

Reproduce with `python train_compare.py`. Exact model parameters and sklearn version are in
`run_metadata.json`; row assignments and raw predictions are saved for audit. No model is deployed.

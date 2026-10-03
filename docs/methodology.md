# Methodology and evaluation boundaries

## Prediction question

Can initial-assessment information distinguish existing angiographic disease presence (`num > 0`)?
The baseline uses age, sex, chest-pain category, resting BP, cholesterol, fasting-sugar indicator, and
resting ECG. Availability of labs and ECG at the intended prediction moment remains an assumption.

ID is excluded because it indexes records and reveals hospital ranges. Hospital is excluded from
predictors but retained for splitting and evaluation. Exercise and imaging measurements are excluded
because an initial-assessment prediction must not use later diagnostic information.

## Preprocessing

1. Drop the two duplicate copies and convert zero cholesterol/BP to missing. These fixed data-quality
   rules are applied before splitting; they do not estimate population statistics or use disease labels.
2. Create an 80/20 split, stratified jointly by hospital and binary target, with random seed 42.
   This gives 734 training records and 184 test records, with IDs stored in `results/split.csv`.
3. Numeric features use training median imputation. A separate binary missingness flag is added for
   each numeric feature, including age, whose flag is always zero here.
4. Categorical values use a `missing` category and one-hot encoding. Unknown categories at evaluation
   map to all zeros. Imputation never treats blank Boolean values as false.
5. Logistic regression standardizes the imputed numeric features; binary missingness indicators and
   categorical dummies stay unscaled. Tree models do not use scaling.

Every estimator is a scikit-learn pipeline. Medians, category mappings, and scaling statistics are fit
inside each training fold. The validation/test partition is transformed with those learned values.

## Initial comparison

Logistic regression, random forest, and histogram gradient boosting use fixed, recorded parameters.
Five-fold stratified CV on the training partition selects a candidate by mean ROC-AUC. All three are
then evaluated on the same test partition, alongside a constant training-prevalence reference.

ROC-AUC and average precision measure ranking. Confusion matrices, sensitivity, specificity, precision,
F1 and balanced accuracy describe threshold-dependent behavior. Brier score, log loss, and calibration
plots assess probability quality. Average precision is not trapezoidal PR-AUC; Brier is not a pure
calibration metric because it also reflects discrimination.

## Thresholds

The initial experiment reports 0.5 and an alternative chosen from training out-of-fold predictions to
achieve at least 80% sensitivity. The follow-up evaluates a 90% sensitivity target. Both are exploratory
assumptions, not clinical operating requirements. No test labels select thresholds.

The largest feasible cutoff is selected, which minimizes false positives among observed cutoffs
meeting the sensitivity target. Each outer fold has its own training-selected cutoff; the follow-up
does not produce a single final deployment threshold.

## Hospital transfer

Within the training partition, fit on three hospitals and evaluate on the fourth, using threshold 0.5.
This checks transfer across source cohorts. It is separate from the mixed-hospital test comparison.
Switzerland's transfer subset has only six negative records, making specificity and AUC unstable.
Average precision must be interpreted alongside prevalence, especially for Switzerland's 93.9% positives.

## Follow-up validation

New comparisons use the original training partition only because the original test results were
already inspected. Five outer folds produce predictions on unseen records; three inner folds select
hyperparameters by ROC-AUC. Calibration and preprocessing stay within the training partitions.

Inner out-of-fold predictions select thresholds after hyperparameter selection on that outer training
partition. These inner predictions reuse the selected hyperparameters, so they are not fully independent
of hyperparameter selection. Outer validation still remains untouched by both steps.

The outer results estimate each experiment's procedure. Selecting the best experiment after looking
at all outer results can still introduce selection optimism; no new untouched confirmation set exists.

## Uncertainty and scope

Paired class-stratified bootstrap resampling compares fixed outer predictions against baseline.
The resulting intervals omit retraining variation, dependence between CV folds, and multiple-experiment
selection. They are descriptive uncertainty checks, not significance claims.

The study is retrospective, small, and source-dependent. It does not establish prospective performance,
patient benefit, transportability to modern practice, or suitability for clinical deployment.

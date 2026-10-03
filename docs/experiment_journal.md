# Experiment journal: questions, decisions, and findings

This is a retrospective account of the study stages. The commit history organizes those stages;
commit timestamps record when the repository was assembled, not a fabricated multi-week timeline.

## 1. Establish what the data can support

**Question:** Are these records suitable for disease classification, and what could contaminate evaluation?

The audit found hospital-dependent missingness, two duplicate pairs hidden by unique IDs, 172 zero
cholesterol measurements, and one zero resting-BP measurement. Only 299 rows were complete, nearly
all from Cleveland. Complete-case deletion would change the population.

**Decision:** Deduplicate conservatively, treat invalid zeros as missing, preserve incomplete records,
and target binary existing disease presence. Exclude ID, hospital, and later test results from the
initial-assessment model. Do not interpret five-class labels as harmonized severity across hospitals.

**Evidence:** [Data audit](data_audit.md), [audit JSON](../results/data_audit.json).

## 2. Test whether complexity earns its place

**Question:** Do tree models materially improve on a regularized linear baseline?

The initial comparison used fixed parameters and fold-local preprocessing. Logistic regression had
the highest mean training CV ROC-AUC, 0.858. On the 184-record test, logistic, forest, and boosting
scored 0.849, 0.851, and 0.843. Small differences did not identify a clear winner.

**Decision:** Keep logistic regression as the reference and investigate errors, thresholds, and transfer,
rather than conclude that the largest point estimate is a robust improvement.

**Evidence:** [Initial report](../results/comparison_report.md), [methodology](methodology.md).

## 3. Move from accuracy to explicit error tradeoffs

**Question:** What does each model get wrong, and is cutoff 0.5 defensible?

At 0.5, logistic produced 17 missed positive cases and 19 false positives; boosting produced 15 and
23. An exploratory 80%-sensitivity training-selected logistic cutoff was 0.564; on test it increased
misses to 21 while reducing false positives to 18. No cutoff was established as optimal.

**Decision:** Report every confusion-matrix cell and compare operating rules with a declared objective.
The original test had now been inspected, so further experiments would use training data only.

**Evidence:** [Error analysis](error_analysis.md), [test matrices](../results/experiments/test_confusion_matrices.png).

## 4. Test specific alternatives using nested validation

Five outer folds supplied unseen predictions on the original 734 training records. Three inner folds
selected hyperparameters. Thresholds came from inner predictions, targeting an exploratory 90%
sensitivity. All experiment variants shared the same outer folds.

| Experiment | Rationale | Outer pooled ROC-AUC | Finding |
| --- | --- | ---: | --- |
| Baseline logistic | Reference | 0.860 | Remained competitive |
| Tuned logistic | Vary regularization | 0.857 | No ranking gain |
| Tuned forest | Search leaf size and depth | 0.856 | No ranking gain |
| Tuned boosting | Search tree complexity and regularization | 0.853 | No ranking gain |
| Remove cholesterol | Test dependence on source-specific missingness | 0.855 | No mixed-source gain; transfer effect not established |
| Remove numeric missingness flags | Test possible source shortcuts | 0.857 | No mixed-source gain; categorical missingness still visible |
| Sigmoid-calibrated logistic | Improve probability quality | 0.860 | Brier worsened from 0.148 to 0.151 |
| After-exercise logistic | Evaluate the value of later diagnostic information | 0.893 | Improved a different prediction-time task |

The logistic sensitivity rule changed false negatives from 66 to 38 and false positives from 89 to
115. Achieved sensitivity was 90.6%; specificity was 64.9%. Threshold choice changed the operational
balance more visibly than switching models.

**Decision:** Do not add complexity or calibration merely because it is available. Keep after-exercise
results separate from the initial-assessment comparison. Paired bootstrap intervals are conditional
on fixed outer predictions and do not establish retraining uncertainty or clinical significance.

**Evidence:** [Nested report](../results/experiments/experiment_report.md),
[selected parameters](../results/experiments/selected_parameters.json),
[outer predictions](../results/experiments/outer_predictions.csv).

## 5. Evaluate the limit that matters most

**Question:** Does performance transfer to another hospital?

In the initial training-only hospital holdout, all three models scored about 0.68–0.70 on VA Long Beach.
The logistic specificity there was 36.6%. This cohort-transfer result is more concerning than small
mixed-source model differences. It uses a different population from the original test, so the AUC gap
is descriptive, not a paired estimate of performance loss.

**Conclusion:** Extra model complexity did not demonstrate value with initial-assessment features in
these experiments. Error objectives and hospital shift deserve more attention. No model is deployed.

Further work would require a new confirmation cohort, measurement timing, patient identifiers to
check repeated records, and explicit operating requirements. The existing dataset cannot resolve
those limitations by adding more hyperparameter searches.

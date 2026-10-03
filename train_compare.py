"""Reproducible initial-assessment heart-disease model comparison.

Run: python train_compare.py
Only training data determine preprocessing and sensitivity thresholds.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, log_loss,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
SEED = 42
NUMERIC = ["age", "trestbps", "chol"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg"]
FEATURES = NUMERIC + CATEGORICAL


def pipeline(model, scale=False):
    steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        steps.append(("scale", StandardScaler()))
    prep = ColumnTransformer([
        ("numeric", Pipeline(steps), NUMERIC),
        ("numeric_missing", MissingIndicator(features="all"), NUMERIC),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAL),
    ], remainder="drop")
    return Pipeline([("preprocess", prep), ("model", model)])


def metrics(y, p, threshold=0.5):
    pred = (np.asarray(p) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": len(y), "positive_fraction": float(np.mean(y)),
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "average_precision": float(average_precision_score(y, p)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "sensitivity": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def choose_threshold(y, p, sensitivity=0.80):
    # Highest threshold achieving the desired sensitivity on training OOF data.
    _, tpr, thresholds = roc_curve(y, p, drop_intermediate=False)
    feasible = np.flatnonzero((tpr >= sensitivity) & np.isfinite(thresholds))
    return float(np.max(thresholds[feasible]))


def main():
    OUT.mkdir(exist_ok=True)
    raw = pd.read_csv(ROOT / "heart_disease_uci.csv")
    duplicate = raw.duplicated(subset=raw.columns.drop("id"), keep="first")
    removed_ids = raw.loc[duplicate, "id"].tolist()
    data = raw.loc[~duplicate].copy().reset_index(drop=True)
    for col in ["trestbps", "chol"]:
        data.loc[data[col] == 0, col] = np.nan
    # Stable string categories avoid mixed Boolean/string imputation behavior.
    for col in CATEGORICAL:
        data[col] = data[col].map(lambda value: str(value) if pd.notna(value) else np.nan)
    X = data[FEATURES]
    y = (data["num"] > 0).astype(int)
    # Preserve both hospital representation and within-hospital disease balance.
    strata = data["dataset"] + "_" + y.astype(str)
    train, test = train_test_split(
        np.arange(len(data)), test_size=0.20, random_state=SEED, stratify=strata
    )
    assert set(train).isdisjoint(test)
    assert not data.drop(columns="id").duplicated().any()
    split = data[["id", "dataset"]].copy()
    split["target"] = y
    split["partition"] = "train"
    split.loc[test, "partition"] = "test"
    split.to_csv(OUT / "split.csv", index=False)

    models = {
        "Logistic regression": pipeline(LogisticRegression(C=1.0, max_iter=2000), scale=True),
        "Random forest": pipeline(RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, max_features="sqrt",
            random_state=SEED, n_jobs=-1,
        )),
        "Gradient boosting": pipeline(HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.05, max_leaf_nodes=7,
            min_samples_leaf=20, l2_regularization=1.0,
            early_stopping=False, random_state=SEED,
        )),
    }
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X.iloc[train], y.iloc[train]))
    cv_rows, test_rows, site_rows, subgroup_rows = [], [], [], []
    predictions = data.loc[test, ["id", "dataset", "sex"]].copy()
    predictions["target"] = y.iloc[test]
    oof_table = data.loc[train, ["id", "dataset"]].copy()
    oof_table["target"] = y.iloc[train]
    thresholds = {}
    fitted = {}
    for name, estimator in models.items():
        print(f"Training {name}", flush=True)
        oof = np.zeros(len(train))
        for fold, (fit_idx, val_idx) in enumerate(folds, start=1):
            model = clone(estimator).fit(X.iloc[train[fit_idx]], y.iloc[train[fit_idx]])
            prob = model.predict_proba(X.iloc[train[val_idx]])[:, 1]
            oof[val_idx] = prob
            cv_rows.append({"model": name, "fold": fold, **metrics(y.iloc[train[val_idx]], prob)})
        oof_table[name] = oof
        threshold = choose_threshold(y.iloc[train], oof)
        thresholds[name] = {"value": threshold, "training_oof": metrics(y.iloc[train], oof, threshold)}
        fitted[name] = clone(estimator).fit(X.iloc[train], y.iloc[train])
        # No calibration, fitting, threshold selection, or tuning on test data.
        prob = fitted[name].predict_proba(X.iloc[test])[:, 1]
        predictions[name] = prob
        for operating_point, cutoff in [("default_0.5", 0.5), ("training_sensitivity_0.80", threshold)]:
            test_rows.append({"model": name, "operating_point": operating_point, **metrics(y.iloc[test], prob, cutoff)})
        for field in ["dataset", "sex"]:
            for group in sorted(data.loc[test, field].unique()):
                mask = (data.loc[test, field] == group).to_numpy()
                subgroup_rows.append({"model": name, "field": field, "group": group,
                                      **metrics(y.iloc[test].to_numpy()[mask], prob[mask])})
        # Transfer evaluation on the training partition only. Fixed 0.5 cutoff;
        # pooled OOF thresholds would have seen the held-out hospital labels.
        for hospital in sorted(data.loc[train, "dataset"].unique()):
            mask = (data.loc[train, "dataset"] == hospital).to_numpy()
            fit_idx, val_idx = train[~mask], train[mask]
            model = clone(estimator).fit(X.iloc[fit_idx], y.iloc[fit_idx])
            prob_site = model.predict_proba(X.iloc[val_idx])[:, 1]
            site_rows.append({"model": name, "held_out_hospital": hospital,
                              **metrics(y.iloc[val_idx], prob_site)})

    prevalence = float(y.iloc[train].mean())
    test_rows.append({"model": "Constant prevalence reference", "operating_point": "default_0.5",
                      **metrics(y.iloc[test], np.full(len(test), prevalence))})
    cv_df = pd.DataFrame(cv_rows)
    cv_summary = cv_df.groupby("model")[
        ["roc_auc", "average_precision", "balanced_accuracy", "sensitivity", "specificity", "brier"]
    ].agg(["mean", "std"])
    cv_df.to_csv(OUT / "cv_folds.csv", index=False)
    cv_summary.to_csv(OUT / "cv_summary.csv")
    pd.DataFrame(test_rows).to_csv(OUT / "test_metrics.csv", index=False)
    pd.DataFrame(site_rows).to_csv(OUT / "hospital_transfer.csv", index=False)
    pd.DataFrame(subgroup_rows).to_csv(OUT / "test_subgroups.csv", index=False)
    predictions.to_csv(OUT / "test_predictions.csv", index=False)
    oof_table.to_csv(OUT / "training_oof_predictions.csv", index=False)
    selected = cv_summary[("roc_auc", "mean")].idxmax()
    metadata = {
        "seed": SEED, "sklearn_version": sklearn.__version__,
        "original_records": len(raw), "deduplicated_records": len(data),
        "removed_duplicate_ids": removed_ids, "train_records": len(train), "test_records": len(test),
        "features": FEATURES, "target": "num > 0", "thresholds": thresholds,
        "candidate_selected_by_training_cv_roc_auc": selected,
        "hyperparameters": {name: est.named_steps["model"].get_params() for name, est in models.items()},
    }
    (OUT / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    plot_results(y.iloc[test], predictions, models)
    write_report(metadata, cv_summary, test_rows, site_rows)
    print(pd.DataFrame(test_rows).to_string(index=False), flush=True)
    print(f"Outputs: {OUT}", flush=True)


def plot_results(y, predictions, models):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for name in models:
        p = predictions[name].to_numpy()
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, label=f"{name} ({roc_auc_score(y, p):.3f})")
        precision, recall, _ = precision_recall_curve(y, p)
        axes[1].plot(recall, precision, label=f"{name} ({average_precision_score(y, p):.3f})")
        observed, predicted = calibration_curve(y, p, n_bins=5, strategy="quantile")
        axes[2].plot(predicted, observed, "o-", label=name)
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[1].axhline(float(y.mean()), color="black", linestyle="--", alpha=0.4)
    axes[2].plot([0, 1], [0, 1], "k--", alpha=0.4)
    for ax, title, xlabel, ylabel in zip(
        axes, ["ROC (AUC)", "Precision–recall (average precision)", "Calibration: 5 equal-count bins"],
        ["False positive rate", "Recall / sensitivity", "Mean predicted probability"],
        ["True positive rate", "Precision", "Observed disease fraction"],
    ):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel, xlim=(0, 1), ylim=(0, 1))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.15)
    fig.suptitle(f"Held-out test set (n={len(y)}); no post-hoc probability calibration")
    fig.tight_layout()
    fig.savefig(OUT / "comparison.png", dpi=180)
    plt.close(fig)


def markdown_table(rows, columns):
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = [f"{row[c]:.3f}" if isinstance(row[c], float) else str(row[c]) for c in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(meta, cv, test_rows, site_rows):
    default = [r for r in test_rows if r["operating_point"] == "default_0.5"]
    adjusted = [r for r in test_rows if r["operating_point"] != "default_0.5"]
    cv_rows = [{"model": name, "ROC-AUC mean": float(cv.loc[name, ("roc_auc", "mean")]),
                "ROC-AUC SD": float(cv.loc[name, ("roc_auc", "std")])} for name in cv.index]
    text = f"""# Initial-assessment model comparison

Target: existing angiographic disease presence (`num > 0`), not future events.
{meta['original_records']} raw records; removed duplicate IDs {meta['removed_duplicate_ids']}.
{meta['train_records']} training and {meta['test_records']} held-out test records.
The 80/20 split is stratified by hospital and binary target, seed {SEED}.

## Preprocessing and comparison design

Features: {', '.join(FEATURES)}. ID, hospital, and all exercise/imaging results are excluded from predictors.
Zero cholesterol and blood pressure are replaced with missing values. Numeric values use training-fold
median imputation, plus an unscaled missingness indicator for each numeric feature. Logistic regression
standardizes imputed numeric values; tree models do not scale. Categorical values use a missing category
and one-hot encoding (unseen categories map to all zeros). Every fit learns its own preprocessing.
No SMOTE or class weights: binary imbalance is mild. Hyperparameters were fixed before evaluation;
this is an initial comparison, with no hyperparameter search or post-hoc probability calibration.

Five-fold stratified CV runs on training data only. A candidate is selected by mean training CV ROC-AUC:
**{meta['candidate_selected_by_training_cv_roc_auc']}**. All three models are reported on the identical test set.
An exploratory alternative cutoff is the largest training OOF cutoff achieving at least 80% sensitivity;
this target is not a clinical requirement and does not guarantee 80% on unseen data.

## Training cross-validation

{markdown_table(cv_rows, ['model', 'ROC-AUC mean', 'ROC-AUC SD'])}

## Held-out test, cutoff 0.5

{markdown_table(default, ['model', 'roc_auc', 'average_precision', 'sensitivity', 'specificity', 'precision', 'f1', 'balanced_accuracy', 'accuracy', 'brier', 'log_loss', 'fn', 'fp'])}

Average precision summarizes precision–recall (not trapezoidal PR-AUC). Brier score and log loss
measure probabilistic prediction quality, including calibration; lower is better. The plot also shows
observed versus predicted probabilities. A constant training-prevalence reference supplies a useful
accuracy and probability baseline.

## Held-out test, training-selected sensitivity cutoff

{markdown_table(adjusted, ['model', 'threshold', 'sensitivity', 'specificity', 'precision', 'fn', 'fp'])}

## Hospital transfer, training partition only, cutoff 0.5

For each hospital, fit on the other three hospitals' training records and evaluate on that hospital's
training records. The final test partition remains separate. No pooled thresholds are used here.

{markdown_table(site_rows, ['model', 'held_out_hospital', 'n', 'positive_fraction', 'roc_auc', 'average_precision', 'sensitivity', 'specificity', 'brier'])}

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
"""
    (OUT / "comparison_report.md").write_text(text)


if __name__ == "__main__":
    main()

"""Training-only nested experiments; preserves the previously inspected test set."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from train_compare import ROOT, SEED, NUMERIC, CATEGORICAL, metrics, choose_threshold, markdown_table
from visualize_errors import plot_matrices

OUT = ROOT / 'results' / 'experiments'


def make_pipeline(model, numeric=NUMERIC, categorical=CATEGORICAL, flags=True, scale=False):
    numeric_steps = [('impute', SimpleImputer(strategy='median', keep_empty_features=True))]
    if scale:
        numeric_steps.append(('scale', StandardScaler()))
    transforms = [('numeric', Pipeline(numeric_steps), numeric),
                  ('categorical', Pipeline([
                      ('impute', SimpleImputer(strategy='constant', fill_value='missing')),
                      ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                  ]), categorical)]
    if flags:
        transforms.append(('missing', MissingIndicator(features='all'), numeric))
    return Pipeline([('preprocess', ColumnTransformer(transforms)), ('model', model)])


def main():
    OUT.mkdir(exist_ok=True)
    data = pd.read_csv(ROOT / 'heart_disease_uci.csv')
    data = data.loc[~data.duplicated(subset=data.columns.drop('id'))].copy()
    data.loc[data['chol'] == 0, 'chol'] = np.nan
    data.loc[data['trestbps'] == 0, 'trestbps'] = np.nan
    cats = CATEGORICAL + ['exang', 'slope']
    for c in cats:
        data[c] = data[c].map(lambda v: str(v) if pd.notna(v) else np.nan)
    assignment = pd.read_csv(ROOT / 'results' / 'split.csv')
    train_ids = assignment.loc[assignment['partition'] == 'train', 'id'].tolist()
    data = data.set_index('id').loc[train_ids].reset_index()
    assert len(data) == 734 and not set(data['id']) & set(assignment.loc[assignment['partition']=='test','id'])
    y = (data['num'] > 0).astype(int)
    X = data.drop(columns=['num', 'id', 'dataset'])
    logistic = LogisticRegression(max_iter=2000)
    grids_lr = {'model__C': [0.03, 0.3, 3.0, 30.0]}
    candidates = {
        'Baseline logistic': (make_pipeline(clone(logistic), scale=True), {}),
        'Tuned logistic': (make_pipeline(clone(logistic), scale=True), grids_lr),
        'Tuned forest': (make_pipeline(RandomForestClassifier(n_estimators=150, random_state=SEED, n_jobs=-1)),
                         {'model__min_samples_leaf': [1, 5, 10], 'model__max_depth': [None, 5]}),
        'Tuned boosting': (make_pipeline(HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05,
                            early_stopping=False, random_state=SEED)),
                           {'model__max_leaf_nodes': [3, 7, 15], 'model__l2_regularization': [1., 10.]}),
        'Logistic without cholesterol': (make_pipeline(clone(logistic), numeric=['age','trestbps'], scale=True), grids_lr),
        'Logistic without missing flags': (make_pipeline(clone(logistic), flags=False, scale=True), grids_lr),
        'Sigmoid-calibrated logistic': (CalibratedClassifierCV(make_pipeline(clone(logistic), scale=True),
                                                             method='sigmoid', cv=3, ensemble=False), {}),
        'After-exercise logistic': (make_pipeline(clone(logistic), numeric=NUMERIC+['thalch','oldpeak'],
                                  categorical=CATEGORICAL+['exang','slope'], scale=True), grids_lr),
    }
    outer = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X,y))
    predictions = data[['id','dataset']].copy()
    predictions['target'] = y
    fold_rows, summaries, selected_parameters = [], [], []
    for name, (estimator, grid) in candidates.items():
        print(f'Experiment: {name}', flush=True)
        p_all, classifications = np.zeros(len(y)), np.zeros(len(y), dtype=int)
        for fold, (fit_idx, val_idx) in enumerate(outer, start=1):
            inner = StratifiedKFold(3, shuffle=True, random_state=SEED+fold)
            if grid:
                search = GridSearchCV(estimator, grid, scoring='roc_auc', cv=inner, n_jobs=1)
                search.fit(X.iloc[fit_idx], y.iloc[fit_idx])
                fitted = search.best_estimator_
                params = search.best_params_
            else:
                fitted = clone(estimator).fit(X.iloc[fit_idx],y.iloc[fit_idx])
                params = {}
            # Hyperparameters are chosen within this outer training fold. The
            # inner OOF threshold is exploratory; outer validation stays unseen.
            inner_p = cross_val_predict(clone(fitted), X.iloc[fit_idx], y.iloc[fit_idx],
                                        cv=inner, method='predict_proba', n_jobs=1)[:,1]
            threshold = choose_threshold(y.iloc[fit_idx], inner_p, sensitivity=0.90)
            prob = fitted.predict_proba(X.iloc[val_idx])[:,1]
            p_all[val_idx] = prob
            classifications[val_idx] = prob >= threshold
            selected_parameters.append({'model':name,'fold':fold,'params':params,'threshold':threshold})
            for op, cutoff in [('default_0.5',0.5),('inner_target_sensitivity_0.90',threshold)]:
                fold_rows.append({'model':name,'fold':fold,'operating_point':op,
                                  **metrics(y.iloc[val_idx],prob,cutoff)})
        predictions[name] = p_all
        predictions[name+'__decision_90'] = classifications
        summaries.append({'model':name,'operating_point':'default_0.5',**metrics(y,p_all)})
        row = metrics(y,p_all)
        # Each record has its own outer-fold training-selected threshold.
        tn,fp,fn,tp = confusion_matrix(y,classifications,labels=[0,1]).ravel()
        row.update(threshold=None,tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),
                   sensitivity=float(tp/(tp+fn)),specificity=float(tn/(tn+fp)),
                   precision=float(tp/(tp+fp)),accuracy=float((tp+tn)/len(y)),
                   balanced_accuracy=float((tp/(tp+fn)+tn/(tn+fp))/2),
                   f1=float(2*tp/(2*tp+fp+fn)))
        summaries.append({'model':name,'operating_point':'inner_target_sensitivity_0.90',**row})
    pd.DataFrame(fold_rows).to_csv(OUT/'nested_folds.csv',index=False)
    pd.DataFrame(summaries).to_csv(OUT/'nested_summary.csv',index=False)
    predictions.to_csv(OUT/'outer_predictions.csv',index=False)
    (OUT/'selected_parameters.json').write_text(json.dumps(selected_parameters,indent=2))
    differences = paired_bootstrap(y.to_numpy(),predictions,list(candidates))
    pd.DataFrame(differences).to_csv(OUT/'paired_auc_differences.csv',index=False)
    # Reuse existing predictions only: no additional models fit to the test set.
    test_rows = pd.read_csv(ROOT/'results'/'test_metrics.csv')
    old = test_rows.loc[(test_rows.operating_point=='default_0.5') &
                       (test_rows.model!='Constant prevalence reference')].to_dict('records')
    plot_matrices(old,OUT/'test_confusion_matrices.png','Previously evaluated test set: n=184, threshold=0.5')
    main_rows = [r for r in summaries if r['operating_point']=='inner_target_sensitivity_0.90'
                 and r['model'] in ['Baseline logistic','Tuned forest','Tuned boosting']]
    plot_matrices(main_rows,OUT/'training_confusion_matrices.png',
                  'Training-only outer validation: n=734; inner-selected cutoffs targeting 90% sensitivity')
    write_report(summaries,differences)
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)


def paired_bootstrap(y,predictions,names):
    rng = np.random.default_rng(SEED)
    neg,pos = np.flatnonzero(y==0),np.flatnonzero(y==1)
    samples = [np.concatenate([rng.choice(neg,len(neg),replace=True),rng.choice(pos,len(pos),replace=True)])
               for _ in range(1000)]
    baseline = predictions['Baseline logistic'].to_numpy()
    reference = np.array([roc_auc_score(y[idx],baseline[idx]) for idx in samples])
    rows = []
    for name in names:
        if name=='Baseline logistic':
            continue
        p = predictions[name].to_numpy()
        delta = np.array([roc_auc_score(y[idx],p[idx]) for idx in samples])-reference
        low,high = np.quantile(delta,[.025,.975])
        rows.append({'model':name,'auc_difference_vs_baseline':float(roc_auc_score(y,p)-roc_auc_score(y,baseline)),
                     'conditional_bootstrap_low':float(low),'conditional_bootstrap_high':float(high)})
    return rows


def write_report(rows,differences):
    default = [r for r in rows if r['operating_point']=='default_0.5']
    adjusted = [r for r in rows if r['operating_point']!='default_0.5']
    baseline = next(r for r in default if r['model']=='Baseline logistic')
    sensitive = next(r for r in adjusted if r['model']=='Baseline logistic')
    calibrated = next(r for r in default if r['model']=='Sigmoid-calibrated logistic')
    after = next(r for r in default if r['model']=='After-exercise logistic')
    report = f'''# Training-only experiments and error analysis

All new experiments use the original 734 training records only. The previously inspected test
set is not reused for selecting a model or threshold. The raw file and original results remain intact.

Five outer folds supply held-out predictions. Three inner folds select hyperparameters by ROC-AUC.
Within each outer training partition, inner OOF predictions select the largest cutoff achieving 90%
sensitivity. Outer predictions measure the achieved sensitivity and specificity; 90% is an exploratory
research target, not a clinical operating requirement. Threshold OOF estimates reuse hyperparameters
chosen on that outer training partition, but outer validation labels never select them.
For sigmoid calibration, all calibration fits are inside the outer training partition.
Preprocessing is fit inside each fit, including inner folds. No oversampling is applied.

## Full existing test confusion matrices

![Existing test confusion matrices](test_confusion_matrices.png)

Rows are actual labels; columns are predicted labels. TN and TP are correct predictions;
FP is a false disease alarm; FN is missed disease. These are the original test results, not a new test.

## Nested outer-validation comparisons at 0.5

{markdown_table(default,['model','roc_auc','average_precision','brier','sensitivity','specificity','tn','fp','fn','tp'])}

## Inner-selected cutoffs targeting 90% sensitivity

{markdown_table(adjusted,['model','sensitivity','specificity','precision','tn','fp','fn','tp'])}

![Outer-validation confusion matrices](training_confusion_matrices.png)

## Paired AUC differences versus baseline

{markdown_table(differences,['model','auc_difference_vs_baseline','conditional_bootstrap_low','conditional_bootstrap_high'])}

Intervals use 1,000 paired, class-stratified bootstrap resamples of fixed outer predictions.
They describe uncertainty conditional on those predictions; they do not capture retraining variation,
dependence between CV folds, or multiple-experiment selection. They are descriptive, not a formal
claim of statistical significance. Fold results and selected parameters are saved separately.

## Observed findings

The baseline's pooled outer-validation ROC-AUC is {baseline['roc_auc']:.3f}. Neither tuned tree
model nor tuned logistic exceeds it in this experiment. Sigmoid calibration changes Brier score from
{baseline['brier']:.3f} to {calibrated['brier']:.3f}, so it does not improve this probability-quality measure.
Removing cholesterol or numeric missingness flags also produces no AUC gain here; this does not
demonstrate that their hospital-transfer behavior is worse.

Training-selected sensitivity cutoffs change baseline missed cases from {baseline['fn']} to
{sensitive['fn']} and false positives from {baseline['fp']} to {sensitive['fp']}, achieving
{sensitive['sensitivity']:.1%} sensitivity and {sensitive['specificity']:.1%} specificity.
This is an observable error tradeoff; choosing it requires a stated objective.

The after-exercise model's ROC-AUC is {after['roc_auc']:.3f}, a difference of
{after['roc_auc']-baseline['roc_auc']:.3f} versus baseline, with Brier score {after['brier']:.3f}.
That is evidence for the value of additional test information in a different workflow, not proof
that the initial-assessment model has been improved.

## Interpretation boundaries

Removing cholesterol tests dependence on a feature missing for an entire hospital. Removing numeric
missingness flags tests a possible source shortcut (categorical missingness still remains visible).
Sigmoid calibration tests probability quality, not just class accuracy. These comparisons do not
establish better hospital transfer; the earlier transfer experiment remains the evidence for that.
After-exercise logistic adds thalch, oldpeak, exang and slope, but excludes ca and thal. It answers a
different prediction-time question and cannot be called an improvement to initial-assessment prediction.
None of the models is clinically validated. This small retrospective dataset limits deployment claims.

Reproduce: `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python experiment.py`.
'''
    (OUT/'experiment_report.md').write_text(report)


if __name__=='__main__':
    main()

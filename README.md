# Heart-disease classification: thresholds and hospital shift

A study of existing disease classification across four hospital cohorts, focusing on data quality,
model complexity, and evaluation beyond accuracy.

## Start with the data

The source CSV contains 920 records and 16 columns. The audit identifies two duplicate copies,
invalid zero measurements, and hospital-dependent missingness.

- [Data dictionary, source attribution, and cleaning decisions](docs/data_audit.md)
- [Machine-readable audit](results/data_audit.json)

```sh
python -m pip install -r requirements.txt
python audit_data.py
```

The source file is preserved. The binary prediction target is `num > 0`; no future outcomes are present.

## Baseline and model comparison

Seven initial-assessment features feed logistic regression, random forest, and histogram gradient
boosting. Imputation and encoding are learned inside each training fold; numeric scaling applies
only to logistic regression. The split is stratified by hospital and binary target.

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python train_compare.py
```

| Model | Test ROC-AUC | Sensitivity | Specificity | Brier score |
| --- | ---: | ---: | ---: | ---: |
| Logistic regression | 0.849 | 83.3% | 76.8% | 0.152 |
| Random forest | 0.851 | 81.4% | 74.4% | 0.149 |
| Gradient boosting | 0.843 | 85.3% | 72.0% | 0.154 |

These small differences do not establish a clear winner. Logistic regression had the highest mean
training CV ROC-AUC. Hospital transfer shows a larger limitation: all three score near 0.69 on
held-out VA Long Beach within the training partition.

- [Methodology and evaluation boundaries](docs/methodology.md)
- [Full initial results](results/comparison_report.md)

![ROC, precision–recall, and calibration](results/comparison.png)

# Data audit and source attribution

## Source

The local `heart_disease_uci.csv` contains 920 records and 16 columns from Cleveland, Hungary,
Switzerland, and VA Long Beach. It is a labeled derivative of the UCI Heart Disease data. The exact
transformation history of this particular CSV is unavailable, so it is not presented as an exact copy
of the canonical processed Cleveland dataset.

Definitions and parent-dataset attribution:

Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). *Heart Disease*.
UCI Machine Learning Repository. DOI: [10.24432/C52P4X](https://doi.org/10.24432/C52P4X).
The [UCI repository](https://archive.ics.uci.edu/dataset/45/heart%2Bdisease) lists the parent dataset
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The scripts create derived targets and evaluation artifacts; the source CSV is preserved.

Run `python audit_data.py` for the machine-readable audit, including the source file's SHA-256 hash.

## Column dictionary

| Column | Meaning | Modeling type | Raw blanks |
| --- | --- | --- | ---: |
| id | Sequential record identifier | Exclude from predictors | 0 |
| age | Age in years | Numeric | 0 |
| sex | Recorded Male/Female category | Categorical | 0 |
| dataset | Hospital cohort | Evaluation metadata | 0 |
| cp | Typical angina, atypical angina, non-anginal pain, asymptomatic | Categorical | 0 |
| trestbps | Resting blood pressure on admission, mmHg | Numeric | 59 |
| chol | Serum cholesterol, mg/dL | Numeric | 30 |
| fbs | Fasting blood sugar above 120 mg/dL | Binary categorical | 90 |
| restecg | Resting ECG category | Categorical | 2 |
| thalch | Maximum achieved heart rate; UCI calls this thalach | Numeric | 55 |
| exang | Exercise-induced angina | Binary categorical | 55 |
| oldpeak | Exercise-induced ST depression relative to rest | Numeric | 62 |
| slope | Peak exercise ST-segment slope | Categorical | 309 |
| ca | Number of major vessels visualized by fluoroscopy, 0–3 | Count | 611 |
| thal | Normal, fixed defect, or reversible-defect test result | Categorical | 486 |
| num | Angiographic disease-status code, 0–4 | Target | 0 |

Raw blanks do not include invalid zeros. No category normalization is needed for excluded `thal`;
the CSV's `reversable defect` spelling is retained in the source.

## Findings and cleaning decisions

| Finding | Evidence | Decision |
| --- | --- | --- |
| ID hides duplicated records | Pairs 405/406 and 860/908 match across every other column | Keep 405 and 860; remove copies 406 and 908 before splitting |
| Invalid cholesterol values | 172 zeros: all 123 Switzerland rows and 49 VA rows | Convert zero to missing, giving 202 missing values |
| Invalid blood pressure | ID 754 has trestbps=0 | Convert to missing, giving 60 missing values |
| Complete-case deletion changes the population | Only 299 complete records; 297 from Cleveland | Preserve incomplete records and impute inside each training fold |
| Some test features are mostly unavailable | ca: 66.4% blank; thal: 52.8% blank | Exclude from initial-assessment inputs, also because of prediction timing |
| Signed ST measurements need interpretation | 12 negative oldpeak values | Preserve; do not assume every negative measurement is invalid |
| Extreme cholesterol may be real | Four values above 500 mg/dL | Preserve; no automatic outlier removal |

Removing duplicates treats the two identical pairs as copied records. Unique patient identifiers are
not available to verify whether they were truly separate patients; this assumption is documented.

## Target and representation

The binary target is `num > 0`, indicating existing disease presence. Raw target counts are 411
negative and 509 positive: mild imbalance. No oversampling or class weighting is used.
The original five-class counts are 411, 265, 109, 107, and 28.

| Hospital | Raw records | Disease present | Complete records |
| --- | ---: | ---: | ---: |
| Cleveland | 304 | 45.7% | 297 |
| Hungary | 293 | 36.2% | 1 |
| Switzerland | 123 | 93.5% | 0 |
| VA Long Beach | 200 | 74.5% | 1 |

Hungary has only codes 0 and 1, so multiclass severity interpretation would require label-harmonization
checks. The sample is 78.9% male. Missingness and disease prevalence vary substantially by hospital.
No follow-up outcomes or measurement timestamps exist; future-event prediction is unsupported.

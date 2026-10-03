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

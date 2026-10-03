"""Audit source data without modifying it. Run: python audit_data.py."""
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent


def main():
    source = ROOT / 'heart_disease_uci.csv'
    data = pd.read_csv(source)
    duplicate_mask = data.duplicated(subset=data.columns.drop('id'), keep=False)
    numeric = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
    cohorts = []
    for site, records in data.groupby('dataset'):
        cohorts.append({
            'hospital': site, 'records': len(records),
            'disease_fraction': float((records['num'] > 0).mean()),
            'complete_records': int(records.notna().all(axis=1).sum()),
            'cholesterol_zero_records': int(records['chol'].eq(0).sum()),
            'missing_counts': records.isna().sum().astype(int).to_dict(),
        })
    audit = {
        'source_file': source.name,
        'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'records': len(data), 'columns': list(data.columns),
        'blank_counts': data.isna().sum().astype(int).to_dict(),
        'exact_duplicate_copies_including_id': int(data.duplicated().sum()),
        'duplicate_copies_excluding_id': int(data.drop(columns='id').duplicated().sum()),
        'all_duplicate_ids_excluding_id': data.loc[duplicate_mask, 'id'].astype(int).tolist(),
        'records_with_any_blank': int(data.isna().any(axis=1).sum()),
        'complete_records': int(data.notna().all(axis=1).sum()),
        'target_counts': data['num'].value_counts().sort_index().astype(int).to_dict(),
        'sex_counts': data['sex'].value_counts().astype(int).to_dict(),
        'zero_cholesterol': int(data['chol'].eq(0).sum()),
        'zero_blood_pressure_ids': data.loc[data['trestbps'].eq(0), 'id'].astype(int).tolist(),
        'negative_oldpeak_records': int(data['oldpeak'].lt(0).sum()),
        'numeric_ranges': {c: {'min':float(data[c].min()),'max':float(data[c].max())} for c in numeric},
        'cohorts': cohorts,
    }
    destination = ROOT / 'results' / 'data_audit.json'
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(audit, indent=2) + '\n')
    print(f'{len(data)} records; {len(data.columns)} columns; '
          f'{audit["duplicate_copies_excluding_id"]} duplicate copies excluding ID.')
    print('Audit saved to results/data_audit.json; source file unchanged.')


if __name__ == '__main__':
    main()

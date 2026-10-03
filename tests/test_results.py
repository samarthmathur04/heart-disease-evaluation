"""Verify data provenance, partition boundaries, and reported prediction metrics."""
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score

from train_compare import CATEGORICAL, FEATURES, NUMERIC, pipeline

ROOT = Path(__file__).resolve().parents[1]


class ResultsIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = pd.read_csv(ROOT / 'heart_disease_uci.csv')
        cls.split = pd.read_csv(ROOT / 'results' / 'split.csv')
        cls.test_predictions = pd.read_csv(ROOT / 'results' / 'test_predictions.csv')
        cls.outer_predictions = pd.read_csv(ROOT / 'results' / 'experiments' / 'outer_predictions.csv')

    def test_source_hash_and_audit(self):
        audit = json.loads((ROOT / 'results' / 'data_audit.json').read_text())
        self.assertEqual(audit['sha256'], hashlib.sha256((ROOT / 'heart_disease_uci.csv').read_bytes()).hexdigest())
        self.assertEqual(audit['records'], len(self.raw))
        self.assertEqual(audit['blank_counts'], self.raw.isna().sum().astype(int).to_dict())
        self.assertEqual(audit['zero_cholesterol'], int(self.raw.chol.eq(0).sum()))

    def test_split_preserves_deduplicated_records_and_targets(self):
        clean = self.raw.loc[~self.raw.duplicated(subset=self.raw.columns.drop('id'))]
        self.assertTrue(self.split.id.is_unique)
        self.assertEqual(set(self.split.id), set(clean.id))
        expected = clean.set_index('id')['num'].gt(0).astype(int)
        for row in self.split.itertuples():
            self.assertEqual(row.target, expected.loc[row.id])
        train = set(self.split.loc[self.split.partition == 'train', 'id'])
        test = set(self.split.loc[self.split.partition == 'test', 'id'])
        self.assertFalse(train & test)
        self.assertEqual((len(train), len(test)), (734, 184))

    def test_predictions_obey_partition_boundaries(self):
        train = set(self.split.loc[self.split.partition == 'train', 'id'])
        test = set(self.split.loc[self.split.partition == 'test', 'id'])
        self.assertTrue(self.test_predictions.id.is_unique)
        self.assertTrue(self.outer_predictions.id.is_unique)
        self.assertEqual(set(self.test_predictions.id), test)
        self.assertEqual(set(self.outer_predictions.id), train)
        expected = self.split.set_index('id')['target']
        for table in [self.test_predictions, self.outer_predictions]:
            np.testing.assert_array_equal(table.target, expected.loc[table.id].to_numpy())

    def check_metric_row(self, row, y, probabilities, decisions):
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
        counts = confusion_matrix(y, decisions, labels=[0, 1]).ravel()
        for key, count in zip(['tn', 'fp', 'fn', 'tp'], counts):
            self.assertEqual(int(row[key]), int(count))
        tn, fp, fn, tp = counts
        self.assertAlmostEqual(row['sensitivity'], tp / (tp + fn))
        self.assertAlmostEqual(row['specificity'], tn / (tn + fp))
        self.assertAlmostEqual(row['roc_auc'], roc_auc_score(y, probabilities))
        self.assertAlmostEqual(row['average_precision'], average_precision_score(y, probabilities))
        self.assertAlmostEqual(row['brier'], brier_score_loss(y, probabilities))

    def test_initial_metrics_match_saved_predictions(self):
        rows = pd.read_csv(ROOT / 'results' / 'test_metrics.csv')
        metadata = json.loads((ROOT / 'results' / 'run_metadata.json').read_text())
        for _, row in rows.iterrows():
            if row.model == 'Constant prevalence reference':
                prevalence = self.split.loc[self.split.partition == 'train', 'target'].mean()
                probability = np.full(len(self.test_predictions), prevalence)
            else:
                probability = self.test_predictions[row.model].to_numpy()
                if row.operating_point != 'default_0.5':
                    self.assertAlmostEqual(row.threshold, metadata['thresholds'][row.model]['value'])
            self.check_metric_row(row, self.test_predictions.target, probability, probability >= row.threshold)

    def test_nested_metrics_match_outer_predictions(self):
        rows = pd.read_csv(ROOT / 'results' / 'experiments' / 'nested_summary.csv')
        self.assertEqual(len(rows), 16)
        for _, row in rows.iterrows():
            probability = self.outer_predictions[row.model].to_numpy()
            if row.operating_point == 'default_0.5':
                decisions = probability >= 0.5
            else:
                decisions = self.outer_predictions[row.model + '__decision_90'].to_numpy()
                self.assertTrue(np.isin(decisions, [0, 1]).all())
            self.check_metric_row(row, self.outer_predictions.target, probability, decisions)

    def test_pipeline_handles_missing_numeric_and_unseen_category(self):
        # Simulate a source with no usable cholesterol and a new ECG category.
        # These are realistic transfer edge cases, separate from stored artifacts.
        X = self.raw.iloc[:80][FEATURES].copy()
        X['chol'] = np.nan
        for c in CATEGORICAL:
            X[c] = X[c].map(lambda value: str(value) if pd.notna(value) else np.nan)
        y = self.raw.iloc[:80]['num'].gt(0).astype(int)
        model = pipeline(LogisticRegression(max_iter=2000), scale=True).fit(X, y)
        new = X.iloc[:2].copy()
        new['restecg'] = 'unseen ECG category'
        new[NUMERIC] = np.nan
        probability = model.predict_proba(new)
        self.assertTrue(np.isfinite(probability).all())
        np.testing.assert_allclose(probability.sum(axis=1), 1.0)


if __name__ == '__main__':
    unittest.main()

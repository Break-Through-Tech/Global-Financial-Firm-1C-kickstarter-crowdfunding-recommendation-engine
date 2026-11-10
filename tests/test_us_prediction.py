import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from utils.us_modeling import conservative_pipeline
from utils.us_prediction import USPredictor, campaign_frame, predict_campaign
from utils.us_validation import error_breakdowns, metrics


class USPredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.campaign = dict(goal=5000, duration=30, category='Art',
                            subcategory='Painting', location='Boston, MA')
        cls.temp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.temp.name) / 'model.joblib'
        X = pd.DataFrame([dict(cls.campaign, goal=100 + i * 100,
                              category=['Art', 'Design'][i % 2]) for i in range(40)])
        cls.pipeline = conservative_pipeline(HistGradientBoostingClassifier(
            max_iter=5, min_samples_leaf=3, random_state=42), 'target').fit(X, [0, 1] * 20)
        joblib.dump(cls.pipeline, cls.path)
        cls.path.with_suffix('.metadata.json').write_text(json.dumps({'threshold': 0.5}))
        cls.predictor = USPredictor(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_probability_consistency_and_serialization(self):
        result = self.predictor.predict(self.campaign)
        expected = self.pipeline.predict_proba(campaign_frame(self.campaign))[0, 1]
        self.assertEqual(result['success_probability'], expected)
        self.assertEqual(result['predicted_class'], int(expected >= 0.5))
        self.assertEqual(result['threshold'], 0.5)
        self.assertEqual(predict_campaign(self.campaign, self.path), result)

    def test_feature_order_and_excluded_fields(self):
        reversed_input = dict(reversed(list(self.campaign.items())))
        reversed_input.update({'status': 'failed', 'reward levels': 'invalid', 'url': None})
        self.assertEqual(self.predictor.predict(reversed_input), self.predictor.predict(self.campaign))

    def test_unseen_categories_and_state(self):
        result = self.predictor.predict(dict(self.campaign, category='New category',
                                            subcategory='New subcategory', location='DC'))
        self.assertTrue(0 <= result['success_probability'] <= 1)
        self.assertEqual(campaign_frame(dict(self.campaign, location='boston, ma')).location.iloc[0], 'boston, MA')

    def test_missing_inputs(self):
        for name in self.campaign:
            row = self.campaign.copy()
            row.pop(name)
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.predictor.predict(row)

    def test_invalid_inputs(self):
        for name in ['goal', 'duration']:
            for value in [None, np.nan, np.inf, 0, -1, True, '100', [], 1j]:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    self.predictor.predict(dict(self.campaign, **{name: value}))
        for name in ['category', 'subcategory', 'location']:
            for value in [None, '', ' ', 3]:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    self.predictor.predict(dict(self.campaign, **{name: value}))
        for location in ['London, UK', 'PR', 'California', 'CA garbage', 'Toronto, ON']:
            with self.subTest(location=location), self.assertRaises(ValueError):
                self.predictor.predict(dict(self.campaign, location=location))

    def test_selected_threshold_is_used(self):
        path = Path(self.temp.name) / 'threshold.joblib'
        joblib.dump(self.pipeline, path)
        path.with_suffix('.metadata.json').write_text(json.dumps({'threshold': 0.9}))
        predictor = USPredictor(path)
        result = predictor.predict(self.campaign)
        self.assertEqual(result['threshold'], 0.9)
        self.assertEqual(result['predicted_class'], int(result['success_probability'] >= 0.9))

    def test_missing_model_raises(self):
        with self.assertRaises(FileNotFoundError):
            USPredictor(Path(self.temp.name) / 'missing.joblib')

    def test_confusion_metrics_and_sparse_groups(self):
        y = pd.Series([0, 0, 1, 1])
        probability = np.array([0.1, 0.9, 0.1, 0.9])
        result = metrics(y, probability)
        self.assertEqual([result[k] for k in ['tn', 'fp', 'fn', 'tp']], [1, 1, 1, 1])
        self.assertEqual(result['failure_recall'], 0.5)
        self.assertAlmostEqual(result['brier'], 0.41)
        nested_thresholds = np.array([0.2, 0.95, 0.05, 0.8])
        self.assertEqual(metrics(y, probability, nested_thresholds)['accuracy'], 1.0)
        X = pd.DataFrame([self.campaign] * 4)
        self.assertTrue(error_breakdowns(X, y, probability).empty)

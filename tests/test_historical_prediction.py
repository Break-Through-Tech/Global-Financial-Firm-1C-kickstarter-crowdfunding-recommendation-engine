import hashlib
import json
from pathlib import Path
import platform
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import sklearn

from scripts.prepare_historical_models import historical_pipeline, load_training_rows
from utils.historical_prediction import (INPUTS, MODEL_HASHES, historical_campaign_frame,
                                         load_historical_predictor, location_frame, predict_historical)

try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    AppTest = None

ROOT = Path(__file__).resolve().parents[1]


class HistoricalPredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.campaign = dict(duration=30, category='Music', subcategory='Rock', location='Berlin, Germany')
        cls.folder = tempfile.TemporaryDirectory()
        cls.path = Path(cls.folder.name) / 'historical.joblib'
        rows = pd.DataFrame([dict(cls.campaign, duration=10 + i,
                            location=['Berlin, Germany', 'Austin, TX'][i % 2]) for i in range(40)])
        cls.pipeline = historical_pipeline().fit(location_frame(rows), [0, 1, 1, 0] * 10)
        joblib.dump(cls.pipeline, cls.path)
        cls.digest = hashlib.sha256(cls.path.read_bytes()).hexdigest()
        cls.metadata = dict(artifact_sha256=cls.digest, model_version='mixed_historical_v1',
                            scope='mixed', threshold=.5, input_order=INPUTS, uses_funding_goal=False,
                            validation_metrics={'accuracy': .6}, majority_class_accuracy=.55,
                            country_training_counts={'Germany': 20, 'United States': 20},
                            training_categories={'category': ['Music'], 'subcategory': ['Rock']},
                            environment=dict(python=platform.python_version(), sklearn=sklearn.__version__,
                                             numpy=np.__version__, pandas=pd.__version__, joblib=joblib.__version__))
        cls.path.with_suffix('.metadata.json').write_text(json.dumps(cls.metadata))

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_probability_labels_serialization_and_no_refit(self):
        with patch.dict(MODEL_HASHES, mixed=self.digest):
            predictor = load_historical_predictor(self.path, 'mixed')
        with patch('sklearn.pipeline.Pipeline.fit', side_effect=AssertionError('Unexpected training')):
            result = predict_historical(predictor, self.campaign, 'mixed')
        expected = self.pipeline.predict_proba(historical_campaign_frame(self.campaign, 'mixed'))[0, 1]
        self.assertEqual(result['success_probability'], expected)
        self.assertEqual(result['predicted_class'], int(expected >= .5))
        self.assertEqual(result['classification'], 'successful' if expected >= .5 else 'failed')
        self.assertTrue(any('Limited listed-region support' in warning for warning in result['warnings']))

    def test_money_and_outcomes_cannot_change_features(self):
        expected = historical_campaign_frame(self.campaign, 'international')
        for goal in [None, 1, 999999, 'GBP 1000', np.nan]:
            row = dict(self.campaign, goal=goal, pledged=999999, updates=999, status='successful')
            pd.testing.assert_frame_equal(historical_campaign_frame(row, 'international'), expected)
        self.assertNotIn('goal', INPUTS)

    def test_scope_and_input_validation(self):
        for duration in [True, '30', 0, -1, np.inf, np.nan, None, [], 1j]:
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                historical_campaign_frame(dict(self.campaign, duration=duration), 'mixed')
        for name in ['location', 'category', 'subcategory']:
            for value in ['', None, '  ', 3]:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    historical_campaign_frame(dict(self.campaign, **{name: value}), 'mixed')
        for location in ['Austin, TX', 'CA', 'Toronto, ON', ', Germany']:
            with self.subTest(location=location), self.assertRaises(ValueError):
                historical_campaign_frame(dict(self.campaign, location=location), 'international')
        self.assertEqual(historical_campaign_frame(dict(self.campaign, location='CA'), 'mixed').country_region.iloc[0],
                         'United States')
        with self.assertRaises(ValueError):
            predict_historical((self.pipeline, self.metadata), self.campaign, 'international')

    def test_unknown_categories_and_regions_have_warnings(self):
        row = dict(self.campaign, location='Lima, Peru', category='Unseen', subcategory='Unseen')
        result = predict_historical((self.pipeline, self.metadata), row, 'mixed')
        self.assertTrue(0 <= result['success_probability'] <= 1)
        self.assertTrue(any('0 training campaigns for Peru' in warning for warning in result['warnings']))
        self.assertTrue(any('Unseen category' in warning for warning in result['warnings']))

    def test_missing_and_changed_artifacts_fail_before_loading(self):
        with self.assertRaises(FileNotFoundError):
            load_historical_predictor(Path(self.folder.name) / 'missing.joblib', 'mixed')
        with patch.dict(MODEL_HASHES, mixed='0' * 64), patch('joblib.load') as load:
            with self.assertRaisesRegex(ValueError, 'integrity'):
                load_historical_predictor(self.path, 'mixed')
            load.assert_not_called()

    def test_official_training_loader_preserves_membership_without_reserved_files(self):
        root = Path('/synthetic')
        ids = pd.DataFrame({'project id': np.arange(1427)[::-1], 'target': np.arange(1427) % 2})
        rows = pd.DataFrame({'project id': ids['project id'],
                             'status': np.where(ids.target.eq(1), 'successful', 'failed'),
                             'location': 'Berlin, Germany'})
        folder = root / 'data/Processed-Data'
        files = {folder / 'train_project_ids.csv': ids,
                 folder / 'international_only/train_project_ids.csv': ids,
                 folder / 'international_only/DSI_kickstarterscrape_dataset_cleaned.csv': rows.iloc[::-1]}
        with patch('scripts.prepare_historical_models.pd.read_csv', side_effect=lambda path: files[path].copy()):
            actual, targets, membership = load_training_rows(root, 'international')
            self.assertEqual(actual['project id'].tolist(), ids['project id'].tolist())
            self.assertEqual(membership.tolist(), ids['project id'].tolist())
            self.assertEqual(targets.tolist(), ids.target.tolist())


@unittest.skipUnless(AppTest is not None and all(
    (ROOT / f'models/{scope}_historical_v1.joblib').exists()
    and (ROOT / f'models/{scope}_historical_v1.metadata.json').exists() for scope in MODEL_HASHES),
    'Streamlit and historical artifacts required')
class HistoricalAppTests(unittest.TestCase):
    def test_real_modes_match_direct_predictors_without_training(self):
        import streamlit as st
        st.cache_resource.clear()
        app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=30)
        with patch('utils.us_prediction.load_primary_predictor', side_effect=AssertionError('Wrong model')), \
                patch('sklearn.pipeline.Pipeline.fit', side_effect=AssertionError('Training')):
            for scope, mode in [('international', 'International (experimental)'), ('mixed', 'Mixed (experimental)')]:
                app.radio[0].set_value(mode).run(timeout=30)
                for i, value in [(1, 'Music'), (2, 'Rock'), (3, 'Berlin, Germany')]:
                    app.text_input[i].set_value(value)
                app.button[0].click().run(timeout=30)
                self.assertEqual(list(app.exception), [])
                self.assertEqual(list(app.error), [])
                predictor = load_historical_predictor(ROOT / f'models/{scope}_historical_v1.joblib', scope)
                expected = predict_historical(predictor, dict(duration=30, category='Music',
                                              subcategory='Rock', location='Berlin, Germany'), scope)
                values = {metric.label: metric.value for metric in app.metric}
                self.assertEqual(values['Probability of Success'], f"{expected['success_probability']:.2%}")
                self.assertEqual(values['Predicted Outcome'], expected['classification'].capitalize())
        st.cache_resource.clear()

    def test_missing_model_displays_unavailable_without_prediction(self):
        import streamlit as st
        st.cache_resource.clear()
        with patch('utils.historical_prediction.load_historical_predictor', side_effect=FileNotFoundError):
            app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=30)
            app.radio[0].set_value('International (experimental)').run(timeout=30)
            for i, value in [(1, 'Art'), (2, 'Painting'), (3, 'Paris, France')]:
                app.text_input[i].set_value(value)
            app.button[0].click().run(timeout=30)
            self.assertTrue(list(app.error))
            self.assertFalse(any(metric.label == 'Probability of Success' for metric in app.metric))
        st.cache_resource.clear()

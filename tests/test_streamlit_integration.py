"""Public app checks without any processed campaign datasets."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from utils.us_prediction import PRIMARY_SHA256, USPredictor, campaign_frame, load_primary_predictor

try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    AppTest = None

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'models/us_primary_model_v1.joblib'


class ArtifactContractTests(unittest.TestCase):
    def test_missing_artifact(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                load_primary_predictor(Path(folder) / 'missing.joblib')

    def test_hash_checked_before_deserialization(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'bad.joblib'
            path.write_bytes(b'not trusted pickle')
            path.with_suffix('.metadata.json').write_text('{}')
            with patch('utils.us_prediction.joblib.load') as load:
                with self.assertRaisesRegex(ValueError, 'integrity'):
                    load_primary_predictor(path)
                load.assert_not_called()

    @unittest.skipUnless(MODEL.exists(), 'Frozen artifact not supplied')
    def test_wrong_metadata_checked_before_deserialization(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.joblib'
            path.write_bytes(MODEL.read_bytes())
            path.with_suffix('.metadata.json').write_text(json.dumps({
                'artifact_sha256': PRIMARY_SHA256, 'model_version': 'v2'}))
            with patch('utils.us_prediction.joblib.load') as load:
                with self.assertRaisesRegex(ValueError, 'metadata'):
                    load_primary_predictor(path)
                load.assert_not_called()

    def test_empty_city_rejected(self):
        for location in [', CA', ' , TX']:
            with self.subTest(location=location), self.assertRaises(ValueError):
                campaign_frame(dict(goal=5000, duration=30, category='Music',
                                    subcategory='Rock', location=location))

    def test_reversed_classes_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.joblib'
            path.with_suffix('.metadata.json').write_text('{"threshold": 0.5}')
            with patch('utils.us_prediction.joblib.load') as load:
                load.return_value.classes_ = [1, 0]
                with self.assertRaisesRegex(ValueError, 'class order'):
                    USPredictor(path)

    @unittest.skipUnless(MODEL.exists(), 'Frozen artifact not supplied')
    def test_missing_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.joblib'
            path.write_bytes(MODEL.read_bytes())
            with self.assertRaises(FileNotFoundError):
                load_primary_predictor(path)

    @unittest.skipUnless(MODEL.exists() and MODEL.with_suffix('.metadata.json').exists(),
                         'Frozen artifact and metadata not supplied')
    def test_python_312_patch_versions_allowed(self):
        for version in ['3.12.3', '3.12.12']:
            with self.subTest(version=version), patch(
                    'utils.us_prediction.platform.python_version', return_value=version):
                predictor = load_primary_predictor(MODEL)
                self.assertEqual(predictor.threshold, .5)

    @unittest.skipUnless(MODEL.exists() and MODEL.with_suffix('.metadata.json').exists(),
                         'Frozen artifact and metadata not supplied')
    def test_python_error_shows_actual_and_required_before_loading(self):
        with patch('utils.us_prediction.platform.python_version', return_value='3.13.9'), \
                patch('utils.us_prediction.joblib.load') as load:
            with self.assertRaisesRegex(ValueError, 'running Python 3.13.9;.*requires Python 3.12'):
                load_primary_predictor(MODEL)
            load.assert_not_called()

    @unittest.skipUnless(MODEL.exists() and MODEL.with_suffix('.metadata.json').exists(),
                         'Frozen artifact and metadata not supplied')
    def test_library_error_shows_versions(self):
        with patch('utils.us_prediction.sklearn.__version__', '0.0'), \
                patch('utils.us_prediction.joblib.load') as load:
            with self.assertRaisesRegex(ValueError, 'server has sklearn 0.0;.*requires 1.9.1'):
                load_primary_predictor(MODEL)
            load.assert_not_called()


@unittest.skipUnless(AppTest is not None, 'Streamlit required')
class HistoricalViewTests(unittest.TestCase):
    def test_experimental_forms_do_not_load_or_train_before_submit(self):
        with patch('utils.us_prediction.load_primary_predictor') as us_load, \
                patch('utils.historical_prediction.load_historical_predictor') as historical_load, \
                patch('sklearn.pipeline.Pipeline.fit') as train:
            app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=30)
            for mode in ['International (experimental)', 'Mixed (experimental)']:
                app.radio[0].set_value(mode).run(timeout=30)
                self.assertEqual(list(app.exception), [])
                self.assertEqual([item.label for item in app.number_input], ['Campaign Duration (days)'])
                self.assertEqual(len(app.text_input), 4)
                self.assertEqual(len(app.button), 1)
                self.assertEqual(len(app.metric), 0)
                self.assertTrue(any('Funding amounts are excluded' in item.value for item in app.warning))
                self.assertFalse(any('Zasha' in item.value for item in app.markdown))
            us_load.assert_not_called()
            historical_load.assert_not_called()
            train.assert_not_called()
            app.radio[0].set_value('U.S. prediction').run(timeout=30)
            self.assertEqual(len(app.number_input), 2)
            self.assertEqual(len(app.metric), 0)


@unittest.skipUnless(AppTest is not None and MODEL.exists()
                     and MODEL.with_suffix('.metadata.json').exists(),
                     'Streamlit and frozen model/metadata required')
class StreamlitTests(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=30)
        self.assertEqual(list(self.app.exception), [])

    def submit(self, category='Music', subcategory='Rock', location='Austin, TX', goal=5000):
        self.app.text_input[1].set_value(category)
        self.app.text_input[2].set_value(subcategory)
        self.app.text_input[3].set_value(location)
        self.app.number_input[0].set_value(float(goal))
        return self.app.button[0].click().run(timeout=30)

    def test_examples_match_direct_without_training(self):
        predictor = load_primary_predictor(MODEL)
        with patch('sklearn.pipeline.Pipeline.fit', side_effect=AssertionError('Training during inference')), \
                patch('utils.preprocessing.FeatureBuilder.fit', side_effect=AssertionError('Feature refit')), \
                patch('utils.preprocessing.TargetEncoder.fit', side_effect=AssertionError('Encoder refit')):
            for category, subcategory, location, goal in [
                ('Music', 'Rock', 'Austin, TX', 5000),
                ('Art', 'Painting', 'CA', 12000),
                ('Music', 'Unseen test subcategory', 'Boston, ma', 3000),
            ]:
                with self.subTest(category=category, subcategory=subcategory):
                    app = self.submit(category, subcategory, location, goal)
                    self.assertEqual(list(app.exception), [])
                    self.assertEqual(list(app.error), [])
                    result = predictor.predict(dict(goal=goal, duration=30, category=category,
                                                    subcategory=subcategory, location=location))
                    values = {m.label: m.value for m in app.metric}
                    self.assertEqual(values['Probability of Success'], f"{result['success_probability']:.2%}")
                    self.assertEqual(values['Predicted Outcome'], result['classification'].capitalize())
                    self.assertEqual(result['predicted_class'], int(result['success_probability'] >= .5))

    def test_missing_and_invalid_location(self):
        for location in ['', 'Toronto, ON']:
            app = self.submit(location=location)
            self.assertTrue(list(app.error))
            self.assertFalse(any(m.label == 'Probability of Success' for m in app.metric))

    def test_invalid_goal(self):
        app = self.submit(goal=0)
        # Streamlit 1.67 rejects out-of-bounds edits and retains the previous value.
        self.assertGreaterEqual(app.number_input[0].value, 1)
        with self.assertRaises(ValueError):
            campaign_frame(dict(goal=0, duration=30, category='Music',
                                subcategory='Rock', location='TX'))

    def test_missing_model_message(self):
        with patch('utils.us_prediction.load_primary_predictor', side_effect=FileNotFoundError):
            # Clear the process-wide Streamlit resource cache before exercising a new load.
            import streamlit as st
            st.cache_resource.clear()
            app = self.submit()
            self.assertTrue(any('could not be found' in e.value for e in app.error))
            self.assertFalse(any(m.label == 'Probability of Success' for m in app.metric))
        st.cache_resource.clear()

    def test_cached_loader(self):
        import streamlit as st
        st.cache_resource.clear()
        with patch('utils.us_prediction.load_primary_predictor', wraps=load_primary_predictor) as load:
            self.submit()
            self.submit(goal=6000)
            self.assertEqual(load.call_count, 1)
        st.cache_resource.clear()

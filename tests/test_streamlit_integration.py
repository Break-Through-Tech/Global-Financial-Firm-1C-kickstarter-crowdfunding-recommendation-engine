"""Public app checks without any processed campaign datasets."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from models.train_and_predict import load_model, predict_campaign
from utils.us_prediction import PRIMARY_SHA256, USPredictor, campaign_frame, load_primary_predictor

try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    AppTest = None

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'models/us_primary_model_v1.joblib'
ONLINE_MODEL = ROOT / 'models/trained_models/online_us_catboostclassifier.pkl'


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
        with patch('models.train_and_predict.load_model') as us_load, \
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


class FakeChunk:
    def __init__(self, text):
        self.text = text


class FakeLlm:
    def __init__(self, reply='**Bottom line** - stub interpretation.', error=None):
        self.reply = reply
        self.error = error
        self.received_messages = []

    def stream(self, messages):
        self.received_messages.append(messages)
        if self.error:
            raise self.error
        yield FakeChunk(self.reply)


@unittest.skipUnless(AppTest is not None and ONLINE_MODEL.exists(),
                     'Streamlit and trained CatBoost model required')
class StreamlitTests(unittest.TestCase):
    def setUp(self):
        import streamlit as st
        st.cache_resource.clear()
        self.addCleanup(st.cache_resource.clear)

    def open_app(self, api_key=''):
        env_file = patch('dotenv.dotenv_values', return_value={'API_KEY': api_key})
        environment = patch.dict(os.environ, {'API_KEY': ''})
        env_file.start()
        environment.start()
        self.addCleanup(env_file.stop)
        self.addCleanup(environment.stop)

        app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=60)
        self.assertEqual(list(app.exception), [])
        return app

    def submit(self, app, name='Indie Album', category='Music', subcategory='Rock',
               location='Austin, TX', goal=5000, rewards='$10,$25,$100,$500'):
        app.text_input[0].set_value(name)
        app.text_input[1].set_value(category)
        app.text_input[2].set_value(subcategory)
        app.text_input[3].set_value(location)
        app.text_input[4].set_value(rewards)
        app.number_input[0].set_value(float(goal))
        return app.button[0].click().run(timeout=120)

    def campaign_for(self, app, name, category, subcategory, location, goal, rewards):
        return {'name': name, 'category': category, 'subcategory': subcategory,
                'location': location, 'goal': float(goal), 'duration': 30.0,
                'reward levels': rewards, 'launch': app.date_input[0].value.isoformat()}

    def test_us_form_has_catboost_inputs(self):
        app = self.open_app()
        self.assertEqual([item.label for item in app.text_input],
                         ['Campaign Name', 'Category', 'Subcategory', 'Location', 'Reward Levels'])
        self.assertEqual([item.label for item in app.date_input], ['Launch Date'])
        self.assertTrue(any('70.4% accuracy' in item.value for item in app.info))

    def test_examples_match_direct_without_training(self):
        pipeline = load_model(ONLINE_MODEL)
        with patch('models.train_and_predict.train_online_model', side_effect=AssertionError('Training during inference')):
            for example in [
                ('Indie Album', 'Music', 'Rock', 'Austin, TX', 5000, '$10,$25,$100,$500'),
                ('Canvas Series', 'Art', 'Painting', 'CA', 12000, '$5,$50'),
                ('Odd One', 'Music', 'Unseen test subcategory', 'Boston, ma', 3000, ''),
            ]:
                with self.subTest(example=example):
                    app = self.submit(self.open_app(), *example)
                    self.assertEqual(list(app.exception), [])
                    self.assertEqual(list(app.error), [])
                    expected = float(predict_campaign(pipeline, self.campaign_for(app, *example))[0])
                    values = {metric.label: metric.value for metric in app.metric}
                    self.assertEqual(values['Probability of Success'], f'{expected:.2%}')
                    self.assertEqual(values['Predicted Outcome'],
                                     'Successful' if expected >= .5 else 'Failed')

    def test_driver_table_shown_after_prediction(self):
        app = self.submit(self.open_app())
        drivers = app.dataframe[0].value
        self.assertEqual(list(drivers.columns),
                         ['Feature', 'Your campaign', 'Push', 'Overall importance (%)'])
        self.assertEqual(len(drivers), 8)
        self.assertTrue((drivers['Push'].abs().diff().dropna() <= 0).all())
        self.assertIn('4 tiers', set(drivers['Your campaign']))

    def test_missing_fields_rejected(self):
        app = self.submit(self.open_app(), location='')
        self.assertTrue(list(app.error))
        self.assertFalse(any(metric.label == 'Probability of Success' for metric in app.metric))

    def test_invalid_goal_keeps_previous_value(self):
        app = self.submit(self.open_app(), goal=0)
        self.assertGreaterEqual(app.number_input[0].value, 1)

    def test_unloadable_model_is_retrained_and_saved(self):
        pipeline = load_model(ONLINE_MODEL)
        with patch('models.train_and_predict.load_model', side_effect=AttributeError('old sklearn pickle')), \
                patch('models.train_and_predict.train_online_model', return_value=pipeline) as train, \
                patch('models.train_and_predict.save_model') as save:
            app = self.submit(self.open_app())
            train.assert_called_once_with(scope='us')
            save.assert_called_once_with(pipeline, ONLINE_MODEL)
        self.assertTrue(any('retraining' in item.value for item in app.warning))
        self.assertTrue(any(metric.label == 'Probability of Success' for metric in app.metric))

    def test_missing_model_and_training_data_message(self):
        with patch('models.train_and_predict.load_model', side_effect=FileNotFoundError), \
                patch('models.train_and_predict.train_online_model', side_effect=FileNotFoundError):
            app = self.submit(self.open_app())
        self.assertTrue(any('could not be found' in item.value for item in app.error))
        self.assertFalse(any(metric.label == 'Probability of Success' for metric in app.metric))

    def test_cached_loader(self):
        with patch('models.train_and_predict.load_model', wraps=load_model) as load:
            app = self.submit(self.open_app())
            self.submit(app, goal=6000)
            self.assertEqual(load.call_count, 1)

    def test_interpretation_hint_without_api_key(self):
        with patch('utils.campaign_interpretation.init_chat_model') as init_chat_model:
            app = self.submit(self.open_app(api_key=''))
            init_chat_model.assert_not_called()
        self.assertTrue(any('API_KEY' in item.value for item in app.info))

    def test_interpretation_gets_prediction_and_drivers(self):
        fake_llm = FakeLlm()
        with patch('utils.campaign_interpretation.init_chat_model', return_value=fake_llm) as init_chat_model:
            app = self.submit(self.open_app(api_key='test-key'))
        self.assertEqual(init_chat_model.call_args.kwargs['api_key'], 'test-key')
        self.assertEqual(init_chat_model.call_args.kwargs['model_provider'], 'google_genai')

        system_prompt, user_prompt = (text for _, text in fake_llm.received_messages[0])
        probability = next(metric.value for metric in app.metric if metric.label == 'Probability of Success')
        self.assertIn('Kickstarter campaign coach', system_prompt)
        self.assertIn('- Reward levels: $10,$25,$100,$500', user_prompt)
        self.assertIn(f'{float(probability.rstrip("%")):.1f}%', user_prompt)
        self.assertIn('Number of reward tiers = 4 tiers: push', user_prompt)
        self.assertIn('Features the model relies on most across all campaigns:\n1. Funding goal', user_prompt)
        self.assertTrue(any('Written by' in item.value for item in app.caption))

    def test_interpretation_failure_does_not_hide_prediction(self):
        failing_llm = FakeLlm(error=RuntimeError('quota exceeded'))
        with patch('utils.campaign_interpretation.init_chat_model', return_value=failing_llm), \
                patch('time.sleep'):
            app = self.submit(self.open_app(api_key='test-key'))
        self.assertEqual(len(failing_llm.received_messages), 3)
        self.assertEqual(list(app.error), [])
        self.assertTrue(any('quota exceeded' in item.value for item in app.warning))
        self.assertTrue(any(metric.label == 'Probability of Success' for metric in app.metric))

    def test_interpretation_retries_after_a_busy_model(self):
        class BusyOnceLlm(FakeLlm):
            def stream(self, messages):
                self.received_messages.append(messages)
                if len(self.received_messages) == 1:
                    yield FakeChunk('half an ans')
                    raise RuntimeError('503 UNAVAILABLE')
                yield FakeChunk('**Bottom line** - full answer.')

        busy_llm = BusyOnceLlm()
        with patch('utils.campaign_interpretation.init_chat_model', return_value=busy_llm), \
                patch('time.sleep'):
            app = self.submit(self.open_app(api_key='test-key'))
        self.assertEqual(len(busy_llm.received_messages), 2)
        self.assertFalse(any('unavailable' in item.value for item in app.warning))
        self.assertTrue(any('Written by' in item.value for item in app.caption))

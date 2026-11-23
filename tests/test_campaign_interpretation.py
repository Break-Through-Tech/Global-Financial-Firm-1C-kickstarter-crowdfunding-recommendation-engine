import math
from pathlib import Path
import unittest

import pandas as pd
from catboost import Pool

from models.train_and_predict import load_model, predict_campaign
from utils.campaign_interpretation import (FEATURE_LABELS, explain_prediction, interpretation_messages,
                                           source_feature, stream_interpretation)

ROOT = Path(__file__).resolve().parents[1]
ONLINE_MODEL = ROOT / 'models/trained_models/online_us_catboostclassifier.pkl'

EXAMPLE_CAMPAIGN = {
    'name': 'Indie Album', 'category': 'Music', 'subcategory': 'Rock', 'location': 'Austin, TX',
    'goal': 5000.0, 'duration': 30.0, 'reward levels': '$10,$25,$100,$500', 'launch': '2026-03-01',
}


class SourceFeatureTests(unittest.TestCase):
    def test_one_hot_and_month_columns_map_back(self):
        self.assertEqual(source_feature('category_Music'), 'category')
        self.assertEqual(source_feature('subcategory_Rock'), 'subcategory')
        self.assertEqual(source_feature('launch_month_sin'), 'launch_month')
        self.assertEqual(source_feature('launch_month_cos'), 'launch_month')
        self.assertEqual(source_feature('log_goal'), 'log_goal')


@unittest.skipUnless(ONLINE_MODEL.exists(), 'Trained CatBoost model required')
class ExplainPredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = load_model(ONLINE_MODEL)
        cls.drivers = explain_prediction(cls.pipeline, EXAMPLE_CAMPAIGN)

    def test_every_model_feature_has_one_readable_row(self):
        self.assertEqual(set(self.drivers['feature']), set(FEATURE_LABELS))
        self.assertFalse(self.drivers['label'].isin(self.drivers['feature']).any())

    def test_drivers_sorted_by_strength(self):
        strengths = self.drivers['push'].abs().tolist()
        self.assertEqual(strengths, sorted(strengths, reverse=True))

    def test_pushes_add_up_to_the_prediction(self):
        catboost_model = self.pipeline.named_steps['model']
        engineered = self.pipeline.named_steps['features'].transform(pd.DataFrame([EXAMPLE_CAMPAIGN]))
        model_input = self.pipeline.named_steps['preprocess'].transform(engineered)
        baseline = catboost_model.get_feature_importance(Pool(model_input), type='ShapValues')[0][-1]

        probability = float(predict_campaign(self.pipeline, EXAMPLE_CAMPAIGN)[0])
        log_odds = math.log(probability / (1 - probability))
        self.assertAlmostEqual(baseline + self.drivers['push'].sum(), log_odds, places=4)

    def test_overall_importance_is_a_percentage_split(self):
        self.assertAlmostEqual(self.drivers['overall_importance'].sum(), 100, places=3)

    def test_values_are_described_in_plain_terms(self):
        values = dict(zip(self.drivers['feature'], self.drivers['value']))
        self.assertEqual(values['log_goal'], '$5,000')
        self.assertEqual(values['n_tiers'], '4 tiers')
        self.assertEqual(values['log_min_tier'], '$10')
        self.assertEqual(values['launch_month'], 'March')
        self.assertEqual(values['category'], 'Music')

    def test_missing_rewards_are_not_provided(self):
        no_rewards = dict(EXAMPLE_CAMPAIGN, **{'reward levels': ''})
        drivers = explain_prediction(self.pipeline, no_rewards)
        values = dict(zip(drivers['feature'], drivers['value']))
        self.assertEqual(values['log_min_tier'], 'not provided')
        self.assertEqual(values['n_tiers'], '0 tiers')


class InterpretationMessageTests(unittest.TestCase):
    def test_messages_without_drivers_skip_blank_inputs(self):
        campaign = dict(EXAMPLE_CAMPAIGN, goal=None, launch=None, **{'reward levels': ''})
        (system_role, _), (human_role, user_prompt) = interpretation_messages(
            campaign, .753, .5, scope_note='Experimental international model.')
        self.assertEqual((system_role, human_role), ('system', 'human'))
        self.assertIn('75.3%', user_prompt)
        self.assertIn('likely to succeed', user_prompt)
        self.assertIn('drivers are not available', user_prompt)
        self.assertIn('Note: Experimental international model.', user_prompt)
        for skipped in ['Funding goal', 'Launch date', 'Reward levels']:
            self.assertNotIn(skipped, user_prompt)

    def test_stream_yields_plain_text(self):
        class Chunk:
            def __init__(self, text):
                self.text = text

        class Llm:
            def stream(self, messages):
                yield from [Chunk('Bottom '), Chunk('line')]

        self.assertEqual(''.join(stream_interpretation(Llm(), [])), 'Bottom line')


if __name__ == '__main__':
    unittest.main()

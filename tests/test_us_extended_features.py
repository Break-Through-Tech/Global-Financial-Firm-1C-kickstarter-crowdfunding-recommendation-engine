import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from utils.us_extended_features import ExtendedFeatures, eligible, experiment_pipeline
from utils.us_validation import candidates


class ExtendedFeatureTests(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame({
            'goal': [100, 1000, 2000, 500] * 5,
            'duration': [30.5, 20, 40, 60] * 5,
            'category': ['Art'] * 20, 'subcategory': ['Painting'] * 20,
            'location': ['Town, CA', 'Town, CA', 'Other, NY', 'Other, NY'] * 5,
            'name': ['ABC 2!', 'Hello world', '', None] * 5,
            'reward levels': ['$25,$1,000', None, '$0,$-5,$nan', '$2.50'] * 5,
        })
        self.y = pd.Series([0, 1] * 10)

    def test_baseline_matches_original_and_serialization(self):
        old = candidates()['HGB tuned'].fit(self.X, self.y)
        new = experiment_pipeline().fit(self.X, self.y)
        np.testing.assert_array_equal(old.predict_proba(self.X), new.predict_proba(self.X))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'experimental.joblib'
            joblib.dump(new, path)
            np.testing.assert_array_equal(new.predict_proba(self.X), joblib.load(path).predict_proba(self.X))

    def test_reward_units_empty_and_title_definitions(self):
        groups = ('reward_count', 'reward_amounts', 'reward_relative', 'title_length', 'title_style')
        out = clone(ExtendedFeatures(groups)).fit_transform(self.X)
        self.assertEqual(out.loc[0, 'reward_count'], 2)
        self.assertAlmostEqual(out.loc[0, 'reward_log_max'], np.log1p(1000))
        self.assertAlmostEqual(out.loc[0, 'reward_relative_min'], np.log1p(25)-np.log1p(100))
        self.assertTrue(np.isnan(out.loc[1, 'reward_log_min']))
        self.assertEqual(out.loc[2, 'reward_count'], 0)
        self.assertEqual(out.loc[0, 'title_characters'], 6)
        self.assertEqual(out.loc[0, 'title_words'], 2)
        self.assertEqual(out.loc[0, 'title_upper_ratio'], 1)
        self.assertEqual(out.loc[0, 'title_numbers'], 1)
        self.assertEqual(out.loc[0, 'title_punctuation'], 1)
        self.assertEqual(out.loc[3, 'title_characters'], 0)

    def test_frequency_and_medians_do_not_refit_on_validation(self):
        builder = ExtendedFeatures(('city_frequency',)).fit(self.X)
        medians = builder.base_.cat_median_.copy()
        validation = self.X.iloc[:1].copy()
        validation['location'] = 'Unseen, CA'
        validation['goal'] = 999999
        self.assertEqual(builder.transform(validation).iloc[0]['log_city_frequency'], 0)
        pd.testing.assert_series_equal(medians, builder.base_.cat_median_)
        self.assertEqual(builder.city_counts_.sum(), len(self.X))

    def test_outcomes_and_unused_sources_are_inert(self):
        builder = ExtendedFeatures(('city_frequency',)).fit(self.X)
        changed = self.X.assign(updates=999, pledged=999, backers=999, name='changed', **{'reward levels': 'bad'})
        pd.testing.assert_frame_equal(builder.transform(self.X), builder.transform(changed))

    def test_guardrails(self):
        baseline = dict(accuracy=.66, roc_auc=.72, failure_recall=.56)
        self.assertTrue(eligible(dict(accuracy=.664, roc_auc=.721, failure_recall=.559), baseline))
        self.assertFalse(eligible(dict(accuracy=.662, roc_auc=.73, failure_recall=.56), baseline))
        self.assertFalse(eligible(dict(accuracy=.68, roc_auc=.73, failure_recall=.54), baseline))


if __name__ == '__main__':
    unittest.main()

"""Prediction input contract, fitted-state isolation and serialization checks."""
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from utils.features import add_base_columns, ask_features, reward_features, effort_flags, timing_features
from utils.preprocessing import FeatureBuilder, make_pipeline

POLICY = dict(include_timing=False, include_flags=False, include_reward=False,
              exclude_renamed=True)


class ConservativeInputTests(unittest.TestCase):
    @staticmethod
    def campaigns():
        return pd.DataFrame({
            'goal': [500., 1000., 2000., 4000., 8000., 16000.],
            'duration': [30., 30.5, 45., 60., 31., 40.],
            'category': ['Art'] * 3 + ['Design'] * 3,
            'subcategory': ['Painting'] * 3 + ['Product Design'] * 3,
            'location': ['San Francisco, CA'] * 3 + ['Boston, MA'] * 3,
        })

    def test_minimal_prediction_interface_without_scraped_fields(self):
        x = self.campaigns()
        pipeline = make_pipeline(LogisticRegression(), categorical_strategy='onehot', **POLICY)
        pipeline.fit(x, [0, 1, 0, 1, 0, 1])
        proposed = pd.DataFrame([dict(goal=1500., duration=30.5, category='Art',
                                     subcategory='Painting', location='San Francisco, CA')])
        self.assertTrue(np.isfinite(pipeline.predict_proba(proposed)).all())
        self.assertEqual(set(pipeline.named_steps['features'].transform(proposed)),
                         {'log_goal', 'goal_vs_subcat', 'goal_vs_cat', 'goal_per_day',
                          'duration', 'category', 'subcategory', 'state'})

    def test_excluded_field_values_cannot_change_features(self):
        x = self.campaigns()
        builder = FeatureBuilder(**POLICY).fit(x)
        sentinel = x.assign(**{c: 'invalid / outcome sentinel' for c in
            ['reward levels', 'url', 'name', 'funded date', 'levels', 'renamed',
             'pledged', 'backers', 'funded percentage', 'updates', 'comments', 'target', 'status']})
        pd.testing.assert_frame_equal(builder.transform(x), builder.transform(sentinel))

    def test_missing_core_columns_fail_in_fit_and_transform(self):
        x = self.campaigns()
        fitted = FeatureBuilder(**POLICY).fit(x)
        for column in x:
            with self.subTest(column=column):
                for operation in (FeatureBuilder(**POLICY).fit, fitted.transform):
                    with self.assertRaisesRegex(ValueError, column):
                        operation(x.drop(columns=column))

    def test_enabled_families_validate_their_inputs(self):
        for option, columns in [('include_reward', ['reward levels']),
                                ('include_flags', ['name', 'url']),
                                ('include_timing', ['funded date'])]:
            policy = {**POLICY, option: True}
            for column in columns:
                with self.subTest(option=option, column=column):
                    x = self.campaigns().assign(**{c: 'placeholder' for c in columns if c != column})
                    with self.assertRaisesRegex(ValueError, column):
                        FeatureBuilder(**policy).fit(x)

    def test_transform_keeps_training_medians_and_scaler(self):
        x = self.campaigns()
        pipeline = make_pipeline(LogisticRegression(), categorical_strategy='onehot', **POLICY)
        pipeline.fit(x, [0, 1, 0, 1, 0, 1])
        builder = pipeline.named_steps['features']
        before = builder.subcat_median_.copy()
        scaler = pipeline.named_steps['prep'].named_transformers_['num'].named_steps['scale']
        means = scaler.mean_.copy()
        row = x.iloc[[0]].assign(goal=1e12, category='Unseen', subcategory='Unseen')
        transformed = builder.transform(row)
        expected = np.log1p(1e12) - before.median()
        self.assertAlmostEqual(transformed.goal_vs_subcat.iloc[0], expected)
        pipeline.predict_proba(row)
        pd.testing.assert_series_equal(before, builder.subcat_median_)
        np.testing.assert_array_equal(means, scaler.mean_)

    def test_defaults_match_legacy_feature_composition(self):
        x = self.campaigns().assign(**{'reward levels': '$5,$25,$100',
            'url': 'https://www.kickstarter.com/projects/creator/original-title',
            'name': 'Original title', 'funded date': '2012-06-25'})
        base = add_base_columns(x)
        ask, _, _ = ask_features(base)
        expected = pd.concat([ask, reward_features(base), effort_flags(base), timing_features(base)], axis=1)
        expected = expected.loc[:, ~expected.columns.duplicated()]
        for c in ['category', 'subcategory', 'state']:
            expected[c] = base[c].astype('object')
        pd.testing.assert_frame_equal(FeatureBuilder().fit_transform(x), expected)

    def test_current_environment_joblib_roundtrip(self):
        x = self.campaigns()
        pipeline = make_pipeline(LogisticRegression(), categorical_strategy='onehot', **POLICY)
        pipeline.fit(x, [0, 1, 0, 1, 0, 1])
        with tempfile.TemporaryDirectory() as folder:
            artifact = Path(folder) / 'conservative.joblib'
            joblib.dump(pipeline, artifact)
            restored = joblib.load(artifact)
            np.testing.assert_allclose(restored.predict_proba(x), pipeline.predict_proba(x))

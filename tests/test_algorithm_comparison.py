import importlib.util
import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone

from utils.us_algorithm_comparison import NativeCategories, comparison_pipeline, choose_threshold
from utils.us_extended_features import experiment_pipeline
from utils.us_recommendation_research import benchmarks, nearest_campaigns, scenario_comparison


class AlgorithmComparisonTests(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame({
            'goal': [100, 1000, 2000, 500]*10, 'duration': [30.5, 20, 40, 60]*10,
            'category': ['Art']*40, 'subcategory': ['Painting']*40,
            'location': ['Town, CA']*40, 'reward levels': ['$25,$100']*40})
        self.y = pd.Series([0, 1]*20)

    def test_native_categoricals_have_no_target_encoding(self):
        X = pd.DataFrame({'category': ['Art', None], 'subcategory': ['Painting', None],
                          'state': ['CA', None], 'duration': [30, 40]})
        first = NativeCategories().fit_transform(X, [0, 1])
        second = NativeCategories().fit_transform(X, [1, 0])
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first.category.tolist(), ['Art', '__missing__'])
        self.assertEqual(list(first), list(X))

    def test_hgb_definition_preserved(self):
        a = comparison_pipeline('HGB', 'city').fit(self.X, self.y)
        b = experiment_pipeline(('city_frequency',)).fit(self.X, self.y)
        np.testing.assert_array_equal(a.predict_proba(self.X), b.predict_proba(self.X))

    def test_threshold_uses_failure_guardrail(self):
        y = pd.Series([0, 0, 1, 1])
        p = np.array([.47, .52, .53, .8])
        t = choose_threshold(y, p, {'failure_recall': .5})
        self.assertEqual(t, .5)

    def test_optional_algorithms_fit_native_and_encoded_inputs(self):
        for module, name in [('xgboost', 'XGBoost'), ('catboost', 'CatBoost'), ('lightgbm', 'LightGBM')]:
            with self.subTest(algorithm=name):
                if importlib.util.find_spec(module) is None:
                    continue
                params = {'depth': 3, 'iterations': 5} if name == 'CatBoost' else {'n_estimators': 5}
                model = clone(comparison_pipeline(name, 'city', params)).fit(self.X, self.y)
                validation = self.X.iloc[:2].assign(category='Unseen', location='Unknown, NY')
                p = model.predict_proba(validation)
                self.assertEqual(p.shape, (2, 2))
                self.assertTrue(np.isfinite(p).all())
                if name == 'CatBoost':
                    self.assertNotIn('prep', model.named_steps)
                    self.assertEqual(model.named_steps['model'].get_cat_feature_indices(), [5, 6, 7])


class RecommendationResearchTests(unittest.TestCase):
    def setUp(self):
        self.training = pd.DataFrame({'project id': range(60), 'goal': range(100, 160),
                                      'duration': [30]*60, 'category': ['Art']*60,
                                      'subcategory': ['Painting']*35+['Sculpture']*25,
                                      'location': ['Town, CA']*60})
        self.campaign = dict(goal=110, duration=30, category='Art', subcategory='Painting', location='CA')

    def test_excludes_self_and_outcomes_do_not_change_retrieval(self):
        a = nearest_campaigns(self.training, self.campaign, exclude_id=10)
        changed = self.training.assign(target=1, updates=999, pledged=999, backers=999)
        b = nearest_campaigns(changed, self.campaign, exclude_id=10)
        self.assertEqual(a, b)
        self.assertNotIn(10, [r['project_id'] for r in a['neighbors']])
        self.assertEqual(benchmarks(self.training, self.campaign)['n'], 35)

    def test_sparse_and_unseen_groups(self):
        c = dict(self.campaign, subcategory='Sculpture')
        self.assertIn('parent category fallback', benchmarks(self.training, c)['scope'])
        self.assertFalse(benchmarks(self.training, dict(c, category='Unknown'))['available'])
        self.assertFalse(benchmarks(self.training.iloc[:20], self.campaign)['available'])
        with self.assertRaises(ValueError):
            nearest_campaigns(self.training, self.campaign, k=0)

    def test_scenario_changes_are_validated_by_predictor(self):
        class Predictor:
            def predict(self, campaign):
                from utils.us_prediction import campaign_frame
                row = campaign_frame(campaign).iloc[0]
                return {'success_probability': float(row.goal)/1000}
        result = scenario_comparison(Predictor(), self.campaign, {'goal': 120})
        self.assertAlmostEqual(result['probability_difference'], .01)
        with self.assertRaises(ValueError):
            scenario_comparison(Predictor(), self.campaign, {'updates': 5})
        with self.assertRaises(ValueError):
            scenario_comparison(Predictor(), self.campaign, {'goal': -1})


if __name__ == '__main__':
    unittest.main()

import importlib.util
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from scripts.prepare_historical_models import historical_pipeline
from scripts.run_scope_improvement_study import (ALGORITHMS, candidate_pipeline, choose_family,
                                                  city_campaign_frame, compare_scope, eligible)


class ScopeImprovementStudyTests(unittest.TestCase):
    def test_promotion_requires_more_than_accuracy(self):
        baseline = dict(accuracy=.60, roc_auc=.65, failure_recall=.50)
        scores = dict(Logistic=baseline, HGB=dict(accuracy=.605, roc_auc=.649, failure_recall=.495),
                      XGBoost=dict(accuracy=.62, roc_auc=.62, failure_recall=.70),
                      CatBoost=dict(accuracy=.61, roc_auc=.66, failure_recall=.48),
                      LightGBM=dict(accuracy=.59, roc_auc=.66, failure_recall=.60))
        self.assertEqual(choose_family(scores, .55), 'HGB')
        self.assertEqual(choose_family(scores, .61), 'Logistic')
        self.assertFalse(eligible(scores['XGBoost'], baseline, .55))
        self.assertFalse(eligible(scores['CatBoost'], baseline, .55))

    def test_city_contract_does_not_invent_city_for_state_only_input(self):
        campaign = dict(goal=5000, duration=30, category='Music', subcategory='Rock', location='CA')
        with self.assertRaises(ValueError):
            city_campaign_frame(campaign)
        frame = city_campaign_frame(dict(campaign, location='Austin, tx'))
        self.assertEqual(frame.location.iloc[0], 'Austin, TX')

    def test_every_validation_prediction_uses_a_disjoint_fitting_partition(self):
        X = pd.DataFrame({'duration': np.arange(60) + 10, 'category': ['Music', 'Art'] * 30,
                          'subcategory': ['Rock', 'Painting'] * 30, 'country_region': ['Germany'] * 60})
        y = pd.Series([0, 1, 1, 0, 1, 0] * 10)
        fit_method = Pipeline.fit
        predict_method = Pipeline.predict_proba
        predicted = []

        def fitting(model, rows, target, **kwargs):
            model.study_fit_indices = set(rows.index)
            self.assertTrue(set(rows.index).issubset(X.index))
            return fit_method(model, rows, target, **kwargs)

        def predicting(model, rows, **kwargs):
            self.assertTrue(model.study_fit_indices.isdisjoint(rows.index))
            predicted.append(len(rows))
            return predict_method(model, rows, **kwargs)

        # Tiny Logistic stand-ins exercise split/selection orchestration without optional libraries.
        with patch('scripts.run_scope_improvement_study.candidate_pipeline', side_effect=lambda name: historical_pipeline()), \
                patch.object(Pipeline, 'fit', fitting), patch.object(Pipeline, 'predict_proba', predicting):
            summary, folds, inner, splits = compare_scope(X, y, 'synthetic')
        self.assertTrue(predicted)
        self.assertEqual(set(summary.algorithm), set([*ALGORITHMS, 'Majority', 'Inner family selection']))
        self.assertEqual(set(splits.validation_rows), {12})
        self.assertEqual(set(inner.fold), set(range(1, 6)))
        self.assertTrue(folds[folds.algorithm.eq('Inner family selection')].selected_algorithm.eq('Logistic').all())

    def test_available_algorithms_fit_the_identical_approved_schema(self):
        X = pd.DataFrame({'duration': np.arange(40) + 10, 'category': ['Music', 'Art'] * 20,
                          'subcategory': ['Rock', 'Painting'] * 20, 'country_region': ['Germany'] * 40})
        y = [0, 1, 1, 0] * 10
        modules = {'XGBoost': 'xgboost', 'CatBoost': 'catboost', 'LightGBM': 'lightgbm'}
        for algorithm in ALGORITHMS:
            if algorithm in modules and importlib.util.find_spec(modules[algorithm]) is None:
                continue
            with self.subTest(algorithm=algorithm):
                pipe = candidate_pipeline(algorithm).fit(X, y)
                self.assertEqual(list(pipe.feature_names_in_), list(X))
                self.assertEqual(list(pipe.classes_), [0, 1])
                self.assertTrue(np.isfinite(pipe.predict_proba(X.iloc[:2])).all())
                self.assertNotIn('goal', pipe.feature_names_in_)

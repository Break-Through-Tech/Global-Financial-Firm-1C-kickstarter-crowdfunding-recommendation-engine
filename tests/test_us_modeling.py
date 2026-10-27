import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd

from utils.us_modeling import (
    CONSERVATIVE_POLICY,
    EXPECTED_SETTLED_ROWS,
    EXPECTED_TRAIN_ROWS,
    development_cv,
    load_us_training_data,
    conservative_pipeline,
    run_randomized_search,
    search_result,
)


class USModelingUtilityTests(unittest.TestCase):
    def test_official_training_membership_and_binary_labels(self):
        X, y, project_ids = load_us_training_data()
        self.assertEqual(len(X), EXPECTED_TRAIN_ROWS)
        self.assertEqual(len(set(project_ids)), EXPECTED_TRAIN_ROWS)
        self.assertEqual(set(y.unique()), {0, 1})
        self.assertEqual(len(X), len(y))
        self.assertNotIn("test_project_ids", X.columns)

    def test_conservative_pipeline_and_inputs(self):
        self.assertEqual(CONSERVATIVE_POLICY, {
            "include_timing": False,
            "include_flags": False,
            "include_reward": False,
            "exclude_renamed": True,
        })
        X, y, _ = load_us_training_data()
        pipe = conservative_pipeline(LogisticRegression(max_iter=100), "onehot")
        pipe.fit(X.iloc[:100], y.iloc[:100])
        self.assertTrue(np.isfinite(pipe.predict_proba(X.iloc[:20])).all())
        transformed = pipe.named_steps["features"].transform(X.iloc[:2])
        self.assertNotIn("funded date", transformed.columns)

    def test_cv_is_reproducible_and_stratified(self):
        left = list(development_cv().split(np.zeros(20), [0, 1] * 10))
        right = list(development_cv().split(np.zeros(20), [0, 1] * 10))
        for (left_train, left_valid), (right_train, right_valid) in zip(left, right):
            np.testing.assert_array_equal(left_train, right_train)
            np.testing.assert_array_equal(left_valid, right_valid)
        self.assertEqual(development_cv().n_splits, 5)

    def test_search_result_contains_training_metrics(self):
        X = pd.DataFrame({
            "goal": np.arange(40, dtype=float) + 100,
            "duration": [30.0, 45.0] * 20,
            "category": ["Art", "Design"] * 20,
            "subcategory": ["Print", "Tools"] * 20,
            "location": ["Boston, MA", "Austin, TX"] * 20,
        })
        y = np.array([0, 1] * 20)
        search, trials = run_randomized_search(
            X, y, LogisticRegression(max_iter=300), "onehot",
            {"model__C": [0.1, 1.0]}, n_iter=2,
        )
        result = search_result(search, trials, "test logistic")
        for field in ("cv_roc_auc", "accuracy", "f1", "fit_seconds", "overfit_gap"):
            self.assertTrue(np.isfinite(result[field]), field)
        self.assertEqual(result["search_iterations"], 2)


if __name__ == "__main__":
    unittest.main()

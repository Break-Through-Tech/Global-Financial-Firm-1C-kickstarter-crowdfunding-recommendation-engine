"""Guardrails for optional U.S. feature experiments."""
import unittest
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from utils.us_launch_features import USLaunchFeatureBuilder
from utils.preprocessing import make_pipeline


class USLaunchFeatureTests(unittest.TestCase):
    @staticmethod
    def rows():
        n = 30
        return pd.DataFrame({
            "project id": np.arange(n),
            "goal": np.where(np.arange(n) % 2 == 0, 2000., 7000.),
            "duration": np.where(np.arange(n) % 3 == 0, 30, 60),
            "category": ["Music", "Art", "Technology"] * 10,
            "subcategory": ["Jazz", "Painting", "Hardware"] * 10,
            "location": ["Oakland, CA", "Albany, NY", "Reno, NV"] * 10,
            "reward levels": ["$15,$50"] * n,
            "url": [f"https://www.kickstarter.com/projects/u/p{i}" for i in range(n)],
            "name": ["project"] * n,
        })

    def test_extra_features_are_optional_and_cloneable(self):
        X = self.rows()
        b = USLaunchFeatureBuilder()
        result = b.fit_transform(X)
        self.assertTrue({"goal_vs_state", "goal_round_1000",
                         "duration_exact_30", "duration_exact_60"}.issubset(result.columns))
        self.assertFalse({"renamed", "backers_needed", "launch_month_sin"}.intersection(result.columns))
        cloned = clone(b)
        self.assertEqual(list(cloned.fit_transform(X).columns), list(result.columns))
        no_extra = USLaunchFeatureBuilder(include_extra_launch_features=False).fit_transform(X)
        self.assertNotIn("goal_vs_state", no_extra.columns)

    def test_state_median_is_training_only(self):
        X = self.rows()
        fit_rows = X.loc[X["location"] == "Oakland, CA"].copy()
        val = X.iloc[[0]].copy()
        val.loc[:, "goal"] = 800000
        builder = USLaunchFeatureBuilder().fit(fit_rows)
        predicted = builder.transform(val)["goal_vs_state"].iloc[0]
        expected = np.log1p(800000) - np.log1p(fit_rows["goal"]).median()
        self.assertAlmostEqual(predicted, expected)
        unknown = val.copy()
        unknown["location"] = "Unknown, ZZ"
        self.assertTrue(np.isfinite(builder.transform(unknown)["goal_vs_state"]).all())

    def test_extended_pipeline_runs_cv(self):
        X = self.rows()
        y = pd.Series(np.arange(len(X)) % 2)
        pipe = make_pipeline(LogisticRegression(max_iter=2000), categorical_strategy="onehot",
                             include_timing=False, include_flags=False,
                             include_reward=False, exclude_renamed=True)
        pipe.set_params(features=USLaunchFeatureBuilder())
        scores = cross_val_score(pipe, X, y,
                                cv=StratifiedKFold(3, shuffle=True, random_state=42),
                                scoring="accuracy", error_score="raise")
        self.assertTrue(np.isfinite(scores).all())


if __name__ == "__main__":
    unittest.main()

import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from utils.preprocessing import FeatureBuilder, make_pipeline
from utils.us_scope import align_to_global_split, us_location_audit, us_state_proxy_mask


class USScopeTests(unittest.TestCase):
    def test_proxy_excludes_missing_and_foreign_suffixes(self):
        df = pd.DataFrame({"location": ["San Francisco, CA", "Albany, NY", "Toronto, ON",
                                           "Berlin, BE", None, "Boston, ma", "London"]})
        self.assertEqual(us_state_proxy_mask(df).tolist(), [True, True, False, False, False, True, False])
        summary, excluded = us_location_audit(df)
        self.assertEqual(int(summary.loc[0, "candidate_us_rows"]), 3)
        self.assertGreaterEqual(len(excluded), 2)

    def test_full_states_territories_and_bare_codes_do_not_expand_proxy(self):
        locations = ["St Croix, Virgin Islands, U.S.", "Cruz Bay, Virgin Islands, U.S.",
                     "Virgin Islands of the United States", "Charlottesville, Virginia",
                     "San Francisco, CA", "CA", "Toronto, ON", "Tbilisi, Georgia",
                     "Kinshasa, Congo"]
        self.assertEqual(us_state_proxy_mask(pd.DataFrame({"location": locations})).tolist(),
                         [False, False, False, False, True, False, False, False, False])

    def test_preserves_global_split_and_targets(self):
        us = pd.DataFrame({"project id": [101, 102, 103, 104],
                           "status": ["successful", "failed", "live", "successful"]})
        tr = pd.DataFrame({"project id": [104, 101, 103, 999], "target": [1, 1, 0, 0]})
        te = pd.DataFrame({"project id": [102, 888], "target": [0, 1]})
        a, b = align_to_global_split(us, tr, te)
        self.assertEqual(a["project id"].tolist(), [104, 101])
        self.assertEqual(b["project id"].tolist(), [102])
        self.assertEqual(a["target"].tolist(), [1, 1])
        self.assertEqual(b["target"].tolist(), [0])

    def test_rejects_global_split_overlap(self):
        us = pd.DataFrame({"project id": [1], "status": ["failed"]})
        ids = pd.DataFrame({"project id": [1], "target": [0]})
        with self.assertRaises(ValueError):
            align_to_global_split(us, ids, ids)

    @staticmethod
    def example(n=40):
        return pd.DataFrame({
            "project id": np.arange(n),
            "goal": np.linspace(100, 5000, n),
            "duration": np.tile([30, 45], n // 2),
            "levels": np.tile([3, 5], n // 2),
            "category": np.tile(["Art", "Technology"], n // 2),
            "subcategory": np.tile(["Painting", "Gadgets"], n // 2),
            "location": np.tile(["Reno, NV", "Boston, MA"], n // 2),
            "funded date": ["not a verified deadline"] * n,
            "reward levels": ["$5,$20"] * n,
            "name": ["Example title"] * n,
            "url": [f"https://www.kickstarter.com/projects/123/project-{k}" for k in range(n)],
        })

    def test_conservative_features_exclude_unverified_groups(self):
        x = self.example()
        safe = FeatureBuilder(include_timing=False, include_flags=False,
                              include_reward=False, exclude_renamed=True).fit_transform(x)
        self.assertIn("duration", safe.columns)
        self.assertIn("goal_vs_cat", safe.columns)
        self.assertNotIn("renamed", safe.columns)
        self.assertNotIn("backers_needed", safe.columns)
        self.assertNotIn("n_tiers", safe.columns)
        self.assertNotIn("launch_month_sin", safe.columns)

    def test_conservative_pipeline_fits_cv_with_unknown_funded_date(self):
        x = self.example()
        y = np.array([0, 1, 1, 0] * 10)
        pipeline = make_pipeline(LogisticRegression(max_iter=400),
                                 categorical_strategy="onehot", include_timing=False,
                                 include_flags=False, include_reward=False,
                                 exclude_renamed=True)
        scores = cross_val_score(pipeline, x, y,
                     cv=StratifiedKFold(3, shuffle=True, random_state=42),
                     scoring="roc_auc", n_jobs=1, error_score="raise")
        self.assertTrue(np.isfinite(scores).all())


if __name__ == "__main__":
    unittest.main()

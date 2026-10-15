import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from utils.international_features import InternationalFeatureBuilder, make_international_pipeline


class InternationalFeatureTests(unittest.TestCase):
    @staticmethod
    def rows(n=50):
        locs = ["Berlin, Germany", "Paris, France", "Toronto, Canada", "Sydney, Australia"]
        return pd.DataFrame({
            "project id": np.arange(n),
            "goal": np.linspace(200, 20000, n),
            "levels": np.arange(n) % 7 + 1,
            "duration": np.tile([30, 45], n//2),
            "category": np.tile(["Art", "Technology"], n//2),
            "subcategory": np.tile(["Painting", "Gadgets"], n//2),
            "location": [locs[i%4] for i in range(n)],
            "name": ["Edited title"]*n,
            "url": [f"https://kickstarter.com/projects/creator/{i}" for i in range(n)],
            "reward levels": ["$5,$15"]*n,
            "funded date": ["unverified postlaunch"]*n,
            "pledged": [1_000_000]*n,
            "updates": [999]*n,
            "comments": [999]*n,
            "backers": [999]*n,
            "funded percentage": [999]*n,
        })

    def test_country_is_not_mislabeled_state(self):
        data = self.rows()
        features = InternationalFeatureBuilder().fit_transform(data)
        self.assertIn("country_region", features)
        self.assertNotIn("state", features)
        self.assertNotIn("goal_vs_cat", features)
        self.assertNotIn("goal_vs_subcat", features)
        self.assertIn("goal_vs_country", features)
        self.assertIn("goal_vs_country_cat", features)
        self.assertNotIn("renamed", features)
        self.assertNotIn("backers_needed", features)
        self.assertNotIn("launch_dow", features)
        self.assertNotIn("updates", features)

    def test_country_median_uses_fit_only(self):
        x = self.rows()
        obj = InternationalFeatureBuilder(min_group_n=2).fit(x.iloc[:40])
        initial_medians = obj.country_goal_medians_.copy()
        future = x.iloc[40:].copy()
        future["goal"] = future["goal"]*1000
        transformed = obj.transform(future)
        self.assertEqual(len(transformed), len(future))
        pd.testing.assert_series_equal(initial_medians, obj.country_goal_medians_)
        self.assertTrue(np.isfinite(transformed["goal_vs_country"]).all())

    def test_unseen_country_falls_back_training_median(self):
        x = self.rows()
        obj = InternationalFeatureBuilder(min_group_n=10).fit(x.iloc[:30])
        future = x.iloc[30:32].copy()
        future["location"] = ["Lima, Peru", "Lima, Peru"]
        out = obj.transform(future)
        self.assertTrue(np.isfinite(out["goal_vs_country"]).all())
        self.assertEqual(out["country_region"].tolist(), ["Peru", "Peru"])

    def test_minimal_approved_inputs_without_rewards_or_url(self):
        x = self.rows()[["goal", "duration", "category", "subcategory", "location"]]
        pipe = make_international_pipeline(LogisticRegression(max_iter=400))
        pipe.fit(x, [0, 1] * 25)
        self.assertTrue(np.isfinite(pipe.predict_proba(x.iloc[:2])).all())

    def test_sparse_group_and_unseen_country_fallbacks_numerically(self):
        x = pd.DataFrame({
            "goal": [100., 200., 300., 400.], "duration": [30.] * 4,
            "category": ["Art", "Art", "Design", "Design"],
            "subcategory": ["Paint", "Paint", "Tools", "Tools"],
            "location": ["Berlin, Germany"] * 3 + ["Lima, Peru"],
        })
        builder = InternationalFeatureBuilder(min_group_n=2).fit(x)
        future = x.iloc[[2, 3, 0]].copy().reset_index(drop=True)
        future.loc[2, "location"] = "Paris, France"
        out = builder.transform(future)
        # Germany/Design is sparse: use supported Germany median.
        germany = np.median(np.log1p([100., 200., 300.]))
        global_ref = np.median(np.log1p(x.goal))
        expected_ref = np.array([germany, global_ref, global_ref])
        expected = np.log1p(future.goal.to_numpy()) - expected_ref
        np.testing.assert_allclose(out.goal_vs_country, expected)
        np.testing.assert_allclose(out.goal_vs_country_cat, expected)

    def test_unknown_locations_refused_not_guessed(self):
        x = self.rows()
        obj = InternationalFeatureBuilder().fit(x)
        bad = x.iloc[:1].copy()
        bad["location"] = "Toronto, ON"
        with self.assertRaises(ValueError):
            obj.transform(bad)

    def test_full_pipeline_refits_and_handles_unverified_funded_date(self):
        x = self.rows(80)
        y = np.array([1,0,1,0,0,1,0,1]*10)
        pipe = make_international_pipeline(LogisticRegression(max_iter=400))
        scores = cross_val_score(pipe, x, y, cv=StratifiedKFold(3, shuffle=True, random_state=42),
                                 scoring="roc_auc", error_score="raise", n_jobs=1)
        self.assertEqual(len(scores), 3)
        self.assertTrue(np.isfinite(scores).all())


if __name__ == "__main__":
    unittest.main()

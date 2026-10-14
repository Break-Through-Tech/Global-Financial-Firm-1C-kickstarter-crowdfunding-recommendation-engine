"""Regression checks for launch-time filtering and leakage-safe target encoding.

Run from repository root: python3 -m unittest discover -s tests -v
"""
import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

from utils.preprocessing import FeatureBuilder, TargetEncoder, drop_leaks, make_pipeline


class TargetEncoderTests(unittest.TestCase):
    def test_same_index_with_new_categories_does_not_return_cached_oof(self):
        original = pd.DataFrame({"category": ["A", "A", "B", "B", "C", "C"]})
        target = np.array([1, 0, 1, 0, 0, 1])
        encoder = TargetEncoder(cols=["category"], smoothing=10, n_splits=3, random_state=17)
        oof = encoder.fit_transform(original, target)

        # Reuse the *identical integer index* with entirely unseen categories.
        unseen = pd.DataFrame({"category": ["Z"] * len(original)}, index=original.index)
        predicted = encoder.transform(unseen).ravel()
        np.testing.assert_allclose(predicted, encoder.prior_)
        self.assertEqual(oof.shape, (6, 1))

    def test_fold_prior_is_fitted_without_validation_targets(self):
        from sklearn.model_selection import KFold

        rows = pd.DataFrame({"category": list("ABCDEFGH")})
        y = np.array([1, 1, 1, 1, 1, 0, 0, 0], dtype=float)
        encoder = TargetEncoder(cols=["category"], n_splits=4, random_state=42)
        oof = encoder.fit_transform(rows, y).ravel()
        for fit_idx, valid_idx in KFold(4, shuffle=True, random_state=42).split(rows):
            # Every category is unique. Inner validation rows must use the
            # inner *training* prior, never the full-dataset target mean.
            np.testing.assert_allclose(oof[valid_idx], y[fit_idx].mean())

    def test_category_mapping_handles_reordered_rows(self):
        rows = pd.DataFrame({"category": ["A", "B", "A", "B", "C", "C"]}, index=[5, 4, 3, 2, 1, 0])
        y = [1, 0, 1, 0, 1, 0]
        encoder = TargetEncoder(cols=["category"], n_splits=3).fit(rows, y)
        expected = encoder.transform(rows)
        swapped = rows.iloc[::-1].copy()
        np.testing.assert_allclose(encoder.transform(swapped), expected[::-1])

    def test_fit_transform_is_deterministic(self):
        rows = pd.DataFrame({"category": list("ABCDABCDABCD")})
        y = np.array([0, 1] * 6)
        enc1 = TargetEncoder(cols=["category"], n_splits=3, random_state=1)
        enc2 = TargetEncoder(cols=["category"], n_splits=3, random_state=1)
        np.testing.assert_allclose(enc1.fit_transform(rows, y), enc2.fit_transform(rows, y))

    def test_invalid_split_count_is_rejected(self):
        rows = pd.DataFrame({"category": ["A", "B", "C"]})
        with self.assertRaises(ValueError):
            TargetEncoder(cols=["category"], n_splits=5).fit_transform(rows, [0, 1, 0])


class FeatureAvailabilityTests(unittest.TestCase):
    @staticmethod
    def example_rows(n=30):
        return pd.DataFrame({
            "project id": np.arange(n),
            "name": [f"A campaign {i}" for i in range(n)],
            "url": [f"https://www.kickstarter.com/projects/creator{i}/original-title-{i}" for i in range(n)],
            "category": ["Art" if i % 2 else "Design" for i in range(n)],
            "subcategory": ["Print" if i % 2 else "Tools" for i in range(n)],
            "location": ["San Francisco, CA" if i % 2 else "Miami, FL" for i in range(n)],
            "goal": np.arange(n) * 100 + 500,
            "duration": [30 if i % 2 else 40 for i in range(n)],
            "funded date": ["2012-06-25"] * n,
            "reward levels": ["$10,$30,$100"] * n,
            "pledged": [10000] * n,
            "backers": [200] * n,
            "funded percentage": [150.0] * n,
            "updates": [5] * n,
            "comments": [10] * n,
        })

    def test_postlaunch_outcomes_removed(self):
        X = self.example_rows()
        stripped = drop_leaks(X)
        for name in ("pledged", "backers", "funded percentage", "updates", "comments"):
            self.assertNotIn(name, stripped.columns)
        self.assertEqual(len(stripped), len(X))

    def test_renamed_can_be_excluded_without_touching_shared_features(self):
        X = drop_leaks(self.example_rows())
        with_rename = FeatureBuilder().fit_transform(X)
        without_rename = FeatureBuilder(exclude_renamed=True).fit_transform(X)
        self.assertIn("renamed", with_rename.columns)
        self.assertNotIn("renamed", without_rename.columns)
        self.assertEqual(set(with_rename.columns) - set(without_rename.columns), {"renamed"})

    def test_pipeline_works_with_cross_validation_without_postlaunch_inputs(self):
        X = drop_leaks(self.example_rows())
        y = np.arange(len(X)) % 2
        model = make_pipeline(
            LogisticRegression(max_iter=300),
            exclude_renamed=True,
        )
        scores = cross_val_score(
            model, X, y,
            scoring="roc_auc",
            cv=StratifiedKFold(3, shuffle=True, random_state=42),
            n_jobs=1,
            error_score="raise",
        )
        self.assertEqual(len(scores), 3)
        self.assertTrue(np.isfinite(scores).all())


if __name__ == "__main__":
    unittest.main()

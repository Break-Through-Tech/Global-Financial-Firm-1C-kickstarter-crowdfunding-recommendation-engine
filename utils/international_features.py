"""Conservative international launch-time features and fold-safe modeling.

This is deliberately separate from the global/U.S. feature builders. The
international geographic unit is an explicit *country/region name*, not the
U.S. state suffix. Country-relative goal medians are learned only from the
training fold and are not labels or causal recommendations.

Caveat: whether historical Kickstarter 'goal' values are in comparable
currencies/converted to USD remains unverified. Do not interpret cross-country
raw goal differences as meaningful until source documentation is checked.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils.international_scope import classify_locations
from utils.preprocessing import FeatureBuilder, TargetEncoder

CATEGORICAL = ["category", "subcategory", "country_region"]


def _eligible_countries(X):
    geo = classify_locations(X["location"])
    if not geo["scope"].eq("explicit_non_us").all():
        raise ValueError("Non-U.S. model input requires an explicit named non-U.S. country/region; unknown locations cannot be inferred")
    return geo["country_region"].astype(str)


class InternationalFeatureBuilder(FeatureBuilder):
    """Drop fake U.S. 'state' and derive country-relative goal features.

    Category and subcategory features from global FeatureBuilder are removed:
    they would compare goal medians across potentially different currencies.
    Country / country-and-category medians are fitted from the current training
    fold only. Rare groups fall back to the country median, not unseen data.
    """

    def __init__(self, include_timing=False, include_flags=False,
                 exclude_renamed=True, include_reward=False, min_group_n=10):
        super().__init__(include_timing=include_timing, include_flags=include_flags,
                         exclude_renamed=exclude_renamed, include_reward=include_reward)
        self.min_group_n = min_group_n

    def fit(self, X, y=None):
        if self.min_group_n < 1:
            raise ValueError("min_group_n must be >= 1")
        country = _eligible_countries(X)
        super().fit(X, y)
        goal = pd.to_numeric(X["goal"], errors="coerce").where(lambda s: s.ge(0))
        lg = np.log1p(goal)
        records = pd.DataFrame({
            "country_region": country.to_numpy(),
            "category": X["category"].fillna("missing").astype(str).to_numpy(),
            "log_goal": lg.to_numpy(),
        })
        country_stats = records.groupby("country_region")["log_goal"].agg(["median", "count"])
        country_cat_stats = records.groupby(["country_region", "category"])["log_goal"].agg(["median", "count"])
        self.overall_goal_median_ = float(lg.median())
        self.country_goal_medians_ = country_stats.loc[country_stats["count"] >= self.min_group_n, "median"]
        self.country_cat_medians_ = country_cat_stats.loc[country_cat_stats["count"] >= self.min_group_n, "median"]
        return self

    def transform(self, X):
        countries = _eligible_countries(X)
        base = super().transform(X)
        base = base.drop(columns=["state", "goal_vs_cat", "goal_vs_subcat"], errors="ignore")
        goal = pd.to_numeric(X["goal"], errors="coerce").where(lambda s: s.ge(0))
        log_goal = np.log1p(goal)
        country_ref = countries.map(self.country_goal_medians_).fillna(self.overall_goal_median_)
        groups = pd.MultiIndex.from_arrays([
            countries.to_numpy(),
            X["category"].fillna("missing").astype(str).to_numpy(),
        ])
        group_ref = pd.Series(self.country_cat_medians_.reindex(groups).to_numpy(), index=X.index)
        group_ref = group_ref.fillna(country_ref)
        base["goal_vs_country"] = (log_goal - country_ref).to_numpy()
        base["goal_vs_country_cat"] = (log_goal - group_ref).to_numpy()
        base["country_region"] = countries.to_numpy()
        return base


def make_international_pipeline(model, categorical_strategy="onehot", **builder_kwargs):
    """Full fold-refitted international pipeline, no global code changes."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    if categorical_strategy == "onehot":
        categorical = Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False)),
        ])
    elif categorical_strategy == "target":
        categorical = TargetEncoder(cols=CATEGORICAL)
    else:
        raise ValueError("Categorical strategy must be onehot or target")
    prep = ColumnTransformer([
        ("numeric", numeric, make_column_selector(dtype_include=np.number)),
        ("categorical", categorical, CATEGORICAL),
    ], remainder="drop", verbose_feature_names_out=False)
    return Pipeline([
        ("features", InternationalFeatureBuilder(**builder_kwargs)),
        ("prep", prep),
        ("model", model),
    ])

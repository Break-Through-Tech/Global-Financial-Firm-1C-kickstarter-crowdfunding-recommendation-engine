"""Preprocessing pipeline for the Kickstarter data.

Anything that learns from the data (medians, category means, scalers) lives
inside a transformer, so it only ever sees the training fold.

    X_train, X_test, y_train, y_test = get_data_split(df)
    pipe = make_pipeline(LogisticRegression())
    pipe.fit(drop_leaks(X_train), y_train)
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils.features import (
    LEAK_COLS,
    add_base_columns,
    ask_features,
    effort_flags,
    reward_features,
    timing_features,
)

CATEGORICAL = ["category", "subcategory", "state"]


class FeatureBuilder(BaseEstimator, TransformerMixin):
    "Raw columns in, engineered features plus the categoricals out."

    def __init__(self, include_timing=True, include_flags=True):
        self.include_timing = include_timing
        self.include_flags = include_flags

    def fit(self, X, y=None):
        # learn the goal medians here so test rows are measured against the training norm
        base = add_base_columns(X)
        _, self.subcat_median_, self.cat_median_ = ask_features(base)
        return self

    def transform(self, X):
        base = add_base_columns(X)
        ask, _, _ = ask_features(base, self.subcat_median_, self.cat_median_)

        blocks = [ask, reward_features(base)]
        if self.include_flags:
            blocks.append(effort_flags(base))
        if self.include_timing:
            blocks.append(timing_features(base))

        out = pd.concat(blocks, axis=1)
        out = out.loc[:, ~out.columns.duplicated()]
        for c in CATEGORICAL:
            out[c] = base[c].astype("object")
        return out


class TargetEncoder(BaseEstimator, TransformerMixin):
    """Replace a category with its smoothed success rate.

    Training values are computed out of fold so a row never sees its own label.
    """

    def __init__(self, cols=None, smoothing=50, n_splits=5, random_state=42):
        self.cols = cols
        self.smoothing = smoothing
        self.n_splits = n_splits
        self.random_state = random_state

    def _mapping(self, keys, y):
        g = pd.DataFrame({"k": keys.values, "y": np.asarray(y, dtype=float)}).groupby("k")["y"]
        m, n = g.mean(), g.size()
        return (m * n + self.prior_ * self.smoothing) / (n + self.smoothing)

    def fit(self, X, y):
        self.cols_ = self.cols if self.cols is not None else list(X.columns)
        y = pd.Series(np.asarray(y, dtype=float), index=X.index)
        self.prior_ = y.mean()
        self.maps_ = {c: self._mapping(X[c].astype(str), y) for c in self.cols_}

        kf = KFold(self.n_splits, shuffle=True, random_state=self.random_state)
        oof = pd.DataFrame(index=X.index, columns=self.cols_, dtype=float)
        for fit_idx, val_idx in kf.split(X):
            fold_y = y.iloc[fit_idx]
            for c in self.cols_:
                m = self._mapping(X[c].astype(str).iloc[fit_idx], fold_y)
                oof.iloc[val_idx, oof.columns.get_loc(c)] = (
                    X[c].astype(str).iloc[val_idx].map(m).fillna(self.prior_).values
                )
        self.oof_ = oof
        self.feature_names_out_ = [f"te_{c}" for c in self.cols_]
        return self

    def transform(self, X):
        # same rows as fit means we are in training, so hand back the out of fold values
        if hasattr(self, "oof_") and X.index.equals(self.oof_.index):
            out = self.oof_.copy()
        else:
            out = pd.DataFrame(
                {c: X[c].astype(str).map(self.maps_[c]).fillna(self.prior_) for c in self.cols_},
                index=X.index,
            )
        out.columns = self.feature_names_out_
        return out.values

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_out_)


class SelectColumns(BaseEstimator, TransformerMixin):
    "Keep a named subset. Needs the preprocessor set to pandas output."

    def __init__(self, keep):
        self.keep = keep

    def fit(self, X, y=None):
        self.keep_ = list(self.keep)
        missing = [c for c in self.keep_ if c not in X.columns]
        if missing:
            raise KeyError(f"not produced by the preprocessor: {missing}")
        return self

    def transform(self, X):
        return X[self.keep_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.keep_)


def drop_leaks(X):
    "Remove the five outcome columns get_data_split passes through."
    return X.drop(columns=[c for c in LEAK_COLS if c in X.columns])


def build_preprocessor(categorical_strategy="target", numeric_impute="median"):
    """Scale the numerics, encode the categoricals.

    categorical_strategy: "target", "onehot" or "drop".
    """
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy=numeric_impute)),
        ("scale", StandardScaler()),
    ])

    if categorical_strategy == "target":
        cat = TargetEncoder(cols=CATEGORICAL)
    elif categorical_strategy == "onehot":
        cat = Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=30,
                                     sparse_output=False)),
        ])
    elif categorical_strategy == "drop":
        cat = "drop"
    else:
        raise ValueError(f"unknown categorical_strategy {categorical_strategy!r}")

    return ColumnTransformer(
        [
            ("num", numeric, make_column_selector(dtype_include=np.number)),
            ("cat", cat, CATEGORICAL),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_pipeline(model, categorical_strategy="target", keep=None, **builder_kwargs):
    "Features, preprocessing and model in one estimator. Pass keep to use a subset."
    prep = build_preprocessor(categorical_strategy)
    steps = [("features", FeatureBuilder(**builder_kwargs)), ("prep", prep)]
    if keep is not None:
        prep.set_output(transform="pandas")
        steps.append(("select", SelectColumns(keep)))
    steps.append(("model", model))
    return Pipeline(steps)


def feature_names(fitted_pipeline):
    "Column names coming out of the preprocessor of a fitted pipeline."
    return list(fitted_pipeline.named_steps["prep"].get_feature_names_out())

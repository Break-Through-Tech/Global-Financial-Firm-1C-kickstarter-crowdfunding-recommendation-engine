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

    def __init__(self, include_timing=True, include_flags=True, exclude_renamed=False):
        self.include_timing = include_timing
        self.include_flags = include_flags
        self.exclude_renamed = exclude_renamed

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
        if self.exclude_renamed:
            out = out.drop(columns=["renamed"], errors="ignore")
        for c in CATEGORICAL:
            out[c] = base[c].astype("object")
        return out


class TargetEncoder(BaseEstimator, TransformerMixin):
    """Smoothed category success rates with training-only cross-fitting.

    fit_transform returns out-of-fold encodings during model training.
    transform ALWAYS uses the full fitted mapping for arbitrary prediction rows.
    In particular, index equality is not a valid test of row identity.
    """

    def __init__(self, cols=None, smoothing=50, n_splits=5, random_state=42):
        self.cols = cols
        self.smoothing = smoothing
        self.n_splits = n_splits
        self.random_state = random_state

    @staticmethod
    def _mapping(keys, y, smoothing, prior):
        grouped = pd.DataFrame({
            "key": pd.Series(keys).astype(str).to_numpy(),
            "y": np.asarray(y, dtype=float),
        }).groupby("key")["y"]
        means, counts = grouped.mean(), grouped.size()
        return (means * counts + prior * smoothing) / (counts + smoothing)

    def fit(self, X, y):
        self.cols_ = list(self.cols) if self.cols is not None else list(X.columns)
        targets = np.asarray(y, dtype=float).reshape(-1)
        if len(targets) != len(X):
            raise ValueError("X and y must have the same number of rows")
        if len(targets) == 0 or not np.isfinite(targets).all():
            raise ValueError("y must contain finite values")
        if self.smoothing < 0:
            raise ValueError("smoothing must be nonnegative")
        if not set(self.cols_).issubset(X.columns):
            raise ValueError("X is missing target-encoding columns")

        self.prior_ = float(targets.mean())
        self.maps_ = {
            c: self._mapping(X[c], targets, self.smoothing, self.prior_)
            for c in self.cols_
        }
        self.feature_names_out_ = [f"te_{c}" for c in self.cols_]
        return self

    def fit_transform(self, X, y, **fit_params):
        # ColumnTransformer/Pipeline call fit_transform on training folds.
        # Build full-data mappings for future predictions, then return strictly
        # out-of-fold encodings to the training model.
        self.fit(X, y)
        if not 2 <= self.n_splits <= len(X):
            raise ValueError("n_splits must be between 2 and the number of training rows")

        targets = np.asarray(y, dtype=float).reshape(-1)
        keys = {c: X[c].astype(str) for c in self.cols_}
        oof = np.empty((len(X), len(self.cols_)), dtype=float)
        splitter = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for fit_idx, val_idx in splitter.split(X):
            fold_prior = float(targets[fit_idx].mean())
            for j, c in enumerate(self.cols_):
                mapping = self._mapping(
                    keys[c].iloc[fit_idx], targets[fit_idx], self.smoothing, fold_prior
                )
                oof[val_idx, j] = (
                    keys[c].iloc[val_idx].map(mapping).fillna(fold_prior).to_numpy()
                )
        return oof

    def transform(self, X):
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self, ["maps_", "prior_"])
        # Never return cached training rows based on their DataFrame index.
        return np.column_stack([
            X[c].astype(str).map(self.maps_[c]).fillna(self.prior_).to_numpy()
            for c in self.cols_
        ])

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

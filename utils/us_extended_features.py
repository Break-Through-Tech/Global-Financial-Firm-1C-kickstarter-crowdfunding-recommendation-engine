"""Training-only feature experiments. No production model policy changes."""
import re
import string
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import StratifiedKFold

from utils.features import parse_tiers
from utils.preprocessing import FeatureBuilder
from utils.us_modeling import CONSERVATIVE_POLICY
from utils.us_validation import candidates, metrics

GROUPS = ('reward_count', 'reward_amounts', 'reward_relative',
          'title_length', 'title_style', 'city_frequency')


def city_keys(X):
    # Full listed city/state string, not a population or creator-history estimate.
    return X['location'].fillna('').astype(str).str.strip().str.casefold()


class ExtendedFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, groups=()):
        self.groups = groups

    def fit(self, X, y=None):
        if set(self.groups) - set(GROUPS):
            raise ValueError('Unknown experimental feature group')
        self.base_ = FeatureBuilder(**CONSERVATIVE_POLICY).fit(X, y)
        if 'city_frequency' in self.groups:
            self.city_counts_ = city_keys(X).value_counts()
        return self

    def transform(self, X):
        out = self.base_.transform(X)
        if any(g.startswith('reward_') for g in self.groups):
            tiers = X['reward levels'].map(parse_tiers)
            amounts = pd.DataFrame({
                'min': tiers.map(lambda t: min(t) if t else np.nan),
                'max': tiers.map(lambda t: max(t) if t else np.nan),
                'median': tiers.map(lambda t: np.median(t) if t else np.nan),
            }, index=X.index)
            if 'reward_count' in self.groups:
                out['reward_count'] = tiers.map(len)
            if 'reward_amounts' in self.groups:
                for c in amounts:
                    out['reward_log_' + c] = np.log1p(amounts[c])
            if 'reward_relative' in self.groups:
                for c in amounts:
                    out['reward_relative_' + c] = np.log1p(amounts[c]) - np.log1p(X['goal'])
        if any(g.startswith('title_') for g in self.groups):
            title = X['name'].fillna('').astype(str)
            if 'title_length' in self.groups:
                out['title_characters'] = title.str.len()
                out['title_words'] = title.str.split().map(len)
            if 'title_style' in self.groups:
                alpha = title.str.count('[A-Za-z]').replace(0, np.nan)
                out['title_upper_ratio'] = (title.str.count('[A-Z]') / alpha).fillna(0)
                out['title_numbers'] = title.str.contains(r'\d').astype(int)
                out['title_punctuation'] = title.str.count('[' + re.escape(string.punctuation) + ']')
                out['title_exclamation'] = title.str.count('!')
        if 'city_frequency' in self.groups:
            out['log_city_frequency'] = np.log1p(city_keys(X).map(self.city_counts_).fillna(0))
        return out


def experiment_pipeline(groups=()):
    pipe = candidates()['HGB tuned']
    if groups:
        pipe.set_params(features=ExtendedFeatures(groups))
    return pipe


def eligible(candidate, baseline):
    return (candidate['accuracy'] >= baseline['accuracy'] + .003
            and candidate['roc_auc'] >= baseline['roc_auc'] - .002
            and candidate['failure_recall'] >= baseline['failure_recall'] - .01)


def compare_features(X, y):
    """Fixed singles plus inner-selected combinations, using existing outer folds.

    The exploratory selector may use uncertain snapshot features; the approved
    selector only considers baseline/city frequency. Neither writes a model.
    """
    names = ['baseline', *GROUPS, 'exploratory_selection', 'approved_selection']
    probability = {n: np.empty(len(y)) for n in names}
    records, inner_records = [], []
    outer = StratifiedKFold(3, shuffle=True, random_state=20261027)
    for fold, (train, valid) in enumerate(outer.split(X, y), 1):
        train_X, train_y = X.iloc[train], y.iloc[train]
        inner = list(StratifiedKFold(2, shuffle=True, random_state=100 + fold).split(train_X, train_y))
        configs = {'baseline': (), **{g: (g,) for g in GROUPS}}
        scores = {}
        for name, groups in configs.items():
            p = np.empty(len(train_y))
            for fit, val in inner:
                fitted = experiment_pipeline(groups).fit(train_X.iloc[fit], train_y.iloc[fit])
                p[val] = fitted.predict_proba(train_X.iloc[val])[:, 1]
            scores[name] = metrics(train_y, p)
        # One combination at most, defined from inner singleton evidence only.
        improving = [g for g in GROUPS if eligible(scores[g], scores['baseline'])]
        best_two = sorted(improving, key=lambda g: scores[g]['accuracy'], reverse=True)[:2]
        if len(best_two) == 2:
            configs['inner_supported_pair'] = tuple(best_two)
            p = np.empty(len(train_y))
            for fit, val in inner:
                fitted = experiment_pipeline(tuple(best_two)).fit(train_X.iloc[fit], train_y.iloc[fit])
                p[val] = fitted.predict_proba(train_X.iloc[val])[:, 1]
            scores['inner_supported_pair'] = metrics(train_y, p)
        allowed = [n for n in configs if eligible(scores[n], scores['baseline'])]
        chosen = max(allowed, key=lambda n: scores[n]['accuracy']) if allowed else 'baseline'
        approved = 'city_frequency' if eligible(scores['city_frequency'], scores['baseline']) else 'baseline'
        for name, score in scores.items():
            inner_records.append(dict(fold=fold, configuration=name, groups=','.join(configs[name]),
                                      selected=name == chosen, **score))
        cache = {}
        for name, groups in configs.items():
            start = time.perf_counter()
            fitted = experiment_pipeline(groups).fit(train_X, train_y)
            fit_seconds = time.perf_counter() - start
            start = time.perf_counter()
            p = fitted.predict_proba(X.iloc[valid])[:, 1]
            predict_seconds = time.perf_counter() - start
            cache[name] = (p, fit_seconds, predict_seconds)
            if name in probability:
                probability[name][valid] = p
            records.append(dict(fold=fold, configuration=name, groups=','.join(groups),
                                fit_seconds=fit_seconds, predict_seconds=predict_seconds,
                                **metrics(y.iloc[valid], p)))
        for label, name in [('exploratory_selection', chosen), ('approved_selection', approved)]:
            p, fit_seconds, predict_seconds = cache[name]
            probability[label][valid] = p
            records.append(dict(fold=fold, configuration=label, groups=','.join(configs[name]),
                                fit_seconds=fit_seconds, predict_seconds=predict_seconds,
                                **metrics(y.iloc[valid], p)))
        print(f'Extended features: outer fold {fold}/3 complete', flush=True)
    folds = pd.DataFrame(records)
    summary = pd.DataFrame({n: metrics(y, p) for n, p in probability.items()}).T
    summary = summary.join(folds.groupby('configuration')[['fit_seconds', 'predict_seconds']].mean())
    return summary, folds, pd.DataFrame(inner_records)

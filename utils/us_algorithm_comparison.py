"""Restricted training-only algorithm comparison with inner model selection."""
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from utils.preprocessing import CATEGORICAL, build_preprocessor
from utils.us_extended_features import ExtendedFeatures, eligible, experiment_pipeline
from utils.us_validation import metrics

FEATURE_SETS = {
    'conservative': (),
    'city': ('city_frequency',),
    'reward_count_exploratory': ('reward_count',),
    'reward_ladder_exploratory': ('reward_count', 'reward_amounts', 'reward_relative'),
}
ALGORITHMS = ('HGB', 'XGBoost', 'CatBoost', 'LightGBM')
THRESHOLDS = (.45, .5, .55)


class NativeCategories(BaseEstimator, TransformerMixin):
    """Keep CatBoost categoricals as strings, not external target encodings."""
    def fit(self, X, y=None):
        self.columns_ = list(X.columns)
        return self

    def transform(self, X):
        out = X[self.columns_].copy()
        for c in CATEGORICAL:
            out[c] = out[c].fillna('__missing__').astype(str)
        return out


def parameter_options(algorithm):
    if algorithm == 'HGB':
        return [{}]
    if algorithm == 'CatBoost':
        return [dict(depth=4, iterations=250), dict(depth=6, iterations=350)]
    if algorithm == 'XGBoost':
        return [dict(max_depth=3, n_estimators=250), dict(max_depth=5, n_estimators=350)]
    if algorithm == 'LightGBM':
        return [dict(num_leaves=15, n_estimators=250), dict(num_leaves=31, n_estimators=350)]
    raise ValueError('Unknown algorithm')


def comparison_pipeline(algorithm, feature_set, parameters=None):
    groups = FEATURE_SETS[feature_set]
    parameters = {} if parameters is None else parameters
    if algorithm == 'HGB':
        return experiment_pipeline(groups)
    if algorithm == 'XGBoost':
        from xgboost import XGBClassifier
        model = XGBClassifier(learning_rate=.05, tree_method='hist',
                              objective='binary:logistic', eval_metric='logloss',
                              n_jobs=1, random_state=42, **parameters)
    elif algorithm == 'LightGBM':
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(learning_rate=.05, min_child_samples=43,
                               verbosity=-1, n_jobs=1, random_state=42,
                               deterministic=True, force_col_wise=True, **parameters)
    elif algorithm == 'CatBoost':
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(learning_rate=.05, l2_leaf_reg=5,
                                   cat_features=tuple(CATEGORICAL), boosting_type='Ordered',
                                   loss_function='Logloss', random_seed=42,
                                   thread_count=1, verbose=False,
                                   allow_writing_files=False, **parameters)
        return Pipeline([('features', ExtendedFeatures(groups)),
                         ('categories', NativeCategories()), ('model', model)])
    else:
        raise ValueError('Unknown algorithm')
    return Pipeline([('features', ExtendedFeatures(groups)),
                     ('prep', build_preprocessor('target')), ('model', model)])


def choose_threshold(y, probability, baseline):
    scores = [(t, metrics(y, probability, t)) for t in THRESHOLDS]
    allowed = [(t, s) for t, s in scores if s['failure_recall'] >= baseline['failure_recall'] - .01]
    # Retain .5 if none meets the failure-recall constraint; record failure later.
    if not allowed:
        return .5
    return max(allowed, key=lambda item: (item[1]['accuracy'], -abs(item[0] - .5)))[0]


def select_inner(X, y, feature_set, seed):
    splits = list(StratifiedKFold(2, shuffle=True, random_state=seed).split(X, y))
    selected, records = {}, []
    for algorithm in ALGORITHMS:
        options = []
        for option, parameters in enumerate(parameter_options(algorithm)):
            start = time.perf_counter()
            p = np.full(len(y), np.nan)
            for fit, valid in splits:
                model = comparison_pipeline(algorithm, feature_set, parameters)
                model.fit(X.iloc[fit], y.iloc[fit])
                p[valid] = model.predict_proba(X.iloc[valid])[:, 1]
            score = metrics(y, p)
            options.append((option, parameters, p, score))
            records.append(dict(algorithm=algorithm, option=option, parameters=str(parameters),
                                inner_seconds=time.perf_counter()-start, **score))
        option, parameters, p, score = max(options, key=lambda item: item[3]['roc_auc'])
        if algorithm == 'HGB':
            baseline = score
            threshold = .5
        else:
            threshold = choose_threshold(y, p, baseline)
        selected[algorithm] = dict(parameters=parameters, option=option, threshold=threshold,
                                   inner_metrics=metrics(y, p, threshold))
        for record in records:
            if record['algorithm'] == algorithm:
                record['selected'] = record['option'] == option
                record['selected_threshold'] = threshold if record['selected'] else np.nan
    # Family selection also stays inside the outer training partition.
    allowed = [a for a in ALGORITHMS[1:] if eligible(selected[a]['inner_metrics'], baseline)]
    winner = max(allowed, key=lambda a: selected[a]['inner_metrics']['accuracy']) if allowed else 'HGB'
    return selected, winner, pd.DataFrame(records)


def compare_algorithms(X, y):
    outer = list(StratifiedKFold(3, shuffle=True, random_state=20261027).split(X, y))
    records, inner_records, pooled = [], [], []
    for feature_set in FEATURE_SETS:
        probabilities = {a: np.empty(len(y)) for a in (*ALGORITHMS, 'inner_family_selection')}
        thresholds = {a: np.full(len(y), .5) for a in probabilities}
        for fold, (fit, valid) in enumerate(outer, 1):
            selection_start = time.perf_counter()
            selected, winner, inner = select_inner(X.iloc[fit], y.iloc[fit], feature_set, 100+fold)
            selection_seconds = time.perf_counter()-selection_start
            inner_records.append(inner.assign(feature_set=feature_set, fold=fold))
            cache = {}
            for algorithm in ALGORITHMS:
                choice = selected[algorithm]
                start = time.perf_counter()
                model = comparison_pipeline(algorithm, feature_set, choice['parameters'])
                model.fit(X.iloc[fit], y.iloc[fit])
                fit_seconds = time.perf_counter()-start
                start = time.perf_counter()
                p = model.predict_proba(X.iloc[valid])[:, 1]
                predict_seconds = time.perf_counter()-start
                probabilities[algorithm][valid] = p
                thresholds[algorithm][valid] = choice['threshold']
                cache[algorithm] = (p, fit_seconds, predict_seconds)
                for policy, threshold in [('fixed_0.5', .5), ('inner_threshold', choice['threshold'])]:
                    records.append(dict(feature_set=feature_set, fold=fold, algorithm=algorithm,
                                        policy=policy, threshold=threshold, option=choice['option'],
                                        parameters=str(choice['parameters']), fit_seconds=fit_seconds,
                                        predict_seconds=predict_seconds, selection_seconds=selection_seconds,
                                        **metrics(y.iloc[valid], p, threshold)))
            p, fit_seconds, predict_seconds = cache[winner]
            probabilities['inner_family_selection'][valid] = p
            thresholds['inner_family_selection'][valid] = selected[winner]['threshold']
            records.append(dict(feature_set=feature_set, fold=fold, algorithm='inner_family_selection',
                                policy='inner_threshold', selected_algorithm=winner,
                                threshold=selected[winner]['threshold'], fit_seconds=fit_seconds,
                                predict_seconds=predict_seconds, selection_seconds=selection_seconds,
                                **metrics(y.iloc[valid], p, selected[winner]['threshold'])))
            print(f'{feature_set}: outer fold {fold}/3 complete; inner selected {winner}', flush=True)
        for algorithm, p in probabilities.items():
            assert np.isfinite(p).all()
            for policy, threshold in [('fixed_0.5', .5), ('inner_threshold', thresholds[algorithm])]:
                if algorithm == 'inner_family_selection' and policy == 'fixed_0.5':
                    continue
                pooled.append(dict(feature_set=feature_set, algorithm=algorithm, policy=policy,
                                   **metrics(y, p, threshold)))
    folds = pd.DataFrame(records)
    summary = pd.DataFrame(pooled).merge(folds.groupby(['feature_set', 'algorithm', 'policy'])[
        ['fit_seconds', 'predict_seconds', 'selection_seconds']].mean().reset_index(),
        on=['feature_set', 'algorithm', 'policy'])
    baseline = folds[folds.algorithm.eq('HGB')][['feature_set', 'fold', 'policy', 'accuracy', 'roc_auc', 'failure_recall']]
    paired = folds.merge(baseline, on=['feature_set', 'fold', 'policy'], suffixes=('', '_hgb'))
    for metric in ['accuracy', 'roc_auc', 'failure_recall']:
        paired[metric+'_delta'] = paired[metric]-paired[metric+'_hgb']
    return summary, folds, pd.concat(inner_records, ignore_index=True), paired

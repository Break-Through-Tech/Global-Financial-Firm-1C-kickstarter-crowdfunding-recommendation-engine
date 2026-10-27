"""Small training-only nested comparison and final artifact preparation."""
from datetime import datetime, timezone
import hashlib
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from utils.us_modeling import CONSERVATIVE_POLICY, conservative_pipeline

INPUTS = ['goal', 'duration', 'category', 'subcategory', 'location']
THRESHOLD = 0.5
LIMITATIONS = (
    'Training-only restricted-candidate nested CV; candidates informed by prior tuning. '
    'Not a nested replay of the original searches or an independent holdout. '
    'Historical exposure of 1,486 reserved U.S. IDs; no reserved evaluation performed. '
    'Listed state/DC is a location proxy. Launch availability, duration meaning, '
    'snapshot provenance and goal currency remain unverified. Probabilities are uncalibrated.'
)


def candidates():
    """Reuse existing settings; do not run another randomized search."""
    return {
        'HGB baseline': conservative_pipeline(HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.06, random_state=42), 'target'),
        'HGB tuned': conservative_pipeline(HistGradientBoostingClassifier(
            l2_regularization=0.0026915045092098205, learning_rate=0.04318571864689941,
            max_iter=219, max_leaf_nodes=14, min_samples_leaf=43, random_state=42), 'target'),
        'Logistic baseline': conservative_pipeline(LogisticRegression(
            max_iter=2000, solver='lbfgs'), 'onehot'),
        'Logistic tuned': conservative_pipeline(LogisticRegression(
            C=0.16949768237884735, max_iter=2500, solver='lbfgs'), 'onehot'),
        'RF reported': conservative_pipeline(RandomForestClassifier(
            n_estimators=257, max_depth=10, min_samples_leaf=3, min_samples_split=9,
            max_features='sqrt', n_jobs=1, random_state=42), 'target'),
        'RF saved winner': conservative_pipeline(RandomForestClassifier(
            n_estimators=193, max_depth=10, min_samples_leaf=19, min_samples_split=11,
            max_features='sqrt', class_weight='balanced', n_jobs=1, random_state=42), 'target'),
        'ExtraTrees reference': conservative_pipeline(ExtraTreesClassifier(
            n_estimators=250, min_samples_leaf=5, n_jobs=1, random_state=42), 'target'),
        'ExtraTrees tuned': conservative_pipeline(ExtraTreesClassifier(
            n_estimators=174, max_depth=16, min_samples_leaf=20, min_samples_split=8,
            max_features=0.5, n_jobs=1, random_state=42), 'target'),
    }


def metrics(y, probability, threshold=THRESHOLD):
    prediction = np.asarray(probability) >= threshold
    precision, recall, f1, support = precision_recall_fscore_support(
        y, prediction, labels=[0, 1], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    result = dict(accuracy=float(accuracy_score(y, prediction)),
                  roc_auc=float(roc_auc_score(y, probability)),
                  brier=float(brier_score_loss(y, probability)),
                  tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    for index, label in enumerate(['failure', 'success']):
        for name, values in [('precision', precision), ('recall', recall),
                             ('f1', f1), ('support', support)]:
            result[f'{label}_{name}'] = float(values[index])
    return result


def nested_comparison(X, y):
    """3 outer folds, 2 inner folds; preprocessing refitted on every fit."""
    estimators = candidates()
    families = ['HGB', 'Logistic', 'RF', 'ExtraTrees']
    probabilities = {name: np.full(len(y), np.nan) for name in families + ['Selection']}
    records = []
    outer = StratifiedKFold(3, shuffle=True, random_state=20261027)
    for fold, (train, valid) in enumerate(outer.split(X, y), 1):
        inner = StratifiedKFold(2, shuffle=True, random_state=100 + fold)
        scores = {name: float(cross_val_score(model, X.iloc[train], y.iloc[train],
                  cv=inner, scoring='roc_auc', n_jobs=1, error_score='raise').mean())
                  for name, model in estimators.items()}
        overall = max(scores, key=scores.get)
        for family in families:
            winner = max((n for n in scores if n.startswith(family)), key=scores.get)
            start = time.perf_counter()
            fitted = clone(estimators[winner]).fit(X.iloc[train], y.iloc[train])
            fit_seconds = time.perf_counter() - start
            start = time.perf_counter()
            p = fitted.predict_proba(X.iloc[valid])[:, 1]
            predict_seconds = time.perf_counter() - start
            probabilities[family][valid] = p
            if winner == overall:
                probabilities['Selection'][valid] = p
            records.append(dict(fold=fold, family=family, candidate=winner,
                inner_auc=scores[winner], overall_selected=winner == overall,
                fit_seconds=fit_seconds, predict_seconds=predict_seconds,
                validation_rows=len(valid), **metrics(y.iloc[valid], p)))
        print(f'Completed outer fold {fold}/3', flush=True)
    assert all(np.isfinite(p).all() for p in probabilities.values())
    summary = pd.DataFrame({name: metrics(y, p) for name, p in probabilities.items()}).T
    return summary, pd.DataFrame(records), probabilities


def error_breakdowns(X, y, probability, min_rows=100, min_class=20):
    groups = {
        'goal': pd.cut(X.goal, [0, 1000, 5000, 10000, 50000, np.inf], include_lowest=True),
        'duration': pd.cut(X.duration, [0, 15, 30, 45, 60, 90, np.inf]),
        'category': X.category.fillna('missing'),
    }
    records = []
    for feature, values in groups.items():
        for group in values.dropna().unique():
            mask = values.eq(group).to_numpy()
            counts = y[mask].value_counts()
            if mask.sum() >= min_rows and counts.get(0, 0) >= min_class and counts.get(1, 0) >= min_class:
                records.append(dict(feature=feature, group=str(group), n=int(mask.sum()),
                                    **metrics(y[mask], probability[mask])))
    return pd.DataFrame(records)


def save_primary(X, y, ids, summary, folds, root):
    """Select settings by full-training inner CV, refit, save, and verify round trip."""
    estimators = candidates()
    inner = StratifiedKFold(2, shuffle=True, random_state=100)
    scores = {name: float(cross_val_score(model, X, y, cv=inner, scoring='roc_auc',
              n_jobs=1, error_score='raise').mean()) for name, model in estimators.items()}
    decision = json.loads((Path(root) / 'models/us_accuracy_decision.json').read_text())
    from utils.us_accuracy import focused_candidates
    name = 'HGB tuned' if decision['final_candidate'] == 'original' else 'HGB ' + decision['final_candidate']
    selected = focused_candidates()[decision['final_candidate']]
    threshold = float(decision['final_threshold'])
    start = time.perf_counter()
    model = clone(selected).fit(X, y)
    fit_seconds = time.perf_counter() - start
    folder = Path(root) / 'models'
    folder.mkdir(exist_ok=True)
    path = folder / 'us_primary_model_v1.joblib'
    joblib.dump(model, path)
    loaded = joblib.load(path)
    before = model.predict_proba(X)
    np.testing.assert_array_equal(before, loaded.predict_proba(X))
    np.testing.assert_array_equal(model.predict(X), loaded.predict(X))
    metadata = dict(model_version='us_primary_model_v1', selected_candidate=name,
        created_utc=datetime.now(timezone.utc).isoformat(),
        accuracy_experiment=decision,
        primary_validation_metrics=(decision['focused_metrics'] if decision['accept_improvement'] else decision['original_metrics']),
        estimator_class=type(model.named_steps['model']).__name__,
        estimator_parameters=model.named_steps['model'].get_params(),
        preprocessing_parameters=dict(
            categorical_strategy='target', target_encoder=dict(smoothing=50, n_splits=5, random_state=42),
            numeric_imputation='median', numeric_scaling='StandardScaler',
            feature_builder_parameters=model.named_steps['features'].get_params(),
            transformed_features=list(model.named_steps['prep'].get_feature_names_out())),
        conservative_policy=CONSERVATIVE_POLICY,
        input_schema={c: ('positive finite number' if c in ['goal', 'duration']
                         else 'nonempty string; location must have supported state/DC suffix') for c in INPUTS},
        input_order=INPUTS, training_rows=len(X), success_rows=int(y.sum()),
        training_cohort='official U.S. training IDs in saved order',
        training_ids_sha256=hashlib.sha256('\n'.join(ids.astype(str)).encode()).hexdigest(),
        threshold=threshold, threshold_policy='Original 0.5 retained unless focused nested accuracy procedure passes prespecified promotion guardrails',
        validation_procedure='3 outer x 2 inner stratified restricted-candidate nested CV',
        validation_metrics=summary.to_dict(orient='index'),
        outer_fold_records=folds.to_dict(orient='records'),
        full_training_inner_selection_auc=scores, final_fit_seconds=fit_seconds,
        environment=dict(python=platform.python_version(), sklearn=sklearn.__version__,
                         numpy=np.__version__, pandas=pd.__version__, joblib=joblib.__version__),
        artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        serialization_verified_rows=len(X), limitations=LIMITATIONS)
    path.with_suffix('.metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    return model, metadata

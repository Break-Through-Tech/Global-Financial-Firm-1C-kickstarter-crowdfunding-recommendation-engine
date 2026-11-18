"""Separate training-only study. Does not change the app or production artifacts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'LOKY_MAX_CPU_COUNT']:
    os.environ[variable] = '1'

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

from scripts.prepare_historical_models import historical_pipeline, load_training_rows
from utils.historical_prediction import location_frame
from utils.us_extended_features import city_keys
from utils.us_modeling import load_us_training_data
from utils.us_prediction import campaign_frame, load_primary_predictor
from utils.us_validation import metrics

ALGORITHMS = ['Logistic', 'HGB', 'XGBoost', 'CatBoost', 'LightGBM']
THRESHOLD = .5
CITY_HASH = '5206dcd826620b9e701a4bc06ee1b669a7385916ac3199dcea3fd839062b4e5e'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected_snapshot(root):
    names = ['app.py', 'requirements.txt', 'utils/us_prediction.py', 'utils/historical_prediction.py',
             'utils/international_scope.py', 'utils/us_extended_features.py', 'utils/preprocessing.py',
             'utils/us_validation.py', 'scripts/prepare_historical_models.py',
             'scripts/run_scope_improvement_study.py']
    names += [str(p.relative_to(root)) for p in (root / 'models').glob('*.joblib')]
    names += [str(p.relative_to(root)) for p in (root / 'models').glob('*.metadata.json')]
    return {name: digest(root / name) for name in sorted(names)}


def candidate_pipeline(algorithm):
    pipe = historical_pipeline()
    if algorithm == 'Logistic':
        return pipe
    if algorithm == 'HGB':
        estimator = HistGradientBoostingClassifier(max_iter=160, learning_rate=.06, random_state=42)
    elif algorithm == 'XGBoost':
        from xgboost import XGBClassifier
        estimator = XGBClassifier(n_estimators=250, max_depth=3, learning_rate=.05,
                                 tree_method='hist', objective='binary:logistic',
                                 eval_metric='logloss', n_jobs=1, random_state=42)
    elif algorithm == 'CatBoost':
        from catboost import CatBoostClassifier
        estimator = CatBoostClassifier(iterations=250, depth=4, learning_rate=.05,
                                       l2_leaf_reg=5, boosting_type='Ordered', loss_function='Logloss',
                                       random_seed=42, thread_count=1, verbose=False, allow_writing_files=False)
    elif algorithm == 'LightGBM':
        from lightgbm import LGBMClassifier
        estimator = LGBMClassifier(n_estimators=250, num_leaves=15, learning_rate=.05,
                                   min_child_samples=43, verbosity=-1, n_jobs=1,
                                   random_state=42, deterministic=True, force_col_wise=True)
    else:
        raise ValueError('Unknown candidate algorithm')
    return pipe.set_params(model=estimator)


def eligible(candidate, baseline, majority):
    return (candidate['accuracy'] >= baseline['accuracy'] + .003
            and candidate['accuracy'] >= majority + .003
            and candidate['roc_auc'] >= baseline['roc_auc'] - .002
            and candidate['failure_recall'] >= baseline['failure_recall'] - .01)


def choose_family(scores, majority):
    allowed = [name for name in ALGORITHMS[1:] if eligible(scores[name], scores['Logistic'], majority)]
    return max(allowed, key=lambda name: (scores[name]['accuracy'], scores[name]['roc_auc'])) if allowed else 'Logistic'


def compare_scope(X, y, scope):
    """Common five outer folds; two inner folds select families at fixed .5."""
    outer = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
    names = [*ALGORITHMS, 'Majority', 'Inner family selection']
    probabilities = {name: np.full(len(y), np.nan) for name in names}
    records, inner_records, split_records = [], [], []
    for fold, (fit, valid) in enumerate(outer, 1):
        inner = list(StratifiedKFold(2, shuffle=True, random_state=100 + fold).split(X.iloc[fit], y.iloc[fit]))
        scores = {}
        for algorithm in ALGORITHMS:
            p = np.full(len(fit), np.nan)
            for train, check in inner:
                model = candidate_pipeline(algorithm)
                model.fit(X.iloc[fit].iloc[train], y.iloc[fit].iloc[train])
                p[check] = model.predict_proba(X.iloc[fit].iloc[check])[:, 1]
            scores[algorithm] = metrics(y.iloc[fit], p)
            inner_records.append(dict(scope=scope, fold=fold, algorithm=algorithm, **scores[algorithm]))
        majority = float(y.iloc[fit].value_counts(normalize=True).max())
        winner = choose_family(scores, majority)
        timings = {}
        for algorithm in [*ALGORITHMS, 'Majority']:
            model = (historical_pipeline().set_params(model=DummyClassifier(strategy='prior'))
                     if algorithm == 'Majority' else candidate_pipeline(algorithm))
            start = time.perf_counter()
            model.fit(X.iloc[fit], y.iloc[fit])
            fit_seconds = time.perf_counter() - start
            start = time.perf_counter()
            p = model.predict_proba(X.iloc[valid])[:, 1]
            predict_seconds = time.perf_counter() - start
            probabilities[algorithm][valid] = p
            timings[algorithm] = (fit_seconds, predict_seconds)
            records.append(dict(scope=scope, fold=fold, algorithm=algorithm, selected_algorithm=winner,
                                fit_seconds=fit_seconds, predict_seconds=predict_seconds,
                                **metrics(y.iloc[valid], p)))
        probabilities['Inner family selection'][valid] = probabilities[winner][valid]
        records.append(dict(scope=scope, fold=fold, algorithm='Inner family selection', selected_algorithm=winner,
                            fit_seconds=timings[winner][0], predict_seconds=timings[winner][1],
                            **metrics(y.iloc[valid], probabilities[winner][valid])))
        split_records.append(dict(scope=scope, fold=fold, fit_rows=len(fit), validation_rows=len(valid),
                                  fitting_positions_sha256=hashlib.sha256(fit.astype('<i8').tobytes()).hexdigest(),
                                  validation_positions_sha256=hashlib.sha256(valid.astype('<i8').tobytes()).hexdigest()))
        print(f'{scope}: fold {fold}/5 complete; inner selected {winner}', flush=True)
    folds = pd.DataFrame(records)
    summary = []
    baseline = metrics(y, probabilities['Logistic'])
    majority = metrics(y, probabilities['Majority'])['accuracy']
    baseline_folds = folds[folds.algorithm.eq('Logistic')].set_index('fold')
    for algorithm, p in probabilities.items():
        if not np.isfinite(p).all():
            raise ValueError('Incomplete out-of-fold probabilities')
        score = metrics(y, p)
        selected = folds[folds.algorithm.eq(algorithm)].set_index('fold')
        positive = int((selected.accuracy > baseline_folds.accuracy).sum())
        summary.append(dict(scope=scope, algorithm=algorithm, training_rows=len(y),
                            accuracy_delta=score['accuracy'] - baseline['accuracy'],
                            roc_auc_delta=score['roc_auc'] - baseline['roc_auc'],
                            failure_recall_delta=score['failure_recall'] - baseline['failure_recall'],
                            positive_accuracy_folds=positive,
                            eligible_for_review=eligible(score, baseline, majority) and positive >= 3,
                            fit_seconds=float(selected.fit_seconds.mean()),
                            predict_seconds=float(selected.predict_seconds.mean()), **score))
    return pd.DataFrame(summary), folds, pd.DataFrame(inner_records), pd.DataFrame(split_records)


def city_campaign_frame(campaign):
    if not isinstance(campaign.get('location'), str) or ',' not in campaign['location']:
        raise ValueError('City candidate requires a full City, ST location')
    return campaign_frame(campaign)


def audit_city(root):
    path = root / 'models/us_candidate_v2.joblib'
    metadata = json.loads(path.with_suffix('.metadata.json').read_text())
    if digest(path) != CITY_HASH or metadata['artifact_sha256'] != CITY_HASH:
        raise ValueError('City candidate identity changed')
    primary = load_primary_predictor(root / 'models/us_primary_model_v1.joblib')
    candidate = joblib.load(path)
    X, _, _ = load_us_training_data(root)
    sample = X.iloc[np.linspace(0, len(X) - 1, 100, dtype=int)]
    normalized = pd.concat([city_campaign_frame(row.to_dict()) for _, row in sample.iterrows()], ignore_index=True)
    normalized.index = sample.index
    raw_p = candidate.predict_proba(sample)[:, 1]
    interface_p = candidate.predict_proba(normalized)[:, 1]
    seen = city_keys(normalized).map(candidate.named_steps['features'].city_counts_).fillna(0)
    primary_result = primary.predict(sample.iloc[0].to_dict())
    return dict(artifact_sha256=CITY_HASH, validation_metrics=metadata['validation_metrics'],
                sample_rows=len(sample), probability_difference_raw_vs_validated=float(np.max(np.abs(raw_p-interface_p))),
                sample_cities_seen_in_training=int(seen.gt(0).sum()),
                primary_class_mapping_valid=primary_result['predicted_class'] in [0, 1],
                required_input='Full listed City, ST. State-only input rejected for candidate.',
                promotion='Not activated. Existing development gain is not independent test evidence.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/scope_improvement_results')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output directory; do not overwrite an existing study')
    before = protected_snapshot(ROOT)
    args.output.mkdir(parents=True)
    import xgboost, catboost, lightgbm
    protocol = dict(scopes=['international', 'mixed'], algorithms=ALGORITHMS,
                    inputs=['duration', 'category', 'subcategory', 'country_region'],
                    outer_cv='5-fold stratified shuffle seed42; same as current historical baselines',
                    inner_cv='2-fold stratified, seed100+outerfold; family selection on outer fitting rows only',
                    threshold=THRESHOLD, minimum_accuracy_gain=.003, maximum_auc_loss=.002,
                    maximum_failure_recall_loss=.01, minimum_positive_outer_folds=3,
                    require_accuracy_above_majority_by=.003,
                    features='No monetary, title/reward, outcome or inferred timing features; identical fold-fitted preprocessing for all algorithms',
                    settings={name: candidate_pipeline(name).named_steps['model'].get_params() for name in ALGORITHMS},
                    environment=dict(python=platform.python_version(), sklearn=sklearn.__version__,
                                     numpy=np.__version__, pandas=pd.__version__, joblib=joblib.__version__,
                                     xgboost=xgboost.__version__, catboost=catboost.__version__, lightgbm=lightgbm.__version__),
                    protected_artifacts=before,
                    limitations='Development-only; prior exposure to official training folds. No independent superiority claim or reserved evaluation.')
    (args.output / 'protocol.txt').write_text(json.dumps(protocol, indent=2, default=str) + '\n')
    try:
        (args.output / 'city_candidate_audit.txt').write_text(json.dumps(audit_city(ROOT), indent=2) + '\n')
        results = []
        for scope in protocol['scopes']:
            rows, y, ids = load_training_rows(ROOT, scope)
            X = location_frame(rows)
            membership_hash = hashlib.sha256(ids.to_numpy(dtype='<i8').tobytes()).hexdigest()
            expected = json.loads((ROOT / f'models/{scope}_historical_v1.metadata.json').read_text())
            if membership_hash != expected['training_ids_sha256']:
                raise ValueError('Current baseline and experiment training membership differ')
            (args.output / f'{scope}_cohort.txt').write_text(json.dumps(dict(
                scope=scope, training_rows=len(X), ordered_training_ids_sha256=membership_hash,
                inputs=list(X), class_counts=y.value_counts().sort_index().astype(int).to_dict()), indent=2) + '\n')
            summary, folds, inner, splits = compare_scope(X, y, scope)
            baseline = summary[summary.algorithm.eq('Logistic')].iloc[0]
            for metric in ['accuracy', 'roc_auc', 'brier', 'failure_recall']:
                if not np.isclose(baseline[metric], expected['validation_metrics'][metric], atol=1e-12, rtol=0):
                    raise ValueError(f'{scope}: current baseline did not reproduce: {metric}')
            for name, table in [('summary', summary), ('folds', folds), ('inner', inner), ('splits', splits)]:
                table.to_csv(args.output / f'{scope}_{name}.csv', index=False)
            results.append(summary)
            print(summary[['algorithm', 'accuracy', 'roc_auc', 'failure_recall', 'eligible_for_review']].to_string(index=False), flush=True)
        pd.concat(results, ignore_index=True).to_csv(args.output / 'summary.csv', index=False)
    finally:
        after = protected_snapshot(ROOT)
        if after != before:
            raise RuntimeError('Protected source or model artifacts changed during study')
        (args.output / 'integrity.txt').write_text('All protected model, metadata and serving-source hashes unchanged.\n')


if __name__ == '__main__':
    main()

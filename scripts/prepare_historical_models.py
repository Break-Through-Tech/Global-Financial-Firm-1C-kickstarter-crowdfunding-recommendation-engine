"""Fit two fixed historical baselines offline. Never run from Streamlit."""
import hashlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils.historical_prediction import INPUTS, LIMITATION, location_frame
from utils.international_scope import classify_locations
from utils.us_prediction import PRIMARY_SHA256
from utils.us_validation import metrics


def load_training_rows(root, scope):
    folder = root / 'data/Processed-Data'
    if scope == 'international':
        folder /= 'international_only'
    elif scope != 'mixed':
        raise ValueError('Unsupported scope')
    ids = pd.read_csv(folder / 'train_project_ids.csv')
    if not ids['project id'].is_unique or not ids.target.isin([0, 1]).all():
        raise ValueError('Official training IDs or labels are invalid')
    clean = pd.read_csv(folder / 'DSI_kickstarterscrape_dataset_cleaned.csv')
    if not clean['project id'].is_unique:
        raise ValueError('Cleaned project IDs are not unique')
    rows = clean.set_index('project id', drop=False).loc[ids['project id']].reset_index(drop=True)
    y = ids.target.astype(int).reset_index(drop=True)
    if not rows.status.isin(['successful', 'failed']).all() or not np.array_equal(rows.status.eq('successful'), y):
        raise ValueError('Training status and saved labels disagree')
    if len(rows) != {'international': 1427, 'mixed': 29288}[scope]:
        raise ValueError('Official training population changed')
    if scope == 'international':
        parent = pd.read_csv(root / 'data/Processed-Data/train_project_ids.csv').set_index('project id')
        if not np.array_equal(parent.loc[ids['project id'], 'target'], y):
            raise ValueError('International training rows disagree with official global training membership')
        if not classify_locations(rows.location).scope.eq('explicit_non_us').all():
            raise ValueError('International membership contains unsupported locations')
    return rows, y, ids['project id']


def historical_pipeline():
    numeric = Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
    categories = Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value='__missing__')),
        ('encode', OneHotEncoder(handle_unknown='ignore', min_frequency=10, sparse_output=False)),
    ])
    prep = ColumnTransformer([('duration', numeric, ['duration']),
                              ('categories', categories, ['category', 'subcategory', 'country_region'])])
    return Pipeline([('prep', prep), ('model', LogisticRegression(max_iter=2000))])


def main():
    protected = ROOT / 'models/us_primary_model_v1.joblib'
    if hashlib.sha256(protected.read_bytes()).hexdigest() != PRIMARY_SHA256:
        raise ValueError('Frozen U.S. model changed')
    paths = [ROOT / f'models/{scope}_historical_v1.joblib' for scope in ['international', 'mixed']]
    if any(path.exists() or path.with_suffix('.metadata.json').exists() for path in paths):
        raise FileExistsError('Historical artifacts already exist; review rather than overwrite them')
    for scope, path in zip(['international', 'mixed'], paths):
        rows, y, ids = load_training_rows(ROOT, scope)
        X = location_frame(rows)
        pipe = historical_pipeline()
        cv = StratifiedKFold(5, shuffle=True, random_state=42)
        probabilities = cross_val_predict(pipe, X, y, cv=cv, method='predict_proba', n_jobs=1)[:, 1]
        result = metrics(y, probabilities)
        pipe.fit(X, y)
        before = pipe.predict_proba(X.iloc[:20])
        joblib.dump(pipe, path)
        np.testing.assert_array_equal(before, joblib.load(path).predict_proba(X.iloc[:20]))
        metadata = dict(
            model_version=f'{scope}_historical_v1', scope=scope,
            input_order=INPUTS, uses_funding_goal=False, threshold=.5,
            estimator='LogisticRegression(max_iter=2000); fixed existing baseline, no search',
            training_rows=len(X), majority_class_accuracy=float(y.value_counts(normalize=True).max()),
            raw_input_order=['duration', 'category', 'subcategory', 'location'],
            training_cohort=f'Official {scope} training IDs; saved order; reserved labels not evaluated',
            training_ids_sha256=hashlib.sha256(ids.to_numpy(dtype='<i8').tobytes()).hexdigest(),
            validation='Five-fold stratified CV, shuffle=True, random_state=42; fold-fitted preprocessing',
            validation_metrics=result, model_selection='Single fixed baseline; no tuning or threshold selection',
            country_training_counts=X.country_region.value_counts().astype(int).to_dict(),
            training_categories={name: sorted(X[name].dropna().unique().tolist()) for name in ['category', 'subcategory']},
            environment=dict(python=platform.python_version(), sklearn=sklearn.__version__,
                             numpy=np.__version__, pandas=pd.__version__, joblib=joblib.__version__),
            limitations=LIMITATION + ' Random folds do not test temporal generalization. Reserved cohorts have historical exposure.',
            artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        path.with_suffix('.metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        print(scope, metadata['artifact_sha256'], result, flush=True)
    assert hashlib.sha256(protected.read_bytes()).hexdigest() == PRIMARY_SHA256


if __name__ == '__main__':
    main()

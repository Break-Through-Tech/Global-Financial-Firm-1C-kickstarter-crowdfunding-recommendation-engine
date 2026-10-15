"""Regenerate namespaced base preprocessors from official training rows only.

Run at repository root: .venv/bin/python docs/audit_evidence/compatible_artifacts.py
Original artifacts/matrices/IDs are never overwritten; reserved labels are not read.
These base artifacts include levels and are NOT the conservative 05 model pipeline.
"""
from pathlib import Path
import hashlib
import json
import platform
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from utils.international_scope import classify_locations


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    namespace = f"compat/python-{platform.python_version()}_sklearn-{sklearn.__version__}"
    reports = []
    for cohort in ['', 'us_only', 'international_only']:
        folder = ROOT / 'data/Processed-Data' / cohort
        clean = pd.read_csv(folder / 'DSI_kickstarterscrape_dataset_cleaned.csv')
        ids = pd.read_csv(folder / 'train_project_ids.csv')
        assert ids['project id'].is_unique and clean['project id'].is_unique
        rows = clean.set_index('project id', drop=False).loc[ids['project id']].reset_index(drop=True)
        cats = ['category', 'subcategory']
        if cohort == 'international_only':
            geo = classify_locations(rows.location)
            assert geo.scope.eq('explicit_non_us').all()
            rows['country_region'] = geo.country_region
            cats += ['country_region']
        nums = ['goal', 'levels', 'duration']
        if not cohort:
            prep = ColumnTransformer([
                ('categorical', Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cats),
                ('numerical', Pipeline([('scaler', StandardScaler())]), nums),
            ], remainder='drop')
        else:
            num_name, cat_name, encode_name = ('num', 'cat', 'onehot') if cohort == 'international_only' else ('numerical', 'categorical', 'encode')
            prep = ColumnTransformer([
                (num_name, Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), nums),
                (cat_name, Pipeline([('impute', SimpleImputer(strategy='constant', fill_value='missing')),
                                    (encode_name, OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cats),
            ], verbose_feature_names_out=True)
        actual = prep.fit_transform(rows)
        saved_train = pd.read_csv(folder / 'X_train_base_preprocessed.csv')
        saved_test = pd.read_csv(folder / 'X_test_base_preprocessed.csv')
        assert saved_train.columns.tolist() == prep.get_feature_names_out().tolist()
        assert saved_train.columns.equals(saved_test.columns)
        assert actual.shape == saved_train.shape
        assert np.isfinite(actual).all() and np.isfinite(saved_train.to_numpy()).all() and np.isfinite(saved_test.to_numpy()).all()
        np.testing.assert_allclose(actual, saved_train.to_numpy(), rtol=1e-10, atol=1e-10)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                old = joblib.load(folder / 'base_preprocessor.joblib')
                np.testing.assert_allclose(old.transform(rows), actual)
                legacy = 'loads; training transform matches'
            except (AttributeError, ValueError) as exc:
                legacy = repr(exc)
        dest = folder / namespace
        dest.mkdir(parents=True, exist_ok=True)
        artifact = dest / 'base_preprocessor.joblib'
        if not artifact.exists():
            joblib.dump(prep, artifact)
        restored = joblib.load(artifact)
        np.testing.assert_allclose(restored.transform(rows), actual, rtol=1e-10, atol=1e-10)
        report = dict(cohort=cohort or 'combined', python=platform.python_version(), sklearn=sklearn.__version__,
                      numpy=np.__version__, pandas=pd.__version__, legacy_load=legacy,
                      legacy_warnings=[str(w.message) for w in caught],
                      train_shape=list(actual.shape), test_shape=list(saved_test.shape),
                      artifact=str(artifact.relative_to(ROOT)), artifact_sha256=digest(artifact),
                      input_sha256={name:digest(folder/name) for name in ['DSI_kickstarterscrape_dataset_cleaned.csv',
                                      'train_project_ids.csv', 'X_train_base_preprocessed.csv']},
                      script_sha256=digest(Path(__file__)),
                      smoke='PASS load and transform; matrix values and ordered columns match')
        # Never overwrite earlier metadata for an existing version namespace.
        metadata = dest / 'manifest.json'
        if not metadata.exists():
            metadata.write_text(json.dumps(report, indent=2)+'\n')
        reports.append(report)
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()

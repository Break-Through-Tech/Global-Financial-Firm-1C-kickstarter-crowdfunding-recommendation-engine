"""Saved-model interface used by the Streamlit app."""
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from utils.us_scope import US_STATE_CODES
from utils.us_modeling import repository_root
from utils.us_validation import INPUTS


def campaign_frame(campaign):
    """Validate five launch inputs; accept a state code or city/state location."""
    missing = [name for name in INPUTS if name not in campaign]
    if missing:
        raise ValueError(f'Missing campaign inputs: {missing}')
    row = {name: campaign[name] for name in INPUTS}
    for name in ['goal', 'duration']:
        if isinstance(row[name], (bool, str)):
            raise ValueError(f'{name} must be a positive finite number')
        try:
            row[name] = float(row[name])
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f'{name} must be a positive finite number') from None
        if not np.isfinite(row[name]) or row[name] <= 0:
            raise ValueError(f'{name} must be a positive finite number')
    for name in ['category', 'subcategory', 'location']:
        if not isinstance(row[name], str) or not row[name].strip():
            raise ValueError(f'{name} must be a nonempty string')
        row[name] = row[name].strip()
    state = row['location'].split(',')[-1].strip().upper()
    if state not in US_STATE_CODES or (',' not in row['location'] and row['location'].upper() != state):
        raise ValueError('location must be a supported U.S. state/DC code or city, ST')
    if ',' in row['location'] and not row['location'].rsplit(',', 1)[0].strip():
        raise ValueError('location must include a city before the state suffix')
    row['location'] = f'State, {state}' if ',' not in row['location'] else row['location'].rsplit(',', 1)[0] + f', {state}'
    return pd.DataFrame([row], columns=INPUTS)


class USPredictor:
    """Load once and reuse. Only load trusted local joblib artifacts."""

    def __init__(self, model_path):
        path = Path(model_path)
        self.metadata = json.loads(path.with_suffix('.metadata.json').read_text())
        self.threshold = float(self.metadata['threshold'])
        if not 0 < self.threshold < 1:
            raise ValueError('Invalid classification threshold')
        self.pipeline = joblib.load(path)
        if list(self.pipeline.classes_) != [0, 1]:
            raise ValueError('Expected class order [failure=0, success=1]')

    def predict(self, campaign):
        probability = float(self.pipeline.predict_proba(campaign_frame(campaign))[0, 1])
        success = probability >= self.threshold
        return dict(success_probability=probability, predicted_class=int(success),
                    classification='successful' if success else 'failed', threshold=self.threshold)


def predict_campaign(campaign, model_path=None):
    """Predict one campaign; use USPredictor to reuse a loaded model in Streamlit."""
    if model_path is None:
        model_path = repository_root() / 'models/us_primary_model_v1.joblib'
    return USPredictor(model_path).predict(campaign)


PRIMARY_SHA256 = '243bb0d53ac7b456e6b636c424bc8d0d74dbd67f014d5f8f175b7d4226bd94b6'


def load_primary_predictor(model_path):
    """Verify the frozen public-app contract before loading trusted pickle bytes."""
    path = Path(model_path)
    metadata = json.loads(path.with_suffix('.metadata.json').read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != PRIMARY_SHA256:
        raise ValueError('Frozen U.S. model integrity check failed')
    if (metadata.get('artifact_sha256') != PRIMARY_SHA256
            or metadata.get('model_version') != 'us_primary_model_v1'
            or metadata.get('input_order') != list(INPUTS)
            or metadata.get('threshold') != 0.5
            or metadata.get('estimator_class') != 'HistGradientBoostingClassifier'):
        raise ValueError('Frozen U.S. model metadata does not match')
    versions = dict(python=platform.python_version(), sklearn=sklearn.__version__,
                    numpy=np.__version__, pandas=pd.__version__, joblib=joblib.__version__)
    for name, version in versions.items():
        expected = metadata['environment'][name]
        if name == 'python':
            required_python = '.'.join(expected.split('.')[:2])
            if version.split('.')[:2] != expected.split('.')[:2]:
                raise ValueError(
                    f'The server is running Python {version}; the frozen model requires '
                    f'Python {required_python}. In Streamlit Community Cloud, recreate '
                    f'the deployment with Python {required_python} selected in Advanced settings.'
                )
            continue
        if version != expected:
            raise ValueError(
                f'The server has {name} {version}; the frozen model requires {expected}. '
                'Check that the deployed branch uses the pinned requirements.txt.'
            )
    predictor = USPredictor(path)
    steps = predictor.pipeline.named_steps
    if (list(steps) != ['features', 'prep', 'model']
            or type(steps['model']).__name__ != 'HistGradientBoostingClassifier'
            or list(steps['prep'].get_feature_names_out()) != metadata['preprocessing_parameters']['transformed_features']
            or any(steps['features'].get_params()[name] != value
                   for name, value in metadata['conservative_policy'].items())):
        raise ValueError('Unexpected frozen pipeline schema')
    return predictor

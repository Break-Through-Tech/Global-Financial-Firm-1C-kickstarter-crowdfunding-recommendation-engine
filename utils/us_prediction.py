"""Minimal saved-model interface for the future Streamlit app."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

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

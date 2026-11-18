"""Experimental estimates for historical listed-location cohorts, without money inputs."""
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from utils.international_scope import classify_locations
from utils.us_scope import US_STATE_CODES

INPUTS = ['duration', 'category', 'subcategory', 'country_region']
MODEL_HASHES = {
    'international': 'd6cbe690005c6bb8afff8bb99daf25d8962de9813ea27011bcfd9554bd9106b7',
    'mixed': 'f46014783e219c7a5f98175ce9d355e0c67a7b738dac872626459a4a5347349d',
}
LIMITATION = ('Experimental estimate from 2009–2012 listed-location data, not validated on current '
              'campaigns or international registration markets. Funding amounts are not used. '
              'Probabilities are uncalibrated; source duration and snapshot timing remain uncertain.')


def location_frame(rows):
    """Apply the existing fixed geography rules; learn no statistics."""
    frame = rows[['duration', 'category', 'subcategory']].copy()
    geography = classify_locations(rows['location'])
    frame['country_region'] = geography.country_region.fillna('__unresolved__')
    # Explicit U.S. names and state suffixes describe the same country group.
    frame.loc[geography.scope.isin(['us_candidate', 'explicit_us']), 'country_region'] = 'United States'
    return frame[INPUTS]


def historical_campaign_frame(campaign, scope):
    if scope not in MODEL_HASHES:
        raise ValueError('Unsupported historical model scope')
    row = {}
    duration = campaign.get('duration')
    if isinstance(duration, (str, bool)):
        raise ValueError('duration must be a positive finite number')
    try:
        row['duration'] = float(duration)
    except (TypeError, ValueError, OverflowError):
        raise ValueError('duration must be a positive finite number') from None
    if not np.isfinite(row['duration']) or row['duration'] <= 0:
        raise ValueError('duration must be a positive finite number')
    for name in ['category', 'subcategory', 'location']:
        if not isinstance(campaign.get(name), str) or not campaign[name].strip():
            raise ValueError(f'{name} must be a nonempty string')
        row[name] = campaign[name].strip()
    location = row['location']
    if location.upper() in US_STATE_CODES:
        row['location'] = f'State, {location.upper()}'
    elif location.startswith(','):
        raise ValueError('location must include a city or a full country name')
    geography = classify_locations(pd.Series([row['location']])).iloc[0]
    if scope == 'international' and geography.scope != 'explicit_non_us':
        raise ValueError('Use an explicit named non-U.S. location, such as Berlin, Germany')
    if scope == 'mixed' and geography.scope not in ['us_candidate', 'explicit_us', 'explicit_non_us']:
        raise ValueError('Use a U.S. state/DC code or explicit country name, such as Toronto, Canada')
    return location_frame(pd.DataFrame([row]))


def load_historical_predictor(model_path, scope):
    if scope not in MODEL_HASHES:
        raise ValueError('Unsupported historical model scope')
    path = Path(model_path)
    metadata = json.loads(path.with_suffix('.metadata.json').read_text())
    expected = MODEL_HASHES[scope]
    if not expected or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('Historical model integrity check failed')
    if (metadata.get('artifact_sha256') != expected or metadata.get('scope') != scope
            or metadata.get('model_version') != f'{scope}_historical_v1'
            or metadata.get('input_order') != INPUTS or metadata.get('threshold') != .5
            or metadata.get('uses_funding_goal') is not False):
        raise ValueError('Historical model metadata does not match')
    versions = dict(python=platform.python_version(), sklearn=sklearn.__version__,
                    numpy=np.__version__, pandas=pd.__version__, joblib=joblib.__version__)
    for name, value in versions.items():
        required = metadata['environment'][name]
        if name == 'python':
            value, required = value.split('.')[:2], required.split('.')[:2]
        if value != required:
            raise ValueError(f'Unsupported {name} version for the historical model')
    pipeline = joblib.load(path)
    if (list(pipeline.classes_) != [0, 1] or list(pipeline.feature_names_in_) != INPUTS
            or list(pipeline.named_steps) != ['prep', 'model']):
        raise ValueError('Historical model pipeline is incompatible')
    return pipeline, metadata


def predict_historical(predictor, campaign, scope):
    pipeline, metadata = predictor
    if metadata['scope'] != scope:
        raise ValueError('Selected scope does not match the loaded model')
    frame = historical_campaign_frame(campaign, scope)
    probability = float(pipeline.predict_proba(frame)[0, 1])
    successful = probability >= metadata['threshold']
    warnings = [LIMITATION]
    if metadata['validation_metrics']['accuracy'] < metadata['majority_class_accuracy']:
        warnings.append('This baseline has lower development accuracy than always predicting '
                        'the majority class. Use it for demonstration, not campaign decisions.')
    country = frame.country_region.iloc[0]
    count = metadata['country_training_counts'].get(country, 0)
    if count < 25:
        warnings.append(f'Limited listed-region support: {count} training campaigns for {country}.')
    for name in ['category', 'subcategory']:
        if frame[name].iloc[0] not in metadata['training_categories'][name]:
            warnings.append(f'Unseen {name}: using the fitted encoder fallback.')
    return dict(success_probability=probability, predicted_class=int(successful),
                classification='successful' if successful else 'failed', threshold=metadata['threshold'],
                warnings=warnings)

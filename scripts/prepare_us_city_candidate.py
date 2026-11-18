"""Save the evaluated city-frequency candidate locally without replacing v1."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd
from utils.us_extended_features import eligible, experiment_pipeline
from utils.us_modeling import load_us_training_data


def main():
    results = ROOT / 'docs/launch_feature_results'
    summary = pd.read_csv(results / 'summary.csv', index_col='configuration')
    folds = pd.read_csv(results / 'outer_folds.csv')
    assert eligible(summary.loc['approved_selection'], summary.loc['baseline'])
    paired = folds.pivot(index='fold', columns='configuration', values='accuracy')
    assert ((paired.approved_selection - paired.baseline) > 0).sum() >= 2
    assert folds.loc[folds.configuration.eq('approved_selection'), 'groups'].eq('city_frequency').all()
    X, y, ids = load_us_training_data(ROOT)
    model = experiment_pipeline(('city_frequency',)).fit(X, y)
    path = ROOT / 'models/us_candidate_v2.joblib'
    # Re-execution verifies an existing candidate rather than overwriting it.
    if not path.exists():
        joblib.dump(model, path)
    restored = joblib.load(path)
    np.testing.assert_array_equal(model.predict_proba(X), restored.predict_proba(X))
    np.testing.assert_array_equal(model.predict(X), restored.predict(X))
    metadata = dict(
        model_version='us_candidate_v2', status='experimental; v1 remains primary',
        feature_groups=['city_frequency'], threshold=.5, training_rows=len(X),
        input_contract='goal, duration, category, subcategory, full listed city/state location; state-only input lacks city frequency',
        transformed_features=list(model.named_steps['prep'].get_feature_names_out()),
        estimator_parameters=model.named_steps['model'].get_params(),
        environment=json.loads((results / 'input_review.txt').read_text())['environment'],
        training_ids_sha256=hashlib.sha256('\n'.join(ids.astype(str)).encode()).hexdigest(),
        artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        validation_metrics=summary.loc['approved_selection'].to_dict(),
        serialization_verified_rows=len(X),
        limitations='Training-only restricted nested comparison with prior development exposure; no independent superiority claim. Listed location and launch availability remain assumptions. Not integrated with USPredictor or Streamlit.')
    metadata_path = path.with_suffix('.metadata.json')
    if not metadata_path.exists():
        metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    else:
        previous = json.loads(metadata_path.read_text())
        assert previous['artifact_sha256'] == metadata['artifact_sha256']
        assert previous['training_ids_sha256'] == metadata['training_ids_sha256']
    print('Candidate saved/verified; exact serialization agreement on', len(X), 'rows.')
    print('Artifact SHA-256:', metadata['artifact_sha256'])


if __name__ == '__main__':
    main()

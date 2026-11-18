"""Save eligible mixed candidates under an ignored research directory, never deploy them."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

from scripts.prepare_historical_models import load_training_rows
from scripts.run_scope_improvement_study import candidate_pipeline, protected_snapshot
from utils.historical_prediction import location_frame


def main():
    results = ROOT / 'docs/scope_improvement_results'
    protocol = json.loads((results / 'protocol.txt').read_text())
    if protected_snapshot(ROOT) != protocol['protected_artifacts']:
        raise ValueError('Protected files differ from the completed experiment')
    if not (results / 'integrity.txt').is_file():
        raise FileNotFoundError('Complete and review the comparison first')
    summary = pd.read_csv(results / 'summary.csv').set_index(['scope', 'algorithm'])
    directory = ROOT / 'models/research_scope_candidates'
    paths = {name: directory / f'mixed_{name.lower()}_candidate_v2.joblib' for name in ['HGB', 'LightGBM']}
    if any(p.exists() or p.with_suffix('.metadata.json').exists() for p in paths.values()):
        raise FileExistsError('Research candidates already exist; do not overwrite them')
    rows, y, ids = load_training_rows(ROOT, 'mixed')
    X = location_frame(rows)
    baseline = json.loads((ROOT / 'models/mixed_historical_v1.metadata.json').read_text())
    if hashlib.sha256(ids.to_numpy(dtype='<i8').tobytes()).hexdigest() != baseline['training_ids_sha256']:
        raise ValueError('Candidate training membership changed')
    directory.mkdir(exist_ok=True)
    for algorithm, path in paths.items():
        result = summary.loc[('mixed', algorithm)]
        if not bool(result['eligible_for_review']):
            raise ValueError(f'{algorithm} did not meet the study guardrails')
        pipeline = candidate_pipeline(algorithm).fit(X, y)
        probability = pipeline.predict_proba(X.iloc[:32])
        joblib.dump(pipeline, path)
        np.testing.assert_array_equal(probability, joblib.load(path).predict_proba(X.iloc[:32]))
        metadata = dict(baseline)
        metadata.update(
            model_version=f'mixed_{algorithm.lower()}_candidate_v2', status='research only; not selected for deployment',
            estimator=algorithm, estimator_parameters=pipeline.named_steps['model'].get_params(),
            model_selection='Nominated after reviewing outer development scores; no independent superiority claim',
            validation='Common 5 outer folds; fixed settings and .5 threshold; see separate inner family selector',
            validation_metrics={name: float(result[name]) for name in baseline['validation_metrics']},
            environment=protocol['environment'],
            artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            deployment_requirement='Explicit team approval and loader/version update; LightGBM also needs its pinned runtime dependency',
        )
        path.with_suffix('.metadata.json').write_text(json.dumps(metadata, indent=2, default=str) + '\n')
        print(path.relative_to(ROOT), metadata['artifact_sha256'], flush=True)
    if protected_snapshot(ROOT) != protocol['protected_artifacts']:
        raise RuntimeError('Production source or artifacts changed')


if __name__ == '__main__':
    main()

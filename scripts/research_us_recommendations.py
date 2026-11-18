"""Save one descriptive training-row example; no model training or LLM calls."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.us_modeling import load_us_training_data
from utils.us_prediction import USPredictor
from utils.us_recommendation_research import benchmarks, nearest_campaigns, scenario_comparison


def main():
    X, _, ids = load_us_training_data(ROOT)
    columns = ['goal', 'duration', 'category', 'subcategory', 'location']
    training = X[columns].assign(**{'project id': ids})
    campaign = X.iloc[0][columns].to_dict()
    query_id = int(ids.iloc[0])
    path = ROOT / 'models/us_primary_model_v1.joblib'
    predictor = USPredictor(path)
    summary = dict(schema_version=1, model_id='us_primary_model_v1',
                   artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   training_ids_sha256=hashlib.sha256('\n'.join(ids.astype(str)).encode()).hexdigest(),
                   cohort='26,945 official U.S. listed-state/DC training campaigns',
                   example_query_training_id=query_id, inputs=campaign,
                   prediction=predictor.predict(campaign),
                   benchmarks=benchmarks(training, campaign, exclude_id=query_id),
                   retrieval=nearest_campaigns(training, campaign, exclude_id=query_id),
                   scenario=scenario_comparison(predictor, campaign, {'goal': float(campaign['goal'])*1.2}),
                   scenario_support=dict(goal_within_training_range=bool(training.goal.min() <= campaign['goal']*1.2 <= training.goal.max()),
                                         duration_within_training_range=bool(training.duration.min() <= campaign['duration'] <= training.duration.max())),
                   limitations=['Descriptive training example, not retrieval-quality validation',
                                'Self ID excluded; outcomes never used in distance or benchmarks',
                                'Historical goal units and duration semantics remain assumptions',
                                'Location is a listed proxy; no state/country similarity asserted',
                                'Scenario may extrapolate; no causal or optimization claim',
                                'No LLM invoked; future explanations must match supplied facts'])
    folder = ROOT / 'docs/additional_model_results'
    folder.mkdir(exist_ok=True)
    (folder / 'recommendation_example.txt').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print('Saved descriptive example; benchmark group n=', summary['benchmarks']['n'])


if __name__ == '__main__':
    main()

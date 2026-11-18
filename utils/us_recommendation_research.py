"""Descriptive training-only retrieval and benchmarks, separate from classifiers."""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

from utils.us_prediction import campaign_frame

MIN_GROUP_ROWS = 30


def comparison_rows(training, campaign, exclude_id=None):
    """Choose a supported group and describe any parent-category fallback."""
    row = campaign_frame(campaign).iloc[0]
    rows = training.copy()
    if exclude_id is not None:
        rows = rows[rows['project id'].ne(exclude_id)]
    category = rows[rows.category.eq(row.category)]
    if category.empty:
        return category, 'unavailable: unseen category'
    parents = rows.loc[rows.subcategory.eq(row.subcategory), 'category'].unique()
    if len(parents) and row.category not in parents:
        raise ValueError('Subcategory does not belong to the supplied category')
    subcategory = category[category.subcategory.eq(row.subcategory)]
    if len(subcategory) >= MIN_GROUP_ROWS:
        return subcategory, 'subcategory; all supported U.S. listed states'
    if len(category) >= MIN_GROUP_ROWS:
        return category, 'parent category fallback; all supported U.S. listed states'
    return category.iloc[:0], 'unavailable: fewer than 30 category rows'


def benchmarks(training, campaign, exclude_id=None):
    rows, scope = comparison_rows(training, campaign, exclude_id)
    result = dict(scope=scope, n=len(rows), available=bool(len(rows)),
                  units='Historical goal units assumed comparable within U.S.; duration in fractional days')
    if len(rows):
        for name in ['goal', 'duration']:
            q = rows[name].quantile([.25, .5, .75])
            result[name] = dict(q25=float(q.loc[.25]), median=float(q.loc[.5]), q75=float(q.loc[.75]))
    return result


def nearest_campaigns(training, campaign, k=5, exclude_id=None):
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError('k must be a positive integer')
    rows, scope = comparison_rows(training, campaign, exclude_id)
    result = dict(scope=scope, n_eligible=len(rows), neighbors=[],
                  distance_features=['log1p(goal)', 'duration'], scaling='training-group median/IQR')
    if rows.empty:
        return result
    query = campaign_frame(campaign)
    values = np.column_stack([np.log1p(rows.goal), rows.duration])
    scaler = RobustScaler().fit(values)
    index = NearestNeighbors(n_neighbors=min(k, len(rows))).fit(scaler.transform(values))
    query_values = np.column_stack([np.log1p(query.goal), query.duration])
    distances, positions = index.kneighbors(scaler.transform(query_values))
    for distance, position in zip(distances[0], positions[0]):
        record = rows.iloc[position]
        result['neighbors'].append(dict(project_id=int(record['project id']), distance=float(distance),
                                        goal=float(record.goal), duration=float(record.duration)))
    return result


def scenario_comparison(predictor, campaign, changes):
    if set(changes) - {'goal', 'duration'}:
        raise ValueError('Research scenarios may change only goal and duration')
    changed = dict(campaign, **changes)
    original = predictor.predict(campaign)
    scenario = predictor.predict(changed)
    return dict(original_inputs=campaign_frame(campaign).iloc[0].to_dict(),
                scenario_inputs=campaign_frame(changed).iloc[0].to_dict(),
                original_prediction=original, scenario_prediction=scenario,
                probability_difference=scenario['success_probability']-original['success_probability'],
                interpretation='Model response to substituted inputs, not a causal effect')

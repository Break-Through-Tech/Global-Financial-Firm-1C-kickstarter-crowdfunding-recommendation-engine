"""Focused HGB accuracy experiment with inner-only threshold selection."""
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd

from utils.us_validation import candidates, metrics

# These tolerances are fixed before running the focused experiment.
MIN_ACCURACY_GAIN = 0.003
MAX_AUC_LOSS = 0.002
MAX_FAILURE_RECALL_LOSS = 0.01
THRESHOLDS = [0.45, 0.5, 0.55]


def focused_candidates():
    original = candidates()['HGB tuned']
    return {
        'original': original,
        'regularized': clone(original).set_params(
            model__l2_regularization=1.0, model__min_samples_leaf=60),
        'slower': clone(original).set_params(
            model__learning_rate=0.03, model__max_iter=300),
    }


def choose_inner(X, y, seed):
    estimators = focused_candidates()
    splitter = StratifiedKFold(2, shuffle=True, random_state=seed)
    records = []
    for name, estimator in estimators.items():
        probability = np.empty(len(y))
        for fit, valid in splitter.split(X, y):
            model = clone(estimator).fit(X.iloc[fit], y.iloc[fit])
            probability[valid] = model.predict_proba(X.iloc[valid])[:, 1]
        for threshold in THRESHOLDS:
            records.append(dict(candidate=name, threshold=threshold, **metrics(y, probability, threshold)))
    table = pd.DataFrame(records)
    baseline = table[table.candidate.eq('original') & table.threshold.eq(0.5)].iloc[0]
    eligible = table[(table.roc_auc >= baseline.roc_auc - MAX_AUC_LOSS) &
                     (table.failure_recall >= baseline.failure_recall - MAX_FAILURE_RECALL_LOSS)]
    best = eligible.sort_values(['accuracy', 'roc_auc'], ascending=False).iloc[0]
    # Keep the original when inner accuracy differences are too small.
    if best.accuracy < baseline.accuracy + MIN_ACCURACY_GAIN:
        best = baseline
    return str(best.candidate), float(best.threshold), table


def focused_experiment(X, y):
    outer = StratifiedKFold(3, shuffle=True, random_state=20261027)
    original_probability = np.empty(len(y))
    experiment_probability = np.empty(len(y))
    thresholds = np.empty(len(y))
    records, inner_records = [], []
    for fold, (fit, valid) in enumerate(outer.split(X, y), 1):
        name, threshold, inner_table = choose_inner(X.iloc[fit], y.iloc[fit], 100 + fold)
        inner_records.append(inner_table.assign(fold=fold))
        original = clone(focused_candidates()['original']).fit(X.iloc[fit], y.iloc[fit])
        old_p = original.predict_proba(X.iloc[valid])[:, 1]
        if name == 'original':
            new_p = old_p
        else:
            new_model = clone(focused_candidates()[name]).fit(X.iloc[fit], y.iloc[fit])
            new_p = new_model.predict_proba(X.iloc[valid])[:, 1]
        original_probability[valid] = old_p
        experiment_probability[valid] = new_p
        thresholds[valid] = threshold
        for label, p, t in [('original', old_p, 0.5), ('focused', new_p, threshold)]:
            records.append(dict(fold=fold, procedure=label, candidate=name if label == 'focused' else 'original',
                                threshold=t, **metrics(y.iloc[valid], p, t)))
        print(f'Focused experiment: outer fold {fold}/3 complete', flush=True)
    # metrics accepts a vector of thresholds for pooled nested classifications.
    old = metrics(y, original_probability)
    new = metrics(y, experiment_probability, thresholds)
    folds = pd.DataFrame(records)
    paired = folds.pivot(index='fold', columns='procedure', values='accuracy')
    accept = bool(new['accuracy'] >= old['accuracy'] + MIN_ACCURACY_GAIN and
                  new['roc_auc'] >= old['roc_auc'] - MAX_AUC_LOSS and
                  new['failure_recall'] >= old['failure_recall'] - MAX_FAILURE_RECALL_LOSS and
                  ((paired.focused - paired.original) > 0).sum() >= 2)
    final_name, final_threshold, final_inner = choose_inner(X, y, 100)
    if not accept:
        final_name, final_threshold = 'original', 0.5
    decision = dict(accept_improvement=accept, final_candidate=final_name,
                    final_threshold=final_threshold, original_metrics=old, focused_metrics=new,
                    minimum_accuracy_gain=MIN_ACCURACY_GAIN, maximum_auc_loss=MAX_AUC_LOSS,
                    maximum_failure_recall_loss=MAX_FAILURE_RECALL_LOSS,
                    limitation='Restricted nested development comparison after prior tuning; promotion uses outer results and adds selection optimism. No significance claim from three overlapping folds.')
    return decision, folds, pd.concat(inner_records), final_inner, original_probability, experiment_probability, thresholds

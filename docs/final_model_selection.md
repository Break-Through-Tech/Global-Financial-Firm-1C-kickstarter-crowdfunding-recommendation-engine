# Final U.S. model selection

Selected: **the original tuned HistGradientBoosting pipeline, threshold 0.5**. The focused accuracy experiments did not establish a meaningful improvement. The complete pipeline is fitted on all **26,945 official U.S. training campaigns**. No reserved labels or reserved predictions were used.

## Selection and validation

We retained the conservative five-input policy: goal, planned duration, category, subcategory and listed state/DC location. Timing, flags, rewards and renamed/title features stay excluded. Goal references, numeric imputation/scaling and cross-fitted target encoding are learned inside each training fold. We added no uncertain feature family.

The existing ordinary five-fold tuning results remain historical development evidence: HGB AUC .7255, Logistic .7201, RF .7209 and Extra Trees .7205. HGB tuning accuracy/F1 were .6694/.7166. Those scores selected settings and are optimistic selection estimates. Failure recall was not recorded in that summary. Do not treat a change from those scores to the new protocol as a measured performance loss.

The RF notebook contains a later 15-trial winner at .7216, with 193 trees, depth 10, minimum leaf 19, minimum split 11, balanced weights and square-root feature sampling. The comparison document records an earlier five-trial result with 257 trees, minimum leaf 3, minimum split 9 and no class weighting. Both records are preserved; both settings enter the restricted comparison. Zasha's combined notebook and its historical results are unchanged and are not U.S. comparisons.

We ran three stratified outer folds (shuffle, seed 20261027), with two stratified inner folds (seed 100 + outer-fold number). Each family compared only existing settings: baseline/tuned HGB, baseline/tuned Logistic, the two RF records, and reference/tuned Extra Trees. Inner AUC selected settings and the overall family. The table reports pooled outer-fold metrics at .5. Timing is mean outer-fit time and prediction time for about 8,982 rows, in seconds on this Mac with one numerical thread.

| Procedure | Accuracy | ROC-AUC | Success F1 | Failure recall | Fit sec | Predict sec |
|---|---:|---:|---:|---:|---:|---:|
| HGB | 0.6693 | 0.7239 | 0.7158 | 0.5572 | 0.347 | 0.060 |
| Logistic | 0.6664 | 0.7196 | 0.7095 | 0.5704 | 0.155 | 0.019 |
| RF | 0.6646 | 0.7197 | 0.6937 | 0.6274 | 2.888 | 0.152 |
| ExtraTrees | 0.6676 | 0.7186 | 0.7142 | 0.5561 | 1.483 | 0.158 |
| Selection | 0.6675 | 0.7219 | 0.7117 | 0.5667 | varies | varies |

The HGB row used the tuned candidate in all three folds. The Selection row evaluates selection across families inside the inner loop; it is a different procedure from the frozen HGB. HGB has the strongest observed AUC and accuracy in this restricted comparison. RF identifies more failures but has lower success F1 and accuracy, and costs more to fit. Logistic is simpler and faster, with slightly higher failure recall than HGB, and remains a reasonable alternative. Extra Trees offers no observed advantage here. HGB uses at most 219 boosting iterations with 14-leaf trees, versus 174 Extra Trees or 193 RF trees. The linear model is easiest to describe; HGB's modest ranking advantage and practical fit cost support retaining the leading candidate. These small score gaps do not prove superiority.

## Focused accuracy experiment

`06_us_accuracy_experiments.ipynb` was copied from HGB tuning, then edited separately. It compares the exact tuned HGB with (1) L2=1 and minimum leaf=60, and (2) learning rate=.03 with 300 iterations. It examines thresholds .45/.50/.55 using only inner out-of-fold predictions.

Rules fixed before execution require at least .003 accuracy gain, at most .002 AUC loss and at most .01 failure-recall loss. Promotion also requires positive accuracy changes in at least two outer folds. These are practical tolerances, not significance criteria. The full-training inner comparison is used for final settings, never the reserved partition.

The best regularized candidate gained only about .0011 accuracy in the final inner comparison, below the .003 guardrail. Raising the original HGB threshold from .50 to .55 raised inner failure recall from .5639 to .6418 but reduced accuracy from .6681 to .6642 and success recall from .7547 to .6827. Lowering it to .45 reduced failure recall to .4808. These are tuning diagnostics, not independent evaluation estimates.

Every outer fold retained the original tuned HGB at .5. Outer accuracy was .6697, .6688 and .6695. Old and focused-procedure metrics therefore match exactly:

| Training-only outer validation | Accuracy | ROC-AUC | Success F1 | Failure recall |
|---|---:|---:|---:|---:|
| Original tuned HGB | .6693 | .7239 | .7158 | .5572 |
| Focused procedure | .6693 | .7239 | .7158 | .5572 |

The focused procedure is nested for settings/threshold choice. The original row is fixed-configuration outer CV, but its configuration came from prior full-training tuning. Neither is a pristine independent estimate. Restricted candidate design used earlier tuning, and reviewing outer scores for final promotion adds selection optimism. This is not a nested replay of the full original searches. Three overlapping folds do not justify a significance claim, and fold SD is not a confidence interval.

## Class errors and calibration

There are 12,229 failed and 14,716 successful training campaigns (54.615% success). Accuracy for an all-success rule would be .5461, with zero failure recall. HGB's held-out-fold results are:

| Actual class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Failed | .6610 | .5572 | .6047 | 12,229 |
| Successful | .6745 | .7625 | .7158 | 14,716 |

| Actual / predicted | Failed | Successful |
|---|---:|---:|
| Failed | 6,814 | 5,415 |
| Successful | 3,495 | 11,221 |

False positives are the **5,415 failed campaigns predicted successful**. False negatives are the **3,495 successful campaigns flagged failed**. Failure recall .5572 means many failed campaigns are still missed. This is a useful development model, not a reliable guarantee of campaign success.

Brier score is .2096 versus .2479 for the outer-training-prior baseline. Ten quantile-bin calibration points are close to the diagonal: the lowest bin predicts .1788 with observed .1722, and the highest predicts .8182 with observed .8186. This is a descriptive training-fold assessment, not proof of calibration in future campaigns. No post-hoc calibrator was fitted.

The validation notebook contains both selection-procedure and primary-HGB confusion matrices, calibration tables/plot, and goal/duration/category error breakdowns. Groups require at least 100 rows and 20 of each class. Failure recall varies substantially: .1157 for goals up to 1,000, .9043 for goals above 10,000 through 50,000, .2243 for durations up to 15 days and .7290 for durations above 60 through 90 days. Category results include Music .2260 and Publishing .8141 failure recall. Case mix and base rates differ; these are descriptive errors, not causal effects or recommendations to alter a campaign.

## Frozen model

Exact estimator parameters, including defaults under the recorded environment:

```json
{
  "categorical_features": "from_dtype",
  "class_weight": null,
  "early_stopping": "auto",
  "interaction_cst": null,
  "l2_regularization": 0.0026915045092098205,
  "learning_rate": 0.04318571864689941,
  "loss": "log_loss",
  "max_bins": 255,
  "max_depth": null,
  "max_features": 1.0,
  "max_iter": 219,
  "max_leaf_nodes": 14,
  "min_samples_leaf": 43,
  "monotonic_cst": null,
  "n_iter_no_change": 10,
  "random_state": 42,
  "scoring": "loss",
  "tol": 1e-07,
  "validation_fraction": 0.1,
  "verbose": 0,
  "warm_start": false
}
```

Preprocessing: `FeatureBuilder(include_timing=False, include_flags=False, include_reward=False, exclude_renamed=True)`; training-fold goal references; numeric median imputation and StandardScaler; target encoding for category/subcategory/state with smoothing 50 and five-fold cross-fitting at seed 42. Class 1 means successful. The threshold is frozen at .5. Full input/derived-feature schema, environment, cohort-ID digest, timings, nested metrics, calibration bins, error groups and limitations are saved in the metadata.

Local files, ignored by Git pending an artifact-sharing decision:

- `models/us_primary_model_v1.joblib`
- `models/us_primary_model_v1.metadata.json`
- `models/us_accuracy_decision.json`
- `models/us_accuracy_outer_folds.csv`

The final fit took 0.554 seconds. All 26,945 rows produced exactly identical probabilities and predictions before and after joblib serialization. SHA-256 of the frozen joblib:

`243bb0d53ac7b456e6b636c424bc8d0d74dbd67f014d5f8f175b7d4226bd94b6`

Runtime: Python 3.12.3, scikit-learn 1.9.1, NumPy 2.5.3, pandas 3.0.5, joblib 1.6.0. Recreate the reviewed environment before sharing a serialized model; joblib portability across library versions is not established.

## Prediction and reproduction

From the repository root:

```python
from utils.us_prediction import USPredictor, predict_campaign

predictor = USPredictor("models/us_primary_model_v1.joblib")
result = predictor.predict({
    "goal": 5000, "duration": 30, "category": "Art",
    "subcategory": "Painting", "location": "CA",
})
print(result)
```

`predict_campaign(campaign)` is the one-call equivalent. Reuse `USPredictor` in Streamlit to avoid loading on every prediction. Goal and duration must be positive finite numbers, including fractional days. Category and subcategory must be nonempty strings; unseen values use training-fitted encoding fallbacks. Location accepts a state/DC code or `City, ST`, with the suffix normalized. Missing fields, unsupported states and invalid numeric values raise ValueError. Input dictionary order is irrelevant. Excluded fields are ignored.

Both training notebooks executed successfully from fresh kernels and saved outputs only after complete execution. No training notebook remains to run manually. To reproduce, run these in order:

```bash
.venv/bin/python scripts_execute_us_validation.py 06_us_accuracy_experiments.ipynb
.venv/bin/python scripts_execute_us_validation.py 06_us_validation.ipynb
```

This refits and replaces local v1 artifacts; do not rerun after the reserved-evaluation freeze. The existing tuning notebooks and interpretability notebook did not need rerunning. The historical combined notebook must not be rerun for this milestone. No Drive authentication or uploads occurred.

## Tests and remaining evaluation

Synthetic unit tests cover training-loader ordering, missing artifacts, fractional labels and status disagreement; conservative fitting; inference validation, unseen categories/states, feature order, excluded fields, model loading, saved threshold and probability consistency. The real-data integration test is separate and skips only when either required processed artifact is missing. Present but invalid data raises an error.

```bash
.venv/bin/python -m unittest discover -s tests -v
US_MODELING_INTEGRATION_ROOT=/tmp/team1c-no-processed-data .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m flake8 utils tests --extend-exclude=.venv --count --select=E9,F63,F7,F82 --show-source --statistics
```

With local data: **57 passed, 0 failed, 0 errors, 0 skipped**. With an explicitly absent integration-data root: **56 passed, 0 failed, 0 errors, 1 skipped**. This leaves local datasets untouched while exercising the data-absent branch. Critical lint passes with zero findings; local virtual environments are excluded from lint. CI uses the same test/lint commands and needs no dataset. A new remote Actions run has not been triggered.

The October 15 audit identified 1,486 reserved U.S. IDs exposed during historical development. We did not open `y_test.csv`, read reserved ID files, or evaluate the reserved partition. The unchanged official split has 11,475 reserved campaigns; it is historically exposed. Listed location is a state/DC proxy, not registration country. Source snapshots, launch availability, goal currency and duration meaning remain unresolved. Random training folds do not demonstrate prospective performance.

`08_us_reserved_evaluation.ipynb` is prepared, unexecuted, and blocked by empty approval fields. Review [the proposed protocol](us_reserved_test_protocol.md), then obtain Anna and Zasha's explicit approval of the protocol and frozen model hash before opening labels. The one approved evaluation will report all 11,475 campaigns plus the 1,486 historically exposed and 9,989 not-known-exposed groups. Do not call the latter independent or tune from any reserved score. External or prospective evaluation remains preferable.

## Changed files in this milestone

- `.github/workflows/python-app.yml`
- `.gitignore`
- `requirements.txt`
- `docs/final_model_selection.md`
- `docs/us_reserved_test_protocol.md`
- `docs/06_handoff.md`
- `notebooks/06_us_accuracy_experiments.ipynb`
- `notebooks/06_us_validation.ipynb`
- `notebooks/08_us_reserved_evaluation.ipynb`
- `scripts_execute_us_validation.py`
- `utils/us_modeling.py`
- `utils/us_validation.py`
- `utils/us_accuracy.py`
- `utils/us_prediction.py`
- `tests/test_us_modeling.py`
- `tests/test_us_modeling_integration.py`
- `tests/test_us_prediction.py`
- `tests/test_reserved_protocol.py`

The four ignored local artifacts are listed above. No files were committed or pushed.

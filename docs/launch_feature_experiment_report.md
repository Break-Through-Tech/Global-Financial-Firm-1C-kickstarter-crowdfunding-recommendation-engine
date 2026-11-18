# Launch feature experiment report

November 10, 2026. Branch `feat/launch-feature-experiments`, based on latest verified remote main `fd7ca97`. No commit, push, merge, frozen dataset change or Streamlit change. HGB v1 and its metadata remain byte-identical. Both downloaded reference ZIPs and extracted repositories are ignored.

## Decision

Keep HGB v1 as primary. The listed city/state frequency candidate passes the development guardrails and is worth a separate team review: accuracy gains of .00646, .01158 and .00512 in the three outer folds, pooled gain .00772, AUC gain .00839 and failure-recall gain .01439. The inner selector chose city frequency in every fold. This is consistent development evidence, not independent proof of superiority. A separate local `models/us_candidate_v2.joblib` and metadata were saved and checked for exact round-trip predictions/probabilities on all 26,945 training rows. Do not activate it in the current dashboard: it needs full listed city/state, whereas state-only input is supported by v1.

Reward groups produce larger measured gains, but historical launch reward configuration cannot be verified. Titles may have later edits and encoding corruption. Neither family is approved for production. No reward/title model was serialized. No global model was trained because currency/snapshot/coverage semantics remain unresolved.

## Training-only comparison

All rows below use the exact frozen HGB configuration, threshold .5, and the existing three outer partitions. Precision, recall and first F1 refer to successful campaigns. Fit/predict seconds are means per outer fit/validation partition; they exclude inner selection overhead. Full metrics, both class precisions/recalls, confusion counts and fold-level results are saved in the CSVs.

| Configuration | Accuracy | AUC | Precision | Recall | Success F1 | Failure F1 | Failure recall | Brier | Fit sec | Predict sec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.6693 | 0.7239 | 0.6745 | 0.7625 | 0.7158 | 0.6047 | 0.5572 | 0.2096 | 0.3467 | 0.0748 |
| reward_count | 0.7015 | 0.7597 | 0.6999 | 0.7939 | 0.7440 | 0.6423 | 0.5904 | 0.1959 | 0.6793 | 0.2192 |
| reward_amounts | 0.6948 | 0.7528 | 0.6954 | 0.7851 | 0.7375 | 0.6354 | 0.5861 | 0.1989 | 0.7210 | 0.2230 |
| reward_relative | 0.6922 | 0.7509 | 0.6926 | 0.7849 | 0.7358 | 0.6314 | 0.5808 | 0.1996 | 0.6882 | 0.1997 |
| title_length | 0.6723 | 0.7247 | 0.6769 | 0.7650 | 0.7183 | 0.6083 | 0.5606 | 0.2094 | 0.4056 | 0.0844 |
| title_style | 0.6726 | 0.7265 | 0.6777 | 0.7635 | 0.7181 | 0.6095 | 0.5632 | 0.2089 | 0.4460 | 0.1054 |
| city_frequency | 0.6770 | 0.7323 | 0.6823 | 0.7647 | 0.7212 | 0.6163 | 0.5716 | 0.2068 | 0.3723 | 0.0643 |
| exploratory_selection | 0.7089 | 0.7697 | 0.7061 | 0.8000 | 0.7501 | 0.6514 | 0.5993 | 0.1920 | 0.7920 | 0.2329 |
| approved_selection | 0.6770 | 0.7323 | 0.6823 | 0.7647 | 0.7212 | 0.6163 | 0.5716 | 0.2068 | 0.3723 | 0.0643 |

Confusion counts, class order failed/successful:

| Configuration | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| baseline | 6814 | 5415 | 3495 | 11221 |
| reward_count | 7220 | 5009 | 3033 | 11683 |
| reward_amounts | 7167 | 5062 | 3162 | 11554 |
| reward_relative | 7102 | 5127 | 3166 | 11550 |
| title_length | 6856 | 5373 | 3458 | 11258 |
| title_style | 6887 | 5342 | 3481 | 11235 |
| city_frequency | 6990 | 5239 | 3463 | 11253 |
| exploratory_selection | 7329 | 4900 | 2943 | 11773 |
| approved_selection | 6990 | 5239 | 3463 | 11253 |

Baseline matches the existing .6693 / .7239 / .7158 / .5572 result exactly at reported precision. Reward count improves accuracy in all three folds; amount and relative groups also improve in all three. Title length improves in all folds but pooled gain .00293 misses the .003 minimum. Title style gains .00323 pooled, with much smaller improvement in fold 3; provenance still excludes promotion. City frequency improves accuracy, AUC and failure recall in each fold. See `paired_accuracy.csv` and `outer_folds.csv`.

The exploratory selector chooses reward_count + reward_amounts in folds 1 and 2 and reward_count + reward_relative in fold 3. The .7089 pooled row evaluates that selection procedure, not one frozen combined model. The pair was formed from the best eligible inner singleton results within each outer training partition. No combination was chosen using outer validation. No production-eligible pair beyond city alone was justified.

“Approved selection” means only inputs allowed by the existing planning-input contract, here baseline or location-derived city frequency. It does not establish verified launch snapshots or team approval to deploy. Inner guardrails require +.003 accuracy, at most .002 AUC loss and at most .01 failure-recall loss. The outer guardrail also requires gains in at least two folds. Restricted nesting accounts for these new inner choices, but prior tuning and repeated use of outer data remain selection bias. No significance claim or confidence interval is made. XGBoost/LightGBM are absent, so the optional comparison was skipped.

## Pipeline and references

See [pipeline review](launch_pipeline_review.md) for the complete audit and reference-feature mapping. Current membership checks confirm global 29,288/12,553, U.S. 26,945/11,475 and named international 1,427/637, with ordered derivative intersections and no train/reserved overlap. Only reserved IDs were opened, never reserved target columns or y_test. The U.S. loader verifies training labels and 38,420 settled rows. Prior October 15 evidence establishes global cleaning replay and historical holdout exposure. No new defect warranted changing established results.

Conceptual additions come from 1A reward/city work and 1B reward ladder/title/listed-city-frequency work. Goal/subcategory references, state, duration and historical reward/title utilities already existed in 1C. Rejected inputs: actual updates, pledged/funded percentages, backers/comments, reward activity/quantities, unverified reconstructed timing and unavailable timed creator history. Planned updates are not a valid substitute for final update counts in training. Rounding/duration/state-goal experiments were not repeated.

## Application and workflow status

U.S. v1 remains available and primary. City v2 is a separate experimental candidate with the same official training population. Global remains unavailable; no global notebook or deployment was attempted without defensible common monetary units. Global includes U.S.; international-only is a separate selected-location exploratory workflow, and unknown locations are not verified international campaigns.

See [dashboard proposal](dashboard_model_interface.md) for required fields, outputs, locations/categories, artifact paths, loading/validation requirements and warnings. Coordination is through documentation only. Existing `USPredictor` continues to serve v1. It must not present global predictions.

See [reproducibility guide](launch_experiments_reproducibility.md) for environment setup, required files, exact commands, active versus archival notebooks and construction order. The only new notebook to run is `06_us_extended_feature_experiments.ipynb`, after tests and source review. Run `scripts/execute_extended_features.py` for a clean kernel, followed optionally by `scripts/prepare_us_city_candidate.py`. Do not rerun the historical validation notebook because it overwrites v1.

## Verification and changed files

Automated suite: 62 tests passed, zero failures/errors/skips. Critical flake8 checks pass. New notebook executed successfully from a clean kernel with computed tables and embedded/exported figure. Exact saved-v1/refitted-baseline probabilities agree on all 26,945 rows; representative inference calls match. New candidate exact serialization agreement also covers all rows. SHA-256 checks confirm all 41 inventoried pre-existing non-reserved-label data/model files unchanged, including v1 metadata. Reserved labels were not opened or hashed. Existing notebooks and utility modules are unchanged.

| Changed/new file | Purpose |
|---|---|
| `.gitignore` | Ignore all local reference downloads/extractions |
| `utils/us_extended_features.py` | Small cloneable feature builder, fixed HGB comparison and inner selection |
| `tests/test_us_extended_features.py` | Five tests for definitions, references, outcome exclusion, baseline and serialization |
| `notebooks/06_us_extended_feature_experiments.ipynb` | Executed training-only experiments and saved outputs |
| `scripts/review_launch_inputs.py` | Read-only IDs and training-source audit |
| `scripts/execute_extended_features.py` | Fresh kernel execution, atomic notebook save |
| `scripts/prepare_us_city_candidate.py` | Guarded separate candidate fitting and serialization verification |
| `docs/launch_pipeline_review.md` | Pipeline audit, reference mapping and global feasibility decision |
| `docs/dashboard_model_interface.md` | Coordination proposal without changing the app |
| `docs/launch_experiments_reproducibility.md` | Exact setup, run order and preservation rules |
| `docs/launch_feature_experiment_report.md` | This report and full comparison |
| `docs/launch_feature_results/*` | Input/environment audit, metrics, folds, pairs, figure, tests and serialization evidence |

Local ignored additions: reference ZIP extractions; `models/us_candidate_v2.joblib` and sibling metadata. No large reference implementation was copied or imported.

## Remaining team decisions

Confirm reward/title snapshots and duration/goal currency semantics; resolve global currency and location policy before training/deployment; review whether city/state inputs and frequency fallback are suitable for the app with Zasha; agree candidate artifact distribution and activation. Current inference does not enforce taxonomy pairing or artifact hashes automatically. Preserve explicit held-out protocol approval and disclose historical exposure. Independent or prospective evaluation remains needed for stronger generalization claims. No causal recommendation follows from these associations.

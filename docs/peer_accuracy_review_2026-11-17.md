# Model accuracy and peer review, November 17, 2026
The headline gap is largely a difference in prediction-time information: the peer models near 81% include actual campaign updates. Without updates their reported scores are about 70–71%, still on different cohorts and with additional snapshot assumptions. Team 1C already has a modest city-frequency candidate and larger but provenance-restricted reward experiments. Review those existing results before repeating model searches. The new international baseline needs particular caution because its accuracy is below its majority baseline.

Fresh public source snapshots were reviewed, not just their READMEs. No external notebook or training code was executed, no peer model was loaded, and no Team 1C model/data/split was changed by this review. Metrics below are **reported peer results**, checked against code and saved outputs, not independently reproduced results.

| Repository | Verified current main commit | Review source |
|---|---|---|
| Team 1A | `1b41a8444848966cd5bebd55d629a45411f0fcd7` | Exact commit archive, inspected as text |
| Team 1B | `0a146abdcdb3cb54d14369883149f081ab72fab2` | Exact commit archive, inspected as text |

Revisions were verified with `git ls-remote ... refs/heads/main`. Exact commit archives were downloaded from GitHub codeload and source/docs/reports extracted under `/private/tmp`; dataset files were not extracted. Existing November 10 reference snapshots were not assumed current. Immutable source links below point to these verified commits.

## What their headline accuracy measures

| Team and reported procedure | Accuracy | ROC-AUC | Failure recall | Prediction-point caveat |
|---|---:|---:|---:|---|
| 1A tuned XGBoost, with updates | 81.39% | .8981 | 78.06% | Includes actual campaign updates, unavailable at pre-launch |
| 1A same selected XGBoost settings, without updates | 70.33% | .7670 | 59.29% | Still includes scraped reward count (`levels`), whose initial snapshot needs verification |
| 1B tuned/isotonic-calibrated Random Forest, with updates, random split | 81.55% | .8976 | 77.73% | Trained/evaluated using final update counts; prototype substitutes planned counts |
| 1B same model procedure, launch_only, random split | 71.35% | .7753 | Not explicitly reported for this exact ablation | Includes rewards, title and inferred launch-calendar features |
| 1B with updates, chronological split | 80.48% | .8874 | Not extracted for this exact row | Includes final updates and uses inferred launch dates |
| 1B launch_only, chronological split | 71.83% | .7884 | Not explicitly reported for this exact ablation | Same input-availability and inferred-date caveats |

Sources: [1A final-selection notebook][A-selection], [1A without-updates fitting notebook][A-xgb], [1B final model card][B-card], [1B finalist CSV][B-final], and [1B modeling notebook][B-notebook]. 1A failure recalls are calculated from its displayed confusion counts: 2067/(2067+1419) without updates and 2721/(2721+765) with updates. 1B final failure recall is 2701/(2701+774), from the displayed final-model confusion matrix in notebook cell 108. Do not confuse the repositories' ordinary `recall` column, which is success recall, with failure recall.

Adding updates accounts for **11.06 percentage points** in 1A's selected XGBoost comparison and **10.20 points** in 1B's final-model comparison. Therefore their roughly 81% headlines do not establish an 81% pre-launch target that Team 1C can obtain by changing algorithms. Removing updates also does not make all remaining inputs verified launch snapshots.

## Team 1A: actual implementation

The reviewed preparation/export workflow uses 38,504 successful/failed U.S.-listed campaigns, with a stratified 80/20 seed-42 split: 30,803 training and 7,701 test rows. It extracts the terminal state/DC token and applies a case-sensitive whitelist, which differs from Team 1C's cohort normalization. Missing locations/reward strings are filled with fixed strings before filtering. No project-ID deduplication was found in the reviewed filter, missing-value notebook, U.S.-filter notebook or export path. This is a duplicate-overlap risk to audit, not proof that a particular number of duplicates crosses its split. Sources: [filtering script][A-filter], [U.S. cohort notebook][A-us], [missing-value notebook][A-missing], [model-preparation notebook][A-prep].

The actual no-updates XGBoost inputs are `goal`, `duration`, `levels`, `category`, `subcategory`, `state`; the with-updates version adds `updates`. The scaler and one-hot encoder are inside the estimator pipeline passed to five-fold GridSearchCV, so they refit within CV. The first grid searches estimators 200/400, learning rates .05/.1 and depths 3/4/6; selected settings are 400, .1, depth 3. A second, wider grid is also evaluated on the same X_test. The final comparison selects among several model families/features using reported test results. The README's “test used once” phrasing should therefore not be treated as a single untouched final evaluation of the whole selection process. Small differences do not prove ties or superiority. Sources: [XGBoost notebook][A-xgb], [with-updates notebook][A-xgb-updates], [final selection][A-selection].

Its feature experiment uses a shared random holdout to compare baseline, city, updates, and city+updates. City-only changes Logistic Regression accuracy .6860 to .6907 and AUC .7463 to .7534 while increasing encoded features from 119 to 2,936; updates-only accuracy is .8025. This is useful feature evidence on that split, not a guaranteed transferable gain. [Feature experiment][A-features].

The new app exports separate Plan A/Plan B XGBoost pipelines offline and loads them from disk, a useful separation of training from serving. However both plans require rewards; Plan B additionally requires updates. The app contract cannot be substituted for Team 1C's frozen five-input model. The export script also retains raw category spellings and maps display labels back, whereas 1C already normalizes HTML categories. Sources: [export script lines 37–50 and 111–169][A-export], [artifact loader lines 39–51 and input contract][A-interface]. Scenario lifts remain model responses, not causal effects.

## Team 1B: actual implementation

Its auditable cleaning log reports 45,957 raw rows, 41,965 settled, 38,533 after its U.S. filter, 41 missing reward rows removed, 113 duplicate project IDs removed and one goal sentinel removed, leaving **38,378** rows. The state whitelist includes five U.S. territories as well as the 50 states/DC. That is a different population from both Team 1C and Team 1A. Random splitting is stratified 80/20, seed 42, 30,702/7,676; chronological splitting sorts inferred launch_date and uses the last 20%. Sources: [cleaning log][B-clean-log], [geography/cleaning code][B-data-clean], [split code lines 451–483][B-split], [split summary][B-splits].

It constructs title length/style, reward-tier counts/amounts/ratios, goal transforms, duration flags, category/subcategory/state, city frequency, subcategory goal references, and calendar features. `launch_date = funded_date - duration` is an assumption in its code, not independent source verification of the CSV field's semantics. Reward/title helper docstrings label features LAUNCH, but code comments cannot establish that a scraped value was captured before launch. Sources: [feature definitions][B-features], [calendar reconstruction lines 266–287][B-date].

`FINAL_FEATURE_SET='with_updates'`. The documentation explicitly states that updates are counted at campaign end and that the prototype substitutes the creator's planned count; [features.py lines 17–19 and 314–324][B-updates]. A planned number is not the measured historical final count. Their warnings acknowledge the limitation; the 81.55% evaluation does not validate that substitution. [Model card][B-card].

The final classifier is RandomForestClassifier(400 trees, min leaf 3, max_features .3, max_depth 12), wrapped in isotonic CalibratedClassifierCV(cv=3). Searches optimize training-CV AUC. This is real tuned/calibrated work, but its surrounding preprocessing is not wholly fold-contained:

- City counts and subcategory goal references are learned from the outer training split before inner tuning/calibration folds are constructed. They exclude the outer test set, but are not refitted inside each inner fold. [data.py lines 509–525 and 545–564][B-stateful].
- `build_model_matrix` one-hot encodes and fills missing values using the supplied table's own medians; `model_matrices` calls it separately on train and test, then aligns columns. Test-side medians are therefore used when relevant missing values exist. Inner CV also receives an already prepared full-training matrix. This is a transductive/statistical-preprocessing concern, not evidence of a target-encoding label leak. [features.py lines 351–442][B-matrices].
- Notebook cells 71, 80, 83/85 and 100 inspect the same holdout for threshold curves (a best-F1 threshold near .46), update transformations, permutation-based feature pruning, and sigmoid versus isotonic calibration. The final classifier still uses its ordinary .5 class decision; do not attribute final accuracy to adopting the displayed .46 threshold. Repeated design inspection of a holdout limits its independence even when parameter searches use training CV. [Modeling notebook][B-notebook], [model construction lines 70–101][B-model].

Its chronological check is a useful direction, but it is still historical, based on inferred dates, and does not establish 2026 performance. The recommendation entrypoint calls load_splits and fits a scorer/advisor on demand in get_recommender; it is not the same as our pre-saved, hash-checked serving contract. Sources: [recommendation construction][B-rec-fit], [entrypoint lines 1033–1065][B-rec-entry]. Its reusable ideas are explicit source/feature timing, fixed train reference statistics, calibration assessment and cohort-aware error reports, with the fold-safety and feature-timing corrections above.

## Implications before any Team 1C experiment

Do not copy either headline score or substitute planned updates for final updates. Align prediction time, cohort membership, allowed inputs, preprocessing placement, validation/selection procedure and threshold before judging whether an algorithm is better. Keep Team 1C's existing split and reserved-evaluation boundary; no peer metric justifies reopening or tuning against the reserved set.

Ideas worth considering on Team 1C's own training folds are compact listed-city frequency, verified initial reward structure, and calibrated probability reporting. City frequency measures dataset representation, not city population or causal advantage. Reward/title/calendar features require source-timing evidence before deployment. Calibration can improve probability reliability without improving classification accuracy. Existing 1C implementations/experiments must be checked before repeating them; the main analysis below should identify what is already done.

[A-selection]: https://github.com/Break-Through-Tech/Global-Financial-Firm-1a-kickstarter-crowdfunding-recommendation-engine/blob/1b41a8444848966cd5bebd55d629a45411f0fcd7/notebooks/milestone3_task8_final_model_selection.ipynb
[A-xgb]: https://github.com/Break-Through-Tech/Global-Financial-Firm-1a-kickstarter-crowdfunding-recommendation-engine/blob/1b41a8444848966cd5bebd55d629a45411f0fcd7/notebooks/milestone3_task5_xgboost.ipynb
[A-xgb-updates]: https://github.com/Break-Through-Tech/Global-Financial-Firm-1a-kickstarter-crowdfunding-recommendation-engine/blob/1b41a8444848966cd5bebd55d629a45411f0fcd7/notebooks/milestone3_task5_xgboost_with_updates.ipynb
[A-filter]: https://github.com/Break-Through-Tech/Global-Financial-Firm-1a-kickstarter-crowdfunding-recommendation-engine/blob/1b41a8444848966cd5bebd55d629a45411f0fcd7/scripts/filter_campaigns.py
[A-us]: https://github.com/Break-Through-Tech/Global-Financial-Firm-1a-kickstarter-crowdfunding-recommendation-engine/blob/1b41a8444848966cd5bebd55d629a45411f0fcd7/notebooks/task_4_us_campaign_records.ipynb
[A-missing]: https://github.com/Break-Through-Tech/Global-Financial-Firm-1a-kickstarter-crowdfunding-recommendation-engine/blob/1b41a8444848966cd5bebd55d629a45411f0fcd7/notebooks/03_handle_missing_values.ipynb
[A-prep]: https://github.com/Break-Through-Tech/Global-Financial-Firm-1a-kickstarter-crowdfunding-recommendation-engine/blob/1b41a8444848966cd5bebd55d629a45411f0fcd7/notebooks/milestone3_task1_model_preparation.ipynb
[A-features]: https://github.com/Break-Through-Tech/Global-Financial-Firm-1a-kickstarter-crowdfunding-recommendation-engine/blob/1b41a8444848966cd5bebd55d629a45411f0fcd7/notebooks/milestone3_feature_selection_experiments.ipynb
[A-export]: https://github.com/Break-Through-Tech/Global-Financial-Firm-1a-kickstarter-crowdfunding-recommendation-engine/blob/1b41a8444848966cd5bebd55d629a45411f0fcd7/scripts/export_artifacts.py#L37-L50
[A-interface]: https://github.com/Break-Through-Tech/Global-Financial-Firm-1a-kickstarter-crowdfunding-recommendation-engine/blob/1b41a8444848966cd5bebd55d629a45411f0fcd7/app/recommender.py#L39-L51
[B-card]: https://github.com/Break-Through-Tech/Global-Financial-Firm-1b-kickstarter-crowdfunding-recommendation-engine/blob/0a146abdcdb3cb54d14369883149f081ab72fab2/reports/m3_model_card.json
[B-final]: https://github.com/Break-Through-Tech/Global-Financial-Firm-1b-kickstarter-crowdfunding-recommendation-engine/blob/0a146abdcdb3cb54d14369883149f081ab72fab2/reports/m3_task10_finalists.csv
[B-notebook]: https://github.com/Break-Through-Tech/Global-Financial-Firm-1b-kickstarter-crowdfunding-recommendation-engine/blob/0a146abdcdb3cb54d14369883149f081ab72fab2/notebooks/02_modeling.ipynb
[B-clean-log]: https://github.com/Break-Through-Tech/Global-Financial-Firm-1b-kickstarter-crowdfunding-recommendation-engine/blob/0a146abdcdb3cb54d14369883149f081ab72fab2/reports/m1_cleaning_step_log.csv
[B-data-clean]: https://github.com/Break-Through-Tech/Global-Financial-Firm-1b-kickstarter-crowdfunding-recommendation-engine/blob/0a146abdcdb3cb54d14369883149f081ab72fab2/src/data.py#L143-L154
[B-split]: https://github.com/Break-Through-Tech/Global-Financial-Firm-1b-kickstarter-crowdfunding-recommendation-engine/blob/0a146abdcdb3cb54d14369883149f081ab72fab2/src/data.py#L451-L483
[B-splits]: https://github.com/Break-Through-Tech/Global-Financial-Firm-1b-kickstarter-crowdfunding-recommendation-engine/blob/0a146abdcdb3cb54d14369883149f081ab72fab2/reports/m3_split_summary.csv
[B-features]: https://github.com/Break-Through-Tech/Global-Financial-Firm-1b-kickstarter-crowdfunding-recommendation-engine/blob/0a146abdcdb3cb54d14369883149f081ab72fab2/src/features.py#L78-L200
[B-date]: https://github.com/Break-Through-Tech/Global-Financial-Firm-1b-kickstarter-crowdfunding-recommendation-engine/blob/0a146abdcdb3cb54d14369883149f081ab72fab2/src/data.py#L266-L287
[B-updates]: https://github.com/Break-Through-Tech/Global-Financial-Firm-1b-kickstarter-crowdfunding-recommendation-engine/blob/0a146abdcdb3cb54d14369883149f081ab72fab2/src/features.py#L17-L19
[B-stateful]: https://github.com/Break-Through-Tech/Global-Financial-Firm-1b-kickstarter-crowdfunding-recommendation-engine/blob/0a146abdcdb3cb54d14369883149f081ab72fab2/src/data.py#L509-L565
[B-matrices]: https://github.com/Break-Through-Tech/Global-Financial-Firm-1b-kickstarter-crowdfunding-recommendation-engine/blob/0a146abdcdb3cb54d14369883149f081ab72fab2/src/features.py#L351-L442
[B-model]: https://github.com/Break-Through-Tech/Global-Financial-Firm-1b-kickstarter-crowdfunding-recommendation-engine/blob/0a146abdcdb3cb54d14369883149f081ab72fab2/src/modeling.py#L70-L101
[B-rec-fit]: https://github.com/Break-Through-Tech/Global-Financial-Firm-1b-kickstarter-crowdfunding-recommendation-engine/blob/0a146abdcdb3cb54d14369883149f081ab72fab2/src/recommend.py#L304-L333
[B-rec-entry]: https://github.com/Break-Through-Tech/Global-Financial-Firm-1b-kickstarter-crowdfunding-recommendation-engine/blob/0a146abdcdb3cb54d14369883149f081ab72fab2/src/recommend.py#L1033-L1065

## Team 1C: current evidence and improvement priorities

The following numbers were checked against the saved metadata and experiment CSVs at the start of this review. They are development-validation results, not new tests on unseen campaigns. The source code, notebooks, trained weights, thresholds and deployment configuration were not changed.

| Current scope/model | Accuracy | Majority baseline | Lift over majority (percentage points) | ROC-AUC | Failure recall |
|---|---:|---:|---:|---:|---:|
| U.S. primary HGB v1 | 66.93% | 54.61% | +12.32 | 0.7239 | 55.72% |
| International historical Logistic | 60.41% | 60.97% | -0.56 | 0.5893 | 30.52% |
| Mixed historical Logistic | 62.23% | 54.74% | +7.49 | 0.6631 | 50.41% |

Sources: [U.S. metadata](../models/us_primary_model_v1.metadata.json), [international metadata](../models/international_historical_v1.metadata.json), [mixed metadata](../models/mixed_historical_v1.metadata.json), [scope-model methodology](historical_scope_models.md). The U.S. evidence comes from a restricted nested comparison informed by earlier tuning. The two new models are single fixed, goal-free baselines evaluated with five-fold stratified training CV. Different cohorts, inputs and procedures prevent treating this table as a controlled comparison of model families.

The international concern is substantive: its overall accuracy is below the always-successful majority rule and failure recall is only 30.52%. AUC above .5 indicates some ranking information, but does not make the current baseline a reliable planning tool. The existing demonstration label and majority warning must remain. Mixed improves on its majority baseline but remains a limited historical model. Their lower scores do not demonstrate that the U.S. pipeline is broken.

Both newer models intentionally omit funding amount because source currency interpretation is unresolved. They are fixed Logistic baselines, not an assessment of the strongest achievable models in those scopes. We cannot attribute the whole score difference from earlier international experiments solely to funding: the feature sets, estimators or summary procedures also differ. The [funding-unit review](funding_currency_review.md) distinguishes historical listed locations from registration markets and explains why arbitrary local-currency input cannot be added merely to increase accuracy.

### Reuse completed experiments before launching new searches

[Existing algorithm results](additional_model_results/summary.csv) and [the city candidate metadata](../models/us_candidate_v2.metadata.json) already cover much of the obvious improvement work:

| U.S. candidate/configuration | Accuracy | ROC-AUC | Failure recall | Interpretation |
|---|---:|---:|---:|---|
| Frozen HGB v1, conservative inputs | 66.93% | 0.7239 | 55.72% | Current primary |
| XGBoost, conservative inputs | 67.11% | 0.7244 | 55.54% | Only +0.18 percentage points; not a demonstrated practical replacement |
| CatBoost, conservative inputs | 67.13% | 0.7240 | 54.23% | Only +0.19 points and lower failure recall; misses existing promotion guardrails |
| HGB with listed-city frequency, v2 | 67.70% | 0.7323 | 57.16% | Most concrete existing candidate for a focused interface/promotion review |
| LightGBM with listed-city frequency | 67.93% | 0.7326 | 57.67% | Best observed city row, only +0.23 points above city HGB; exploratory, no approved saved replacement |
| HGB with reward count | 70.15% | 0.7597 | 59.04% | Historical snapshot experiment; launch-time eligibility unverified |
| HGB with reward ladder | 71.17% | 0.7706 | 60.25% | Same provenance restriction; not deployable launch-time performance |

This is existing research, not new training. Same-input algorithm swaps have much smaller observed gains than adding different information. It would be inaccurate to say that XGBoost/CatBoost/LightGBM have not been tried, or to promise a large gain from trying them again.

City HGB v2 improves development accuracy by **0.77 percentage points**, AUC by .0084 and failure recall by 1.44 points over v1. It uses fold-fitted frequency of the complete listed city/state string, not population, income or creator history. The gain met the earlier practical development guardrails across the recorded folds, but prior tuning and repeated use of outer folds still add selection optimism. This is not independent proof, and no significance claim is supported by three overlapping fold results.

Its interface is also different: it requires a full listed city/state string. The current U.S. form accepts state-only input. A meaningful candidate review must preserve v1 and settle city normalization, unseen-city/typo behavior and the state-only policy. Do not silently send an invented city such as 'State, CA' into v2 and claim that the city feature is available. [Feature definitions](../utils/us_extended_features.py), [prior development report](launch_feature_experiment_report.md).

Reward features show a larger apparent gain, but scraped reward counts and amounts may include tiers added during the campaign. Being useful predictors in retrospective data does not establish their availability at launch. User-entered initial rewards cannot automatically substitute for historical final snapshots. Title styling and reconstructed launch dates have similar provenance restrictions. Planned update counts cannot substitute for actual completed update counts. Preserve these research results without promoting them into the launch-time model.

### Recommended next work

1. **U.S.: review the already-saved city HGB candidate first.** Resolve its full-city input contract and decide whether the modest development gain warrants a separate versioned option. No retraining is needed merely to review that artifact. Do not replace v1 by default. Review city LightGBM as additional recorded evidence, not a reason to select whichever outer-fold row is highest.
2. **International/mixed: run a small controlled comparison only if further experimentation is authorized.** Compare fixed non-linear baselines against the current Logistic models using identical official training memberships, the same four approved non-monetary inputs and the same fitting/validation partitions. Fit preprocessing within folds, retain threshold .5 initially, and report majority baselines, failure recall, AUC and calibration/error measures. Define the candidate list and promotion rules before looking at results. Do not pool these cohorts into one accuracy competition or invent expected gains.
3. **Improve source quality before relaxing feature policy.** Resolve exact monetary units and initial reward/title snapshots. If funding goals become defensible, evaluate their contribution as a separate controlled change; do not change algorithm and input families at the same time and attribute the whole gain to one feature.
4. **Obtain stronger evaluation evidence.** Newer licensed campaigns with known input snapshots and outcomes would support a temporal/external test. Existing reserved data remain historically exposed and approval-gated. Do not open them, retune from them, or make a new random split simply to obtain a better score. See [reserved protocol](us_reserved_test_protocol.md).

Do not optimize the classification threshold on a test set. A threshold can trade false positives for false negatives; it cannot improve ranking quality and may reduce failure recall while raising apparent accuracy. Accuracy is a useful summary only alongside class balance and the cost of falsely telling a failed campaign it will succeed.

Passing a Streamlit example or matching the known outcome of a campaign in our training set verifies application consistency. It does not demonstrate generalization accuracy or prospective performance. The honest near-term objective is a modest, defensible improvement with the same prediction point, rather than matching another team's headline percentage.

No additional fitting, reserved-label evaluation, code copying, model promotion, commit or push was performed for this review. Only this evidence report was created/updated. Existing frozen results and contributor work remain unchanged.

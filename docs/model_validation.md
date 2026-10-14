# Pre-tuning validation and launch-time feature review

**Status:** Computational diagnostics completed (10/10 sensitivity experiments, 3/3 chronological comparisons, and 8/8 preprocessing tests). Source-field verification, code-owner review, and approval of the final feature policy remain open.

**Scope:** Use the official split from `03_preprocess_data.ipynb`. All model selection and diagnostic experiments use the **training partition only**; the official held-out test set is reserved for final evaluation.

## 1. Why this work exists

The historical `05_modeling_v3.ipynb` compared six estimators with five-fold cross-validation. Before tuning, the team needed to verify the correctness of the modeling pipeline, assess the influence of potentially unavailable features, and investigate performance across campaign time.

The previous `TargetEncoder` could return cached training out-of-fold (OOF) values for unrelated new data if the input index matched the fitted training index. The revised implementation uses `fit_transform()` to construct training OOF encodings, and `transform()` to map unseen rows using fitted category statistics. Inner-fold smoothing priors are computed from the corresponding fitting folds.

The revised `utils/preprocessing.py` also accepts `exclude_renamed=True`. This removes a possible look-ahead feature at the model-pipeline boundary **without changing** the shared `utils/features.py` generator or overwriting the historical V3 notebook.

## 2. Intended prediction time and feature policy

The project is currently framed as predicting success at or near the **campaign launch**. A feature is acceptable only if its value would have been observable at that point. These classifications remain provisional pending source documentation and team review.

| Field or feature group | Policy | Rationale / follow-up |
|---|---|---|
| `status`, `target` | Never inputs | Outcome labels. |
| `pledged`, `funded percentage`, `backers`, `updates`, `comments` | Excluded | Post-launch financial or engagement outcomes. |
| `project id`, `url`, `creator`, `slug` | Not direct inputs | Identifiers; derived values require historical-availability review. |
| `goal`, `duration`, `category`, `subcategory`, `location`, `levels` | Provisionally allowed | Plausibly available at launch, but confirm whether archived values reflect later edits. |
| `name`, `reward levels` | Not yet verified as launch snapshots | A subsequent scrape could capture changes after launch. |
| `renamed` | **Exclude from conservative models** | Current title versus URL slug may reveal a later rename. |
| `funded date` | **Source meaning unresolved** | Code treats it as campaign end/deadline date; if outcome-dependent, launch-timing derivations may leak information. |
| `backers_needed` | Historical naming retained | Computed as `log1p(goal) - log1p(median_tier)`; not a literal number of needed backers. |
| `creator_history` | Deferred | Requires strictly time-ordered outcome availability and fold-safe inference. Not in current `make_pipeline()`. |
| Target-encoded categories/states | Allowed with fold discipline | Train OOF values and inference mappings must not use validation/test labels. |

### Meaning of `funded date`

The existing feature code calculates an implied launch timestamp:

```python
launch = pd.to_datetime(funded_date) - pd.to_timedelta(duration, unit="D")
```

This supports a launch-time feature only if `funded date` is a suitable *scheduled end/deadline* value, known under the intended prediction protocol. It is **not** sufficient that the column parses cleanly or produces plausible historical years. Obtain the raw-source definition, not an inference based on the numbers.

### Geographic scope

The challenge brief discusses U.S.-based Kickstarter projects, while the current modeling population is all 41,841 settled campaigns after cleaning. State abbreviations in a location field cannot establish the project's country with certainty. Ask the Challenge Advisor whether the broader sample is acceptable. If U.S.-only is mandatory, establish a reliable country indicator first and regenerate the official split and all downstream results.

## 3. Data-integrity diagnostics

`notebooks/05b_pre_tuning_validation.ipynb` reconstructs only the official training rows from cleaned data and `train_project_ids.csv`. It verifies project-ID membership/order and agreement between saved labels and campaign status.

| Diagnostic | Observed result | Interpretation |
|---|---:|---|
| Official training campaigns | 29,288 | Verified |
| Training success rate | 54.736% | Verified |
| Unparseable `funded date` values | 0 | Format valid; semantics unverified |
| Unparseable implied launch dates | 0 | Arithmetic valid; semantics unverified |
| Nonpositive durations | 0 | Passed |
| Median duration | 32 days | Descriptive |
| Earliest implied launch | 2009-04-25 | Provisional |
| Latest implied launch | 2012-05-29 | Provisional |
| Missing `location` values | 887 | Observed |
| Recognized U.S. state/DC suffixes | 26,945 | Geographic proxy, not country verification |
| Default engineered feature columns | 35 | Observed |
| Columns with `exclude_renamed=True` | 34 | Exactly `renamed` removed |

The implied launch-year counts were: 2009 **707**, 2010 **5,791**, 2011 **14,681**, and 2012 **8,109**. No official test-set scores were calculated in this notebook.

## 4. Preprocessing regression tests

**Result: 8/8 tests passed.** The suite in `tests/test_preprocessing.py` checks:

- Exclusion of known post-launch outcome columns.
- Pipeline compatibility with cross-validation.
- Exclusion of `renamed` without modifying `utils/features.py`.
- Correct target-encoding mappings for reordered/new observations.
- Protection against the prior same-index/cached-OOF inference bug.
- Deterministic OOF encoding and fold-specific smoothing priors.
- Rejection of invalid split counts.

From the repository root:

```bash
python3 -m unittest discover -s tests -v
```

These tests establish behavior for the tested cases; they do not validate source-field definitions or deployment performance. **Code-owner review remains pending.**

## 5. Feature-sensitivity comparisons — completed

**Execution: 10/10 model/configuration experiments completed.** Each experiment used the same **three-fold stratified CV** on the official training partition. The models were Logistic Regression (`max_iter=1500`) and HistGradientBoosting (`max_iter=150`, `learning_rate=0.06`, `random_state=42`). These settings **differ from V3's historical HGB model** and are diagnostic rather than an updated five-fold leaderboard.

| Configuration | Model | ROC-AUC | Accuracy | Balanced accuracy | F1 | Avg. precision | AUC fold SD |
|---|---|---:|---:|---:|---:|---:|---:|
| Current features, target encoding | Logistic | 0.7647 | 0.7055 | 0.6982 | 0.7422 | 0.7686 | 0.0052 |
| Current features, target encoding | HGB | 0.7671 | 0.7062 | 0.6967 | 0.7481 | 0.7744 | 0.0055 |
| Exclude `renamed`, target encoding | Logistic | 0.7647 | 0.7053 | 0.6981 | 0.7422 | 0.7685 | 0.0051 |
| Exclude `renamed`, target encoding | HGB | 0.7665 | 0.7065 | 0.6970 | 0.7483 | 0.7737 | 0.0054 |
| Exclude title/setup flags, target encoding | Logistic | 0.7607 | 0.7015 | 0.6940 | 0.7393 | 0.7650 | 0.0057 |
| Exclude title/setup flags, target encoding | HGB | 0.7641 | 0.7055 | 0.6961 | 0.7473 | 0.7712 | 0.0054 |
| Exclude timing **and** `renamed`, target encoding | Logistic | 0.7630 | 0.7015 | 0.6942 | 0.7388 | 0.7661 | 0.0054 |
| Exclude timing **and** `renamed`, target encoding | HGB | 0.7642 | 0.7057 | 0.6960 | 0.7480 | 0.7713 | 0.0051 |
| One-hot encoding, without `renamed` | Logistic | 0.7661 | 0.7070 | 0.6998 | 0.7435 | 0.7696 | 0.0049 |
| One-hot encoding, without `renamed` | HGB | 0.7662 | 0.7056 | 0.6959 | 0.7481 | 0.7731 | 0.0035 |

### Findings

1. **Removing `renamed` has a negligible effect on these CV metrics.** HGB ROC-AUC changes **0.7671 → 0.7665** (−0.0006); Logistic Regression remains **0.7647** to four decimal places. This supports removing a questionable feature without meaningfully impairing the baseline.
2. **Removing title/setup flags has a somewhat larger but still modest effect**: Logistic ROC-AUC changes **0.7647 → 0.7607** (−0.0040) and HGB **0.7671 → 0.7641** (−0.0030). This does not establish that those features are historically valid.
3. **Removing timing derivatives *and* `renamed`** gives ROC-AUC **0.7630** for Logistic and **0.7642** for HGB. Relative to *excluding `renamed` alone*, the additional timing removal changes ROC-AUC by **−0.0017** and **−0.0023**, respectively. Do not attribute the full difference from the current baseline solely to timing.
4. **One-hot encoding slightly helps Logistic Regression**: the one-hot/no-`renamed` result **0.7661** exceeds target-encoding/no-`renamed` **0.7647** by **0.0014** ROC-AUC; HGB slightly favors target encoding **0.7665 vs 0.7662**. No clear universal winner is established.
5. **The differences are small.** The measured fold SDs are approximately **0.0035–0.0057**. These SDs are not uncertainty intervals for pairwise differences; no paired statistical significance test was performed. Select an acceptable feature set based primarily on prediction-time validity, not tiny score differences.

**Provisional modeling candidates:** HGB with target encoding and `exclude_renamed=True`; Logistic Regression with one-hot encoding and `exclude_renamed=True`. Any timing-derived features remain conditional on source verification.

## 6. Chronological robustness — completed, provisional interpretation

**Execution: 3/3 comparisons completed.** The experiment sorted implied launch dates and selected the 80th-percentile launch timestamp as the cutoff. It trained Logistic Regression only on projects whose recorded end date was **before** that cutoff and evaluated later-launched projects.

- **Implied cutoff:** 2012-02-10 15:53:21 UTC.
- **Earlier completed campaigns used for training:** 21,351.
- **Later launches used for validation:** 5,858.
- **Excluded from either group:** 2,079 campaigns, including those that do not meet the strict training-completed or validation-launched conditions. This separation helps avoid using labels of campaigns still active at the cutoff.

| Configuration (Logistic Regression) | Train n | Later validation n | ROC-AUC | Avg. precision | Accuracy | Balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Current features, target encoding | 21,351 | 5,858 | 0.7904 | 0.7921 | 0.7235 | 0.7155 |
| Exclude `renamed`, target encoding | 21,351 | 5,858 | **0.7905** | **0.7921** | **0.7245** | **0.7167** |
| Exclude timing and `renamed`, target encoding | 21,351 | 5,858 | 0.7888 | 0.7893 | 0.7183 | 0.7116 |

### Interpretation

- The later validation sample does **not** show deterioration compared with the separate three-fold random-CV scores in this particular diagnostic, but the samples, fitting protocol, and time-composition differ, so the numbers should not be treated as directly interchangeable or proof of improved future generalization.
- Excluding `renamed` gives essentially the same discrimination and slightly higher threshold metrics. Removing timing derivatives changes temporal ROC-AUC by **−0.0017** relative to the no-`renamed` configuration.
- **This is a provisional time-ordering diagnostic, not a validated temporal holdout.** Both the ranking and the completed-before-cutoff rule depend on the unverified meaning of `funded date`. Historical values of scraped titles, reward tiers, and other features remain another limitation.
- Temporal checks here involved **Logistic Regression only**; there is no HGB temporal validation result yet.
- This comparison never used the canonical `X_test` rows/labels for scoring or selection.

## 7. Relationship to historical V3 baselines

V3's original five-fold classification comparison used the older preprocessing implementation and different estimator settings. Retain these as historical reference numbers unless the team agrees to rerun V3 with the corrected encoder.

| Model | Historical V3 ROC-AUC | Accuracy | F1 |
|---|---:|---:|---:|
| HistGradientBoosting | 0.7674 | 0.7068 | 0.7493 |
| Random Forest | 0.7650 | 0.7037 | 0.7462 |
| Logistic Regression | 0.7649 | 0.7046 | 0.7415 |
| Decision Tree | 0.7292 | 0.6796 | 0.7275 |
| KNN (k=25) | 0.6964 | 0.6551 | 0.7208 |
| Dummy majority | 0.5000 | 0.5474 | 0.7075 |

V3's raw scores should not be directly compared as controlled before/after estimates with the 05b scores: the CV folds and HGB hyperparameters differ in addition to the target-encoder change.

## 8. Decisions still required before definitive tuning

### 8.1 Source meaning and prediction-time validity

- [ ] Cite original source metadata defining `funded date`: scheduled deadline/end date, achieved-funding date, or another event.
- [ ] Verify whether `name`, `reward levels`, and other launch-related fields are original snapshots or potentially modified later.
- [ ] Document the intended point in time when the recommendation is generated.
- [ ] Exclude any feature group that cannot be shown to be available at prediction time; pipeline switches support experiments but cannot restore missing historical snapshots.

### 8.2 Target population

- [ ] Confirm with Challenge Advisor whether the 41,841-campaign settled population is acceptable or U.S.-only filtering is required.
- [ ] If a different population is required, identify defensible location/country evidence, regenerate the split, and recompute downstream results.

### 8.3 Team approval and reproducibility

- [x] Correct target-encoder transform/OOF behavior.
- [x] Pass 8/8 regression tests.
- [x] Verify `exclude_renamed` toggle.
- [x] Complete 10/10 three-fold sensitivity comparisons.
- [x] Complete 3/3 chronological diagnostic comparisons.
- [x] Interpret and document the numerical experiment results.
- [ ] Have the owner of `utils/preprocessing.py` review the patch and tests.
- [ ] Approve launch-valid features and encoding configuration for tuning.
- [ ] Decide whether to archive V3 unchanged or create an updated five-fold baseline with the corrected encoder.

## 9. Interpretation and model-selection rules

- Accuracy, ROC-AUC, F1, balanced accuracy, and average precision answer different questions. F1 for a majority-positive dummy can appear high despite ROC-AUC 0.5.
- Standard deviation across only three CV folds is not a generalization confidence interval, nor does it justify declaring a tiny apparent advantage statistically significant.
- Feature selection based on all training rows before CV (such as the historical top-12 experiment) is exploratory. Formal evaluation must nest selection inside the validation process.
- Predictions based on categorical target encodings require strictly fold-safe training. Prediction on new rows must not reuse training OOF values by index.
- A feature's predictive usefulness does not justify look-ahead leakage.
- Campaign feature associations are not evidence that changing those features will *cause* a campaign to succeed. A recommendation prototype must distinguish descriptive predictions from causal claims.
- Do not tune features, algorithms, thresholds, or hyperparameters using the official test set; use it only once the modeling protocol is finalized.

## 10. Provisional handoff to `06_evaluation_and_tuning.ipynb`

**Computational validation is finished. The validation issue can move to code-owner/Challenge Advisor review, but the prediction-time feature policy is not fully approved.**

Suggested starting configurations after that review:

1. **HGB:** `categorical_strategy="target", exclude_renamed=True`, with timing enabled **only if** `funded date` semantics and availability are verified.
2. **Logistic Regression:** `categorical_strategy="onehot", exclude_renamed=True`, with the same timing/availability condition.
3. Retain a no-timing conservative variant (`include_timing=False, exclude_renamed=True`) if the date assumption cannot be validated. Exclude other unverified feature families as needed.

A modest, training-only search across the leading models can then proceed, followed by a prespecified model-selection decision and one final evaluation against the canonical untouched test set.

### Reproduce checks

From the repository root:

```bash
source .venv/bin/activate
python3 -m unittest discover -s tests -v
python3 -m jupyter notebook
```

Open `notebooks/05b_pre_tuning_validation.ipynb`. The completed results above came from its saved outputs and need **not** be rerun merely to document them.

## Summary

The 05b computational phase is **complete**: 29,288 official training projects validated, 8/8 regression tests passed, 10/10 sensitivity comparisons finished, and 3/3 chronological experiments finished. Removing `renamed` costs little or no measured discrimination. One-hot encoding is a reasonable Logistic Regression candidate and target encoding remains reasonable for HGB. **Source-field semantics, U.S. scope, and teammate approval are still required before declaring any feature configuration final.**

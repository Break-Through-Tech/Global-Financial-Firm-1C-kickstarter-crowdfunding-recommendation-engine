# Pre-tuning validation and launch-time feature review

**Status (October 15 remediation):** U.S. training-only development may proceed under the frozen conservative policy below. Final held-out scoring waits for Anna and Zasha to agree on the evaluation protocol, ideally with advisor input. Historical 05b tables below are preserved outputs, not exact reproduction claims under the current encoder/environment.

**Scope:** Use the official split from `03_preprocess_data.ipynb`. All model selection and diagnostic experiments use the **training partition only**; retain the existing official split. The reserved test set has historical development exposure; see the evaluation policy below.

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
| `goal`, `duration`, `category`, `subcategory`, `location` | Retained conservative inputs | Assume availability and meaning around launch; historical snapshots, duration semantics and currency are unverified. |
| `levels`, `reward levels` and all reward-derived features | Excluded from active conservative models | Launch-time reward snapshot not verified; historical base matrices still include levels. |
| `name` | Excluded; not verified as launch snapshot | A subsequent scrape could capture changes after launch. |
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

Parth Rana has confirmed U.S. campaigns as the primary evaluation population (team-provided context). The active cohort is the existing state/DC **comma-qualified suffix proxy**, not verified registration country. Combined/global and explicit named international cohorts remain secondary. Keep all official memberships and ordering; advisor approval of the U.S. scope does not resolve historical exposure or source semantics.

### Frozen conservative U.S. feature policy

Use `include_timing=False, include_flags=False, include_reward=False, exclude_renamed=True`. Inputs are goal, duration, category, subcategory and location. Exclude pledged, backers, funded percentage, updates, comments, outcome-derived fields, unverified title/slug flags, renamed, rewards/levels, and reconstructed date/timing features. Retain training-fold category/subcategory log-goal references, log goal, goal per day, duration and categorical state proxy. The optional 05a extension is a documented experiment, not a newly approved baseline. Do not broaden families to raise AUC. A computationally leakage-aware pipeline is not a historically verified launch-time feature set.

### Historical development exposure and evaluation claims

`notebooks/01_explore_data.ipynb` and the original `notebooks/05_modeling.ipynb` load raw, undeduplicated data and call `get_data_split()` before the current cleaned-data protocol. That historical raw split overlaps itself by 54 project IDs. Its training rows include **1,630 current combined reserved test IDs**, including **1,486 U.S.** and **84 international** IDs. The current official splits are disjoint; this is prior target-aware exploration/model development exposure, not current membership overlap. The amount of resulting optimism has not been measured.

The final holdout cannot honestly be described as untouched by all prior development. Existing fold-contained training CV remains development validation; it does not repair the historical exposure or establish prospective performance. Rerandomizing the same explored dataset cannot guarantee independence and is not authorized. Retain the official split, label any future approved test score as historically exposed retrospective evaluation, and disclose the development/selection history. Do not claim prospective generalization from this historical dataset alone. An independent external or prospective evaluation would provide stronger evidence if available and separately authorized.

See [proposed evaluation policy](evaluation_decisions.md). No final test score may be calculated or published until Anna and Zasha agree on that protocol, ideally with advisor input.

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

**Historical result: 8/8 preprocessing tests passed.** The remediation suite adds coverage; current totals are in the remediation report. The suite in `tests/test_preprocessing.py` checks:

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

- [x] Advisor confirmed U.S. primary scope (user-provided confirmation).
- [x] Preserve existing state/DC proxy population and ordered official splits.
- [ ] Verify source registration-country/snapshot evidence; do not automatically expand cohorts.
- [ ] Agree final evaluation wording/protocol in light of historical holdout exposure.

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

## 10. Handoff to shared 06

Anna and Zasha can start training-only tuning with conservative U.S. Logistic Regression (one-hot) and HGB (cross-fitted target encoding); Random Forest is an optional existing comparator. Reuse official U.S. training IDs/labels and the existing builders. Put preprocessing, feature selection and hyperparameter/threshold selection inside training folds; use nested CV when estimating the performance of a selection procedure. Do not execute existing 06/07 cells blindly: their global paths and final-evaluation cells need a separate team review.

Final held-out scoring remains gated by the proposed protocol in [evaluation decisions](evaluation_decisions.md). Source-field meaning and historical exposure remain limitations even after agreement. Historical V3 supervised top-12 ranking occurred before CV and is exploratory. Preserve V3 and 05b historical tables; current-code reruns cannot retroactively turn those into corrected-encoder results. Their sensitivity and chronological numbers are not universally numerically reproduced in the current environment.

See [06 handoff](06_handoff.md), [remediation evidence](workflow_audit_remediation_2026-10-15.md) and [environment/artifact guidance](environment_reproducibility.md). Run `.venv/bin/python -m unittest discover -s tests -v` from the repository root.

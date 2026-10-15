# Workflow audit remediation — October 15, 2026

**Result:** targeted defects fixed; all six requested active training-only notebooks
executed and saved; **42 tests pass**. Anna and Zasha can begin training-only 06
collaboration using the frozen U.S. policy. **Final held-out scoring/publication
remains gated** by their evaluation-protocol agreement, ideally with advisor input.
This work does not establish pristine holdout independence or verified launch snapshots.

## Scope, original state and preservation

Read the complete local `docs/workflow_audit_2026-10-15.md`, especially §§3–9,
current utilities/tests, active notebooks, requirements and workflow/data docs.
Entry branch `main`, HEAD `52b5b12dd8140db71e15fb4a2afd3646e89309e4`.
`docs/international_only_workflow.md` was already staged; international notebooks,
utilities/tests and audit evidence were untracked. No applicable AGENTS.md was found.

The complete `git ls-files --stage` inventory is identical before/after. Nothing
was staged, committed, pushed, reset or deleted. The international document retains
its original staged blob and has the requested corrections only in its working copy
(`AM`). Existing untracked work is preserved. Original challenge documents remain.

Only `/docs/workflow_audit_2026-10-15.md` was added to `.gitignore`; the original
report remains local and unchanged. It was untracked, so no index removal was
needed. The remediation report, handoff and audit scripts remain eligible for Git.
An ignore rule prevents ordinary adds; explicitly forcing an add can override it.

Initial/final SHA-256 checks of existing raw/cleaned CSVs, ID files, matrices,
training labels and original joblibs show no byte changes except the authorized
international training-CV summary. `y_test.csv` was never opened, hashed or modified.
Reserved ID files were read with `usecols=['project id']` only. Historical combined
notebooks and their saved outputs remain unchanged. Original U.S./international
modeling notebook outputs and the old international CV summary are preserved in
ignored `data/Processed-Data/remediation_2026-10-15/historical_outputs/`.

## Implemented corrections mapped to the audit

| Audit finding | Correction and verification |
|---|---|
| §5/§8 conservative input contract | `add_base_columns` conditionally parses rewards/URL; `ask_features` conditionally computes reward-derived asks. Builder validates core columns and active family inputs in fit/transform. Defaults retain the historical feature composition. No essential values are fabricated. |
| §4/§8 territory suffix parsing | Match complete geographic suffixes longest-first, including comma-containing official names and full U.S. Virgin Islands aliases; recognize canonical territory identity before international assignment. |
| §4 bare CA mismatch | International classifier requires the same comma-qualified state/DC suffix as the U.S. mask; rowwise equality and direct intersection assertions replace summed-count reasoning. Bare CA is unknown. |
| §4 full-state evidence | Charlottesville, Virginia is explicit U.S. evidence outside the frozen proxy. Georgia/Congo/truncated strings stay unresolved. No city-based country guesses. |
| §6 duration presentation | Active U.S. 04b/international 04 use exact fractional intervals `(0,15]`, `(15,30]`, etc., including `>90`; underlying bins/data unchanged. Combined historical heatmap limitations are documented without rewriting historical outputs. |
| §5/§8 serialization | Preserve originals; regenerate three versioned base preprocessors from official training rows under the installed environment. Load/transform smoke checks match matrix values and ordered columns. |
| §3/§7–9 methodology/docs | Record historical exposure, freeze conservative primary U.S. policy, separate historical/active workflows and corrected-encoder provenance, document selected-location/currency/sparsity/snapshot limitations and top-level-only Drive sync. Create proposed policy and shared 06 handoff. |

The minimal synthetic prediction interface takes only goal, duration, category,
subcategory and location. Tests fit/predict with those fields, vary excluded fields
(including invalid reward/URL/date/outcome sentinels), validate missing active
columns, check training medians/scalers remain fixed, check default composition,
and serialize/load a conservative pipeline. International tests additionally verify
minimal inputs and numeric sparse/unseen-country fallback values.

## Reconciliation and geography

`reconcile.py` completed successfully: deterministic raw cleaning equals saved
combined CSV values/order; official seed-42 split replay exactly matches ordered
IDs; derivative cleaned rows equal their parent filters; derivative partitions
are ordered intersections; all current split IDs are unique/disjoint; training
labels agree with source status and `y_train.csv`.

| Cohort | Settled | Training | Reserved IDs | Before/after membership |
|---|---:|---:|---:|---|
| Combined | 41,841 | 29,288 | 12,553 | Unchanged |
| U.S. primary proxy | 38,420 | 26,945 | 11,475 | Unchanged |
| Named international | 2,064 | 1,427 | 637 | Unchanged |

An independent replay of the original classifier logic was compared with the
corrected classifier over all 45,815 parent rows. Both ordered modeling masks
are identical before/after and match the saved derivative cohorts. Only three
reporting classifications change:

| Project ID | Location | Before → after |
|---|---|---|
| 82500894 | St Croix, Virgin Islands, U.S. | explicit_us → us_territory |
| 1475984188 | Cruz Bay, Virgin Islands, U.S. | explicit_us → us_territory |
| 1846069134 | Charlottesville, Virginia | unknown → explicit_us (full-state, outside proxy) |

Corrected all-status accounting: **42,175 proxy + 1 full-state U.S. + 28 territories
+ 2,279 explicit non-U.S. + 1,332 unknown = 45,815**. This replaces the original
audit's classifier presentation (26 territories, 2 explicit U.S., 1,333 unknown);
the original report is preserved. Charlottesville is a live record, not a new
settled member. Neither U.S. nor international membership is expanded.

Historical raw split replay still shows 54 overlapping historical IDs and raw
training exposure of 1,630 current combined reserved IDs, 1,486 U.S., 84
international. This is historical development exposure, not present official
membership overlap. Its effect on optimism is unmeasured. Re-splitting does not
undo development history. See [proposed policy](evaluation_decisions.md).

## Tests and read-only audit execution

Both `python3 -m unittest discover -s tests -v` and the explicit
`.venv/bin/python -m unittest discover -s tests -v` completed with **42 passed,
0 failures, 0 errors**, increased from 29. All existing tests remain; the obsolete
Virginia-unknown assertion is replaced by an explicit full-state-U.S. assertion.

| Test module | Passed |
|---|---:|
| `test_preprocessing.py` | 8 |
| `test_us_scope.py` | 6 |
| `test_us_launch_features.py` | 3 |
| `test_international_scope.py` | 11 |
| `test_international_features.py` | 7 |
| `test_conservative_inputs.py` | 7 |

All four original read-only audit runners completed with exit 0:
`reconcile.py`, `eda_and_base_matrices.py`, `training_cv.py`, `combined_cv.py`.
Commands are in [environment guidance](environment_reproducibility.md).
The EDA runner independently passed all four EDA namespaces (including historical
combined 04 without saving it). Fresh fits match all three stored training base
matrices. All six saved train/test matrices are finite and have matching ordered
columns within cohorts: combined **29,288×65 / 12,553×65**; U.S.
**26,945×65 / 11,475×65**; international **1,427×190 / 637×190**.

The CV runners completed 30 existing training-only comparisons: 6 U.S. baselines,
2 U.S. extensions, 6 international baselines, 10 combined 05b sensitivity
comparisons and 6 current-code V3 baseline comparisons. Existing three chronological
training-partition diagnostics also completed. No top-12 search, new tuning,
final holdout metric, 06/07, deployment or final production fitting was run.
Full-precision metrics, data fingerprints, notebook execution evidence and the
three reporting reclassifications are saved in
[`audit_evidence/verification_results_2026-10-15.txt`](audit_evidence/verification_results_2026-10-15.txt).

## Clean-kernel notebook outputs

All cells were inspected for side effects before execution. Each ran from a new
kernel using `.venv/bin/python`, repository-root working directory, unchanged
estimator settings/seeds and serial thread limits. A restricted first attempt
could not create kernel sockets and saved nothing; after execution permission,
all six completed. Saves were atomic and only followed complete success.

| Notebook | Executed code cells | Saved successfully |
|---|---:|---|
| `notebooks/us_only/04_feature_engineering.ipynb` | 5 | Yes |
| `notebooks/us_only/04b_us_eda.ipynb` | 4 | Yes |
| `notebooks/us_only/05_modeling.ipynb` | 3 | Yes |
| `notebooks/us_only/05a_safe_feature_check.ipynb` | 3 | Yes |
| `notebooks/international_only/04_feature_engineering.ipynb` | 4 | Yes |
| `notebooks/international_only/05_modeling.ipynb` | 3 | Yes |

Final JSON/nbformat validation passed, counts are sequential from 1 per kernel,
and saved outputs contain no cell errors. No source/markdown/model-setting edits
occurred during execution; only the two documented duration-label changes preceded
it. International 05's sole local CSV write is its training-CV summary, backed up
before running. No protected cohort or split artifact was rewritten. Reusable
`execute_active_notebooks.py` records results and preserves originals on failure.

Current U.S. AUC / accuracy / F1: HGB **.7235/.6684/.7156**; Logistic
**.7199/.6684/.7111**; RF **.7117/.6583/.7036**; KNN **.6990/.6531/.7073**;
tree **.6867/.6491/.6958**; dummy **.5000/.5461/.7065**. All AUC references match
to four decimals. 05a's baseline/extended values and paired deltas also match to
four decimals: Logistic AUC .7199→.7235; HGB .7235→.7232. This is a joint four-feature
experiment and descriptive development evidence, not a significance test.

Changes from the previously saved active outputs, at their displayed precision:

| Model | AUC before → after | Accuracy | F1 | Balanced accuracy | Avg. precision | AUC fold SD |
|---|---|---|---|---|---|---|
| U.S. KNN | .6990 → .6990 | .6527 → .6531 | .7069 → .7073 | .6411 → .6415 | .7053 → .7056 | .0060 → .0057 |
| International KNN | .6507 → .6510 | .6503 → .6503 | .7455 → .7455 | .5969 → .5969 | .7254 → .7257 | .0225 → .0229 |
| International HGB | .6410 → .6405 | .6195 → .6160 | .7041 → .7015 | .5849 → .5810 | .7245 → .7267 | .0245 → .0295 |

Other active displayed model rows match to four decimals. International RF,
tree, Logistic and dummy AUCs remain .6685/.6584/.6573/.5000. These differences
are observed under the reviewed environment; their precise cause is not isolated.
No estimator was altered to force agreement.

Combined historical notebook outputs remain untouched. Current-code audit V3
AUCs are HGB .767606, RF .765522, Logistic .764718, tree .730078, KNN .696401,
dummy .500000; they do not reproduce all historical values and cannot recreate
the old encoder's software state. 05b current AUC pairs (LR/HGB) are
.764538/.766458 default, .764467/.766489 no renamed, .760543/.764177 no flags,
.763857/.766422 no timing/renamed, .766138/.766230 one-hot/no renamed.
Chronological diagnostic AUCs are .790355/.790438/.790409, versus historical
.7904/.7905/.7888. Date semantics remain unverified; these are not prospective
holdouts. No claim of universal historical score reproduction is made.

## Environment and artifact status

Reviewed CPython 3.12.3, sklearn 1.9.1, NumPy 2.5.3, pandas 3.0.5 and the
full installed stack are captured in `requirements-reviewed.txt`. Original
`requirements.txt` stays broad. No package changes or private-class monkeypatches.
The snapshot constrains recreation; fresh installation/portability was not tested.

| Artifact | Original load | Fresh versioned load + training transform |
|---|---|---|
| Combined base | Fails: sklearn 1.6.1 `_RemainderColsList` | Pass; 29,288×65 values and names match |
| U.S. base | Pass; training transform matches | Pass; 26,945×65 values and names match |
| International base | Fails: sklearn 1.6.1 `_RemainderColsList` | Pass; 1,427×190 values and names match |

Fresh artifacts and hash manifests are ignored local files under each cohort's
`compat/python-3.12.3_sklearn-1.9.1/`. `compatible_artifacts.py` fits only training
rows and verifies load/transform; originals provide the preserved historical copy.
These base artifacts include levels and are not conservative final model objects.
Serialization/matrix compatibility does not establish model-score reproduction.
See [exact paths and commands](environment_reproducibility.md).

## Remaining limitations and team decisions

Historical holdout exposure cannot be undone by code. Agreement on final-evaluation
protocol and wording remains outstanding; U.S.-scope confirmation does not resolve
it. Independent external/prospective evidence is preferable if available.

Historical launch snapshots, duration and `funded date` meaning, goal currency,
registration country and corrupted/truncated source strings remain unresolved.
International explicitly named locations are selected, sparse groups: 125 training
regions, 113 with <25 records, 438/467 country/category groups with <10. Country-relative
features retain potentially incomparable absolute goals and pooled fallbacks.
No new feature family or cohort member was admitted. A future conservative combined
comparison could be useful after review; no such new experiment was implemented.

Combined V3 top-12 selection before CV remains exploratory; existing combined
heatmaps still color sparse cells, now explicitly documented. Original 00/v2 path
and error-message robustness issues remain historical-workflow follow-ups; they
were not needed for the six active notebook runs and were left unchanged to
preserve historical notebooks. The Drive helper omits cohort subdirectories;
no authentication/upload occurred. No Streamlit implementation was added.

## Complete remediation file list

Existing tracked/staged files modified in the working tree:

- `.gitignore`
- `README.md`
- `data/Data_setup.md`
- `data/data_dictionary.md`
- `docs/model_validation.md`
- `docs/us_only_workflow.md`
- `docs/international_only_workflow.md` (original staged blob preserved)
- `notebooks/us_only/04_feature_engineering.ipynb`
- `notebooks/us_only/04b_us_eda.ipynb`
- `notebooks/us_only/05_modeling.ipynb`
- `notebooks/us_only/05a_safe_feature_check.ipynb`
- `tests/test_us_scope.py`
- `utils/features.py`
- `utils/preprocessing.py`

Preexisting untracked files modified:

- `utils/international_scope.py`
- `tests/test_international_scope.py`
- `tests/test_international_features.py`
- `notebooks/international_only/04_feature_engineering.ipynb`
- `notebooks/international_only/05_modeling.ipynb`

New remediation files:

- `requirements-reviewed.txt`
- `tests/test_conservative_inputs.py`
- `docs/evaluation_decisions.md`
- `docs/environment_reproducibility.md`
- `docs/06_handoff.md`
- `docs/workflow_audit_remediation_2026-10-15.md`
- `docs/audit_evidence/compatible_artifacts.py`
- `docs/audit_evidence/execute_active_notebooks.py`
- `docs/audit_evidence/verification_results_2026-10-15.txt`

Local ignored output changes: international `training_cv_baselines.csv` refreshed;
three new versioned base joblibs and three sibling manifests; preserved historical
modeling notebooks/CV CSV under `remediation_2026-10-15/historical_outputs/`.
Temporary logs/backups also remain in `/private/tmp/team1c-remediation/`.
Other preexisting untracked international/audit files are unchanged; they are not
new remediation work. The original audit file is ignored and preserved.

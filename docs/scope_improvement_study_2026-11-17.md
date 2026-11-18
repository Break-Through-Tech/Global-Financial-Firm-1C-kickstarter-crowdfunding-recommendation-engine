# Separate scope improvement study

November 17, 2026. This covers the four requested improvement directions. All new code/results are separate from the application. No serving source, production model, metadata, dependency pin, threshold, official split or reported production result was changed. No reserved-label file was opened, and no release, commit, push or message to teammates was sent.

## Decisions for team review

| Track | What was completed | Result | Deployment decision |
|---|---|---|---|
| U.S. city candidate | Verified the existing v2 hash and full-city interface on 100 training examples; reused its recorded validation | 67.70% accuracy vs v1 66.93%; raw/validated probabilities match exactly in the sample | Worth reviewing as a separate version; state-only compatibility still needs an explicit UI policy |
| International/mixed models | Fixed-setting comparison of HGB, XGBoost, CatBoost and LightGBM against the current Logistic baseline, with identical approved inputs/folds | Mixed has modest candidates; international has no candidate passing all guardrails | Prepare mixed candidates for review only; do not promote an international winner based on accuracy alone |
| Funding/reward/title readiness | Primary-source review of units, original provenance and live-edit rules | Exact monetary basis and initial title/reward snapshots remain unverified | Keep these features out of the public improvement candidates; no fabricated FX or planned-update substitute |
| Stronger external evaluation | Reviewed newer primary data sources and a bounded recent schema audit where feasible | See the external readiness report for actual verified fields and remaining alignment/access gaps | No external accuracy claim or reserved evaluation; first establish a permitted, time-aligned external cohort |

Sources: [peer comparison](peer_accuracy_review_2026-11-17.md), [feature-readiness audit](scope_feature_readiness_2026-11-17.md), [external/temporal readiness](external_evaluation_readiness_2026-11-17.md). The latter two tracks are data-validity investigations. They cannot be completed by selecting a higher CV score.

## Controlled model comparison

The experiment uses only existing official international (1,427) and mixed (29,288) training IDs in saved order. Their digests must match the current model metadata. The current Logistic baseline is reproduced to absolute tolerance 1e-12 for accuracy, AUC, Brier and failure recall before accepting results. Both baselines reproduced.

All algorithms receive the same four columns: duration, category, subcategory and normalized listed country/region. Monetary, reward/title, outcome/engagement and inferred calendar inputs remain excluded. All use the same fold-fitted duration median/scaler and categorical imputer/one-hot encoding with min_frequency=10 and unknown-value fallback. CatBoost is deliberately tested on this common numeric encoding; this is not a claim about its best possible native-categorical configuration.

Settings are fixed from existing baseline/comparison work, not a new hyperparameter search:

| Model | Fixed settings |
|---|---|
| Reference Logistic | max_iter=2000, existing baseline defaults |
| HGB | max_iter=160, learning_rate=.06, seed42 |
| XGBoost | 250 trees, depth3, learning_rate=.05, histogram method, seed42 |
| CatBoost | 250 iterations, depth4, learning_rate=.05, L2=5, Ordered boosting, seed42 |
| LightGBM | 250 trees, 15 leaves, learning_rate=.05, min_child_samples=43, deterministic, seed42 |

Five outer stratified folds at shuffle seed42 match the current historical baseline evaluation. Each outer fitting partition also has two inner stratified folds at seed100+outer_fold. Inner scores alone choose an algorithm; settings and threshold .5 remain fixed. The outer-evaluated inner-family-selection row describes that selection procedure, while the individual algorithm rows describe fixed configurations. Full results include pooled out-of-fold metrics, paired fold results, inner metrics, timing and split-position digests. No raw project rows, per-project labels or IDs are exported.

The protocol was saved before fitting in [protocol.txt](scope_improvement_results/protocol.txt). An algorithm must gain at least .003 accuracy over current Logistic, exceed majority accuracy by .003, lose no more than .002 AUC or .01 failure recall, and improve accuracy in at least three of five outer folds to be eligible for review. These are practical checks, not significance tests or automatic deployment authorization.

## International results

| Algorithm | Accuracy | ROC-AUC | Failure recall | Eligible for review |
|---|---:|---:|---:|---|
| Current Logistic | 60.41% | .5893 | 30.52% | Reference |
| HGB | 57.74% | .5698 | 37.16% | No |
| XGBoost | 61.60% | .5838 | 23.88% | No |
| CatBoost | 61.18% | .5908 | 21.54% | No |
| LightGBM | 60.48% | .5732 | 33.93% | No |
| Inner family selection | 60.55% | .5878 | 29.08% | No |
| Always-majority class | 60.97% | — | 0.00% | Reference |

XGBoost has the highest observed accuracy, but loses about 6.64 percentage points of failure recall and .0055 AUC relative to Logistic. CatBoost loses about 8.98 recall points and does not exceed majority accuracy by the required .3 points. Selecting either from accuracy alone would violate the prespecified checks. HGB finds more failures but loses accuracy/AUC. No acceptable replacement was demonstrated under these fixed settings. This does not prove that better models or data are impossible; it does rule out presenting this run as an international improvement.

The inner selector chose CatBoost in one outer fold and Logistic in four. Its 60.55% accuracy is below majority. Keep the international model explicitly demonstration-only. No new international artifact was saved.

## Mixed results

| Algorithm | Accuracy | ROC-AUC | Failure recall | Positive accuracy folds vs Logistic | Eligible for review |
|---|---:|---:|---:|---:|---|
| Current Logistic | 62.23% | .6631 | 50.41% | — | Reference |
| HGB | 62.75% | .6698 | 53.44% | 4/5 | Yes |
| XGBoost | 62.87% | .6711 | 53.35% | 5/5 | Yes |
| CatBoost | 62.88% | .6705 | 47.97% | 4/5 | No |
| LightGBM | 63.02% | .6722 | 53.59% | 5/5 | Yes |
| Inner family selection | 62.75% | .6681 | 51.01% | 4/5 | Yes |
| Always-majority class | 54.74% | — | 0.00% | — | Reference |

LightGBM is the strongest observed fixed row: +0.79 percentage points accuracy, +.0091 AUC and +3.18 failure-recall points over Logistic. HGB gains +0.52 accuracy points and +3.03 recall points using only the existing scikit-learn runtime. LightGBM's additional accuracy over HGB is about .27 points, so runtime complexity matters. XGBoost is another eligible result but adds a dependency. CatBoost's higher accuracy does not offset its 2.44-point failure-recall loss under the chosen rules.

The inner selector chose LightGBM twice, CatBoost twice and Logistic once. Its separate 62.75% accuracy result is more relevant to assessing this restricted selection procedure than treating the highest outer row as an unbiased winner estimate. Prior development exposure and reviewing these same folds still limit independence. Overlapping folds do not justify a significance claim or a confidence interval computed from five fold scores.

## Research artifacts, not deployment changes

Two separate full-training candidates were fitted only after the mixed review checks passed:

- `models/research_scope_candidates/mixed_hgb_candidate_v2.joblib`
- `models/research_scope_candidates/mixed_lightgbm_candidate_v2.joblib`

Each has sibling metadata with schema, class mapping, fixed .5 threshold, training-ID digest, parameters, environment, development metrics, limitations and hash. Both were verified for exact prediction equality on 32 representative rows before/after serialization. They remain ignored and are not attached to a release, staged, published or loaded by Streamlit. Existing production files were hash-checked before/after both scripts.

The candidate nomination used outer development results, so candidate metrics remain development evidence. No independent superiority claim is made. The app still loads the original mixed Logistic artifact and original U.S. HGB v1. LightGBM also requires its separately pinned runtime dependency and a Linux-host smoke check; neither requirement was added to production requirements. HGB uses the existing runtime, but still needs explicit version/loader and team approval before activation.

Candidate SHA-256:

```text
mixed HGB:      7f136f9bc28a082a08ca650f03935b533e9f3d487073f725e2ec994cd373904c
mixed LightGBM: ca76eb1bade4d3a93d7e0102548e13d4dd97cf1310410ee595ced22f948ac3ba
```

## Reproduce and inspect

The existing `.venv-model-comparison` overlays the reviewed Python3.12 environment with XGBoost3.4.1, CatBoost1.2.10 and LightGBM4.7.0. It is separate from the application runtime; see [its existing dependency file](../requirements-model-comparison.txt). No production package installation was performed.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1 .venv-model-comparison/bin/python -m unittest discover -s tests -p test_scope_improvement_study.py -v
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1 .venv-model-comparison/bin/python scripts/run_scope_improvement_study.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1 .venv-model-comparison/bin/python scripts/prepare_scope_research_candidates.py
```

The comparison refuses to overwrite its output directory; the candidate script refuses existing candidate files. They do not replace existing artifacts. To reproduce a comparison, use `--output` with a new directory and review its results separately. The candidate preparation script intentionally uses the reviewed default run; a different run requires explicit source/result review rather than silently changing the selected study.

Outputs: `docs/scope_improvement_results/` contains protocol/environment, cohort and split digests, city-interface audit, per-scope summary/fold/inner tables, combined summary and integrity verification. The original U.S. v2 score is reused with provenance, not claimed as a new test result. The majority baseline's pooled fold-prior probabilities can have AUC slightly different from .5; its reported reference here is majority-class accuracy, not a useful ranking model.

## Handoff for Anna and Zasha

Review U.S. city v2's full-city input policy and the mixed HGB/LightGBM tradeoff. Preserve v1 and current modes until the team agrees on a concrete candidate version, requirements, validation limitations and rollback. For stronger evidence, resolve the external cohort's fields/snapshots before any outcome-based evaluation. Do not use the existing approval-gated reserved set as a tuning loop.

The defensible finding is a modest potential U.S./mixed improvement, no accepted international improvement, and unresolved source/evaluation work. There is no evidence here for matching another team's 81% pre-launch accuracy or for claiming present-day accuracy from these historical folds. No teammate notification was sent; this file is the review handoff.

## Final verification, November 18

All 96 repository tests pass in the combined reviewed-model/dashboard test environment, including the optional boosting packages. The initial attempt to combine only the optional-package directory with the clean dashboard environment lacked CatBoost's `six` dependency; adding the existing reviewed-base package directory resolved that test-environment issue without installing or changing any production packages. Critical lint and whitespace checks pass. Full test command used:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1 PYTHONPATH="$PWD/.venv-model-comparison/lib/python3.12/site-packages:$PWD/.venv/lib/python3.12/site-packages" /private/tmp/team1c-deploy-venv/bin/python -m unittest discover -s tests -v
/private/tmp/team1c-deploy-venv/bin/python -m flake8 scripts/run_scope_improvement_study.py scripts/prepare_scope_research_candidates.py tests/test_scope_improvement_study.py --count --select=E9,F63,F7,F82 --show-source --statistics
git diff --check
```

The four new study tests check accuracy/AUC/failure-recall guardrails, the full-city contract, disjoint fitting/validation rows for every CV prediction, and identical approved feature schemas with finite probabilities for all installed algorithms. Both current scope baselines reproduced exactly. The study's integrity manifest still matches all protected serving source and production model/metadata hashes.

A follow-up city-input check confirms case normalization gives identical probabilities and an unseen city receives log frequency zero with a finite prediction. This verifies the fallback operates; it does not establish accuracy for unseen cities. See `city_input_followup.txt`.

The newer-data track completed an actual bounded schema read of the November 16, 2026 Web Robots archive: HTTP206, 512KiB compressed maximum, 256KiB decoded maximum. Its 42 columns include goal/currency and launch/deadline, with nested category parent-name and listed-location state keys. No project IDs, outcome values or raw rows were retained or used for scoring. This makes a harmonization pilot concrete, while snapshot timing, units and dataset-use rights remain unresolved. No modern accuracy number was computed.

New tracked-eligible study files are the two scripts, one test module, this report, the feature/external-readiness reports and `docs/scope_improvement_results/`. The earlier peer report remains a separate pending document. No tracked application file changed. Research candidate binaries/metadata remain ignored. Nothing is staged, committed, pushed, merged, published or deployed.

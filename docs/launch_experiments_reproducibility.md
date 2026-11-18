# Reproduce launch feature experiments

Run from the repository root. Keep all official memberships and existing outputs. No Drive access is needed when the local files below exist. References are already extracted locally under `reference/1A/` and `reference/1B/`; `.gitignore` excludes ZIPs, extracted repositories and nested `.git` directories.

## Environment and local inputs

Use the existing reviewed `.venv`: Python 3.12.3, sklearn 1.9.1, NumPy 2.5.3, pandas 3.0.5, joblib 1.6.0. For a separate environment, use Python 3.12.3, `python -m venv .venv-launch`, then `.venv-launch/bin/python -m pip install -r requirements-reviewed.txt`. This pinned snapshot was captured on macOS arm64; a fresh installation on another platform has not been verified. Do not change the environment used by frozen v1 merely to load incompatible historical artifacts. XGBoost/LightGBM are absent, so the optional comparison was skipped without installing dependencies.

Required files:

- `data/Raw-data/DSI_kickstarterscrape_dataset.csv` for source checks.
- `data/Processed-Data/DSI_kickstarterscrape_dataset_cleaned.csv` and `train_project_ids.csv`.
- The corresponding cleaned CSV and `train_project_ids.csv` in `us_only/` and `international_only/`.
- All three cohorts' `test_project_ids.csv`, opened with `usecols=['project id']` only for membership checks. No `y_test.csv` is needed or opened.
- `models/us_primary_model_v1.joblib` and sibling `.metadata.json` for baseline consistency checks.

Obtain these existing frozen files through the team's agreed local/Drive handoff if missing. The current top-level Drive helper omits cohort subdirectories; do not flatten them or run 03 to fabricate replacement splits. Credentials are local and ignored. Request missing frozen artifacts rather than regenerate official memberships.

## Current rerun order

The new experiment uses the frozen rows directly and does not depend on rerunning historical notebooks:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/review_launch_inputs.py
.venv/bin/python scripts/execute_extended_features.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/prepare_us_city_candidate.py
.venv/bin/python -m flake8 utils tests scripts --extend-exclude=.venv,.venv-launch --count --select=E9,F63,F7,F82 --show-source --statistics
```

The executor starts a clean kernel with serial numerical threads and repository-root working directory. Allow local kernel sockets if the sandbox blocks Jupyter startup. It saves the notebook only after full success. On this machine the new notebook finishes within a few minutes; timing varies. The candidate command is optional, uses completed evaluation tables, writes only the new ignored candidate namespace and verifies an existing candidate instead of replacing it. It never saves reward/title models. No reserved evaluation command belongs in this development sequence.

Generated review outputs are `docs/launch_feature_results/input_review.txt`, `summary.csv`, `outer_folds.csv`, `inner_comparisons.csv`, `paired_accuracy.csv` and `paired_accuracy.png`. The notebook embeds computed tables and a figure. `candidate_serialization.txt` records local candidate verification. Candidate model/metadata stay ignored in `models/`.

## Workflow map and construction order

| Workflow | Historical construction order | Current use |
|---|---|---|
| Global | `00_load_data` → `02_clean_data` → `03_preprocess_data` → `04_feature_engineering` → `05_modeling_v3`; 01 is exploration | 00/02/03 create or replace local artifacts. Preserve frozen outputs. Original 01/05 use the obsolete raw evaluation; 05/v2/v3, combined 04, 05b and combined 06 are archival/diagnostic, not a deployable global pipeline. No global rerun or deployment is authorized by a favorable U.S. score. |
| U.S. construction | Parent cleaned data and global split → `us_only/02_clean_data` → `us_only/03_preprocess_data` → `us_only/04_feature_engineering` → `us_only/05_modeling` | 02/03 construct derivative artifacts and must not be rerun over frozen files. Existing base matrices are not the input to current 06 training. |
| U.S. development | `us_only/04b_us_eda` → `us_only/05a_safe_feature_check` → model-specific tuning notebooks → `06_us_accuracy_experiments` → `06_us_validation` → `07_interpretability` → `06_us_feature_audit` | Completed evidence. Existing validation notebook writes v1 and metadata: **do not rerun it in this task**. Tuning/accuracy/audit informed earlier selection. The new extended notebook is the only required notebook now. 07 explains baseline candidates, not the new city candidate. |
| Optional international | Parent frozen global artifacts → `international_only/01_location_audit` → `02_clean_data` → `03_preprocess_data` → `04_feature_engineering` → `05_modeling` | Optional selected-location exploration, not global. 01–03 write cohort/split artifacts; do not rerun them over frozen files. 04 then 05 can be reviewed manually using frozen artifacts; 05 writes its historical training-CV CSV, so preserve it before any separately authorized refresh. Not required here. |
| Reserved evaluation | `08_us_reserved_evaluation` | Unexecuted approval-gated retrospective protocol. Do not run for feature selection or claim a pristine holdout. |

The table gives dependency order, not permission to blindly rebuild datasets. Recreating from scratch requires a separate destination and exact frozen-ID comparison, not replacing these files. No global experimental notebook was created because currency semantics remain unresolved. Global and U.S. metrics must stay separate, and global includes U.S.

## Validation and consistency

The source audit confirms unique/disjoint memberships, derivative ordering, official U.S. labels against training statuses, parser agreement and missingness. Current checks do not replay the original random split using reserved statuses; the prior exact replay is documented in the October 15 remediation.

The new notebook fits v1's exact configuration on official training rows and checks exact probabilities against the saved trusted v1 on all 26,945 rows. It checks representative current `USPredictor` calls and unchanged model/metadata hashes. The candidate command checks exact predictions and probabilities before/after serialization on all training rows. Production location normalization is different from city-frequency input needs; coordinate a separate full-city validator with Zasha before activating v2.

New tests cover baseline equivalence/serialization, reward parsing and units, title definitions, missing values, frozen city references and category medians, unseen cities, excluded-outcome invariance and promotion guardrails. Existing target-encoding regression tests remain intact. Training-fold differences are descriptive; selection across many configurations and repeated use of these data limit generalization claims.

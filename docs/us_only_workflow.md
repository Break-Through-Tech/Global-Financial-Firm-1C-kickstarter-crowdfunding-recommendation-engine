# Provisional U.S. cohort workflow (Team 1C)

## Purpose and boundaries

The challenge's Milestone 1 calls for U.S.-based campaigns. The existing
`notebooks/00-05b` experiments use the broader cleaned cohort and are **not
replaced**. The new `notebooks/us_only/01-05` notebooks create a separate
provisional U.S. *state-suffix candidate* cohort, reuse the original cleaned
dataset, and build training-only baseline models. The main `06` and `07`
notebooks remain reserved for the team's shared tuning, final evaluation,
and interpretation work.

**Important: there is no verified country column in this historical CSV.**
The filter accepts locations ending in one of the 50 U.S. state codes or DC,
for example `San Francisco, CA`. This is a transparent heuristic, not proof
of country. It excludes unclear/missing locations and some U.S. territories.
Parth Rana has confirmed U.S. primary scope. That confirmation does not verify
registration country, historical snapshots, or holdout independence. Keep the
existing cohort membership; do not add full-state names or territories.

## First-time setup

From the repository root, activate your existing environment:

```bash
source .venv/bin/activate
python3 -m pip install -r requirements-reviewed.txt
python3 -m unittest discover -s tests -v
python3 -m jupyter notebook
```

`01` and `02` need the **existing** globally cleaned file:

```text
data/Processed-Data/DSI_kickstarterscrape_dataset_cleaned.csv
```

`03` also needs the **existing global** official split files:

```text
data/Processed-Data/train_project_ids.csv
data/Processed-Data/test_project_ids.csv
```

If they are missing, restore them from your team's shared processed data or
use the documented reconstruction workflow only after team review; do not
recreate or reshuffle official splits as part of routine setup. The patch **cannot** download
private data, and it doesn't change Drive folders or authentication.

## Run these notebooks **in order**

| Notebook | Purpose | Outputs |
|---|---|---|
| `notebooks/us_only/01_explore_data.ipynb` | Audit which locations match state suffixes; review exclusions | None |
| `notebooks/us_only/02_clean_data.ipynb` | Reuse global cleaning and select candidate U.S. projects | U.S. candidate cleaned CSV |
| `notebooks/us_only/03_preprocess_data.ipynb` | Intersect **existing global** train/test IDs with candidate U.S. project IDs, fit base preprocessor on training only | U.S. train/test IDs, labels, matrices, joblib |
| `notebooks/us_only/04_feature_engineering.ipynb` | Inspect conservative features and outcomes using **training data only** | None |
| `notebooks/us_only/05_modeling.ipynb` | Five-fold CV of six baselines using pipelines refitted in each fold | Notebook metrics (no official test evaluation) |

All generated U.S. files are stored under
`data/Processed-Data/us_only/` (already covered by existing `.gitignore`).
Global paths and notebooks are unchanged.

## Why reuse global split membership?

The original global test set was saved in `03` and not used for final model
selection. Creating a fresh U.S. random split could put earlier global
*training* projects into a new U.S. test set. Restricting the **existing**
partition instead preserves the current split boundary and allows a later
global-vs-U.S. model comparison on exactly the same U.S. holdout rows. The
resulting subset may not be perfectly stratified; inspect class balance.

## Conservative feature policy

The new 05 baseline sets `include_timing=False`, `include_flags=False`,
`include_reward=False`, `exclude_renamed=True`. This omits unverified
inferred launch timing, title/slug changes, and reward-string features,
while preserving duration, goal and category/state features. New options
in `utils/preprocessing.py` default to prior behavior and therefore do
**not** silently change the original global notebooks. Learned category
medians and encoders are fit inside each CV training fold.

Even retained goal, duration, category, subcategory and location rely on
assumptions about availability and meaning around launch. Computational fold
isolation does not verify a historical launch-time feature set.

The historical meaning of `funded date` and the launch-time availability
of reward/title fields are still unverified. If verified, compare an
expanded feature configuration in training-only CV before final selection.

## Shared 06 and evaluation boundary

Run active notebooks in this order after restoring existing processed data:
04 feature engineering, 04b U.S. EDA, 05 baselines, then optional 05a feature check.
01 is a scope audit; 02/03 are data preparation, not routine verification reruns.
The historical combined raw 01/original 05 protocol is not the current model protocol.

Training-only 06 work can proceed with the frozen conservative policy and
fold-contained or nested selection. The existing official U.S. reserved test
includes 1,486 IDs used in historical raw training; it is historically exposed,
although current partitions are disjoint. Do not rerandomize to claim independence.
Final scoring/publication waits for Anna and Zasha's evaluation agreement, ideally
with advisor input. See [evaluation decisions](evaluation_decisions.md) and
[handoff](06_handoff.md). No final test labels are needed for 04/04b/05/05a.

The Drive upload helper syncs only top-level processed files, not `us_only/`
or `international_only/`. Restore cohort folders separately through the team's
approved data-sharing process. Do not flatten colliding filenames or regenerate
splits. See [environment guidance](environment_reproducibility.md).

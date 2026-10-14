# Optional U.S.-only EDA and feature check

This add-on **does not edit** existing `notebooks/us_only/01-05`, any global
notebook, or shared `utils/features.py` / `utils/preprocessing.py`.

## Run order

After existing `us_only/01-05` have completed:

1. `notebooks/us_only/04b_us_eda.ipynb` — state/subcategory/goal/duration
   associations, including Wilson intervals, using U.S. **training rows only**.
2. `notebooks/us_only/05a_safe_feature_check.ipynb` — five-fold CV comparing
   original conservative Logistic Regression and HGB baselines against the
   *same* models with four extra launch-time candidate features.

Run `python3 -m unittest discover -s tests -v` before the notebooks. These
notebooks use the already-generated `data/Processed-Data/us_only/` data. No
Google Drive sync or re-splitting is needed. Neither notebook loads `test_project_ids.csv`
or any test labels. No saved global/U.S. outputs are overwritten.

## Extra candidate features

- `goal_vs_state`: `log1p(goal)` minus a state-level training-fold median.
  Unseen/missing states use the corresponding overall training-fold median.
- `goal_round_1000`: whether the funding goal is a multiple of $1,000.
- `duration_exact_30` / `duration_exact_60`: campaign duration flags.

These features are computed inside a **cloneable transformer** under
`utils/us_launch_features.py`. The original `FeatureBuilder` remains the default
for existing code. No outcome-dependent statistics are added.

All inputs still require historical launch-time validity. The geographic
filter remains a state-code suffix **proxy**, not a verified country variable.
Small CV changes are not statistical proof of superiority. Do not update the
final model based on test-set results; review with Zasha before `06`.

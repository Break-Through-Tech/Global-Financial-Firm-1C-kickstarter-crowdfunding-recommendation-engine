# U.S. training-only model comparison

Executed October 27, 2026 from fresh kernels with the reviewed local environment. Every run used the official U.S. training IDs, 26,945 rows, success rate 54.615%, and the conservative policy: `include_timing=False`, `include_flags=False`, `include_reward=False`, `exclude_renamed=True`. Preprocessing, target encoding, imputation, scaling, and one-hot encoding were refit within each CV training fold. No reserved U.S. IDs or labels were loaded.

## Results

All scores below are ordinary five-fold tuning-CV estimates. They are useful for development, but the selected score can be optimistic because the same folds select the hyperparameters. Accuracy and F1 use the default 0.5 threshold.

| Model | Search trials | Baseline AUC | Tuned AUC | AUC SD | Accuracy | F1 | Best-trial train AUC | Gap | Mean fit sec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HistGradientBoosting | 15 | 0.7235 | **0.7255** | 0.0087 | 0.6694 | 0.7166 | 0.7464 | 0.0209 | 0.443 |
| Logistic Regression | 15 | 0.7199 | 0.7201 | 0.0054 | 0.6684 | 0.7114 | 0.7252 | 0.0051 | 0.217 |
| Random Forest | 5 | 0.7117 | 0.7209 | 0.0064 | 0.6672 | 0.7148 | 0.7881 | 0.0672 | 4.368 |
| Extra Trees | 5 | 0.7191* | 0.7205 | 0.0061 | 0.6685 | 0.7155 | 0.7579 | 0.0374 | 1.878 |

`*` Extra Trees has no previously verified U.S. baseline in the handoff. The 0.7191 reference is a new conservative 250-tree, minimum-leaf-5 run under this notebook's protocol, not a historical approved baseline.

The tuned HGB score is numerically highest, but the comparison is not a final model-selection result. HGB and Logistic used 15 trials; RF and Extra Trees used five because tree searches exceeded the practical runtime budget. None of the searches used nested CV.

Best settings were HGB: learning rate 0.0432, 219 iterations, 14 leaf nodes, minimum leaf 43, no depth limit, L2 0.0027. Logistic: `C=0.1695`, no class weighting. RF: 257 trees, depth 10, minimum leaf 3, minimum split 9, square-root feature sampling, no class weighting. Extra Trees: 174 trees, depth 16, minimum leaf 20, minimum split 8, 0.5 feature sampling, no class weighting.

## Preliminary interpretability

`07_interpretability.ipynb` used out-of-fold predictions and fold-specific validation permutation importance for conservative baseline Logistic Regression and HGB. OOF ROC-AUC was 0.7198 for Logistic Regression and 0.7233 for HGB. The largest mean raw-input permutation signals were:

- HGB: `goal` 0.0902, `subcategory` 0.0619, `location` 0.0254, `duration` 0.0251.
- Logistic Regression: `goal` 0.0870, `location` 0.0296, `category` 0.0283, `subcategory` 0.0249, `duration` 0.0149.

These are model-behavior diagnostics on training-validation folds. They are not causal effects, do not justify changing a campaign feature, and do not support causal country or state comparisons.

## Historical combined experiment

Zasha's `notebooks/06_evaluation_and_tuning.ipynb` remains unchanged. Its reported CatBoost, XGBoost, and LightGBM scores of approximately 0.7724, 0.7716, and 0.7705 came from the historical combined/raw-data workflow, including a fresh raw split and later test-label evaluation. They are not U.S.-only results and are not directly comparable with this table. That notebook should not be rerun as part of this workflow.

LightGBM, XGBoost, and CatBoost were checked in the reviewed environment and were unavailable, so no U.S.-only scores or extra notebooks were created for them.

## Limitations and next decision

The U.S. cohort is the approved state/DC suffix proxy, not verified registration country. Source snapshot provenance, currency comparability, duration meaning, and the semantics of `funded date` remain limitations. Final test evaluation remains gated by Anna and Zasha's evaluation-policy decision and must be disclosed as historically exposed retrospective evidence if approved. An independent prospective or external evaluation would be stronger.

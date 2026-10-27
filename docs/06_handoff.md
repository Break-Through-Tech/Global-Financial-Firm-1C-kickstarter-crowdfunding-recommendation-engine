# Anna + Zasha: shared 06 handoff

**Ready for training-only collaboration. Final held-out scoring remains gated.**
Parth confirmed U.S. primary scope. Agree on the proposed
[evaluation policy](evaluation_decisions.md), ideally with advisor input, before
calculating or publishing final test scores. No 06/07 was executed in remediation.

Use `data/Processed-Data/us_only/DSI_kickstarterscrape_dataset_cleaned.csv`, ordered
`train_project_ids.csv` and its `target` column (also `y_train.csv`). Select rows
by saved IDs and assert training-status agreement as in U.S. 05. There are
**26,945 training campaigns** out of **38,420 settled**, with **11,475 reserved**.
Keep official IDs/order; do not resplit. The comma-qualified state/DC suffix is a
listed-location proxy, not verified registration country. Do not open `y_test.csv`.

Freeze `include_timing=False, include_flags=False, include_reward=False,
exclude_renamed=True`. Approved prediction inputs: **goal, duration, category,
subcategory, location**. Retain training-fold engineered goal references and
state categories. Exclude outcomes/engagement, title/slug/renamed, rewards/levels,
and reconstructed timing. Even retained inputs assume launch-time availability
and meaningful historical snapshots; duration, currency and `funded date` source
semantics are not verified. Fold isolation does not establish launch provenance.

Current saved five-fold training CV (seed 42, unchanged settings):

| Candidate | ROC-AUC | Accuracy | F1 |
|---|---:|---:|---:|
| HGB, target encoding | .7235 | .6684 | .7156 |
| Logistic Regression, one-hot | .7199 | .6684 | .7111 |
| Random Forest, target encoding | .7117 | .6583 | .7036 |
| Dummy | .5000 | .5461 | .7065 |

Start with **HGB and Logistic Regression**; RF is an optional existing comparator.
05a's existing joint four-feature experiment (state-relative goal, round-1000
goal, exact-30/exact-60 duration) gives Logistic AUC **.7235** versus **.7199**;
HGB **.7232** versus **.7235**. It is optional, not four isolated ablations or
proof of superiority. Fold SD is not a confidence interval. Account for previous
candidate selection when designing nested validation; do not rerun selection
outside validation folds.

Reuse `utils/preprocessing.py` (`make_pipeline`, `drop_leaks`, corrected
`TargetEncoder`), `utils/us_scope.py` and optionally `utils/us_launch_features.py`.
The minimal-input interface now works without reward strings or URLs; regression
tests cover excluded-field invariance and fitted-state isolation. Read U.S. 04 →
04b → 05 → optional 05a; all six recommended U.S./international training notebooks
were fully executed and saved. Existing 06 code still needs a separate U.S.-path
and final-evaluation-cell review before execution.

Use CPython 3.12.3 and `requirements-reviewed.txt`. See
[artifact guidance](environment_reproducibility.md): all three fresh namespaced
base preprocessors load and match training matrices; original combined/international
1.6.1 artifacts do not load in sklearn 1.9.1. Base matrices contain reward levels
and are not the conservative model inputs; fit the raw-input pipeline inside CV.
No final production model has been trained.

Historical raw 01/original 05 training included **1,486 current U.S. reserved
IDs** (combined 1,630; international 84). Current official partitions are disjoint,
but the test is historically exposed. Optimism has not been quantified. No claim
of a pristine holdout or prospective generalization is supported. Independent
external/prospective evaluation is preferable if available. Do not rerandomize
this explored dataset to claim independence. Combined V3 top-12 selection was
before CV and remains exploratory; historical combined outputs are preserved.

Proceed now: agree training-only search/selection rules, keep all processing and
selection within folds, and document the chosen procedure together. Wait for
evaluation-protocol agreement before test scoring/publication; obtain source
provenance or independent data before stronger launch-time/generalization claims.
See [verification and complete file list](workflow_audit_remediation_2026-10-15.md).

## October 27 model-specific tuning handoff

The four model-specific notebooks and preliminary interpretability notebook were
implemented and executed from fresh kernels. They use only the official U.S.
training rows and do not load reserved U.S. IDs or labels. See
[the consolidated training-only comparison](06_training_comparison.md) for the
actual outputs, search budgets, hyperparameters, and limitations.

HGB had the highest observed tuning-CV AUC at **.7255**, compared with its
verified baseline of **.7235**. Logistic moved from **.7199** to **.7201**.
Random Forest moved from **.7117** to **.7209**, but only five trials were
completed because the larger search was too slow. Extra Trees reached **.7205**
in a five-trial preliminary search; its reference baseline in that notebook is
new and should not be treated as an approved historical number.

These are ordinary tuning-CV estimates, not nested estimates. The apparent
ranking is therefore preliminary, and the different RF/Extra Trees budgets make
the comparison less balanced. HGB and Logistic are the clearest candidates for
the next protocol discussion. The interpretability notebook provides fold-held
out permutation importance for those two conservative baselines; it does not
select a final model or make causal claims.

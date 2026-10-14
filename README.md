# Kickstarter Crowdfunding Recommendation Engine — Team 1C

**Break Through Tech AI Studio · Fall 2026 · State Street challenge**

We are developing a machine-learning workflow to predict whether a Kickstarter campaign is likely to succeed using information that could be available at or near launch. We also aim to turn model findings into responsible, understandable guidance for creators. **This project does not use State Street data.** See the [challenge brief](Challenge-Project-Overview.md).

**Current stage:** U.S.-candidate data exploration, conservative feature engineering, and baseline model comparisons are complete. Hyperparameter tuning, final held-out testing, interpretability, and the recommendation prototype are still in progress.

## Current results — U.S. candidate cohort

| Cohort | Campaigns |
|---|---:|
| Cleaned records across all campaign statuses | 45,815 |
| Candidate U.S. records identified by location suffix | 42,175 |
| Successful/failed U.S. modeling candidates | **38,420** |
| Training campaigns | **26,945** |
| Reserved test campaigns | **11,475** |

The U.S. subset is **provisional**. The dataset does not include a verified country column: the filter identifies locations ending with a U.S. state or DC abbreviation. It may omit legitimate U.S. records or include ambiguous matches. See [the scope and split documentation](docs/us_only_workflow.md). The U.S. subset preserves membership in the original global training/test partition.

### Baseline modeling: five-fold CV on training data only

| Model | Mean ROC-AUC | Mean accuracy | Mean F1 |
|---|---:|---:|---:|
| HistGradientBoosting | **0.7235** | **0.6684** | **0.7156** |
| Logistic Regression | 0.7199 | 0.6684 | 0.7111 |
| Random Forest | 0.7117 | 0.6583 | 0.7036 |
| KNN | 0.6990 | 0.6527 | 0.7069 |
| Decision Tree | 0.6867 | 0.6491 | 0.6958 |
| Dummy (majority) | 0.5000 | 0.5461 | 0.7065 |

These are **cross-validation results**, not final test-set results. The majority-class dummy's F1 is relatively high because it predicts the positive class for every row; use the complete set of metrics to compare models.

An additional **training-only feature experiment** found that four optional launch-time candidate features changed Logistic Regression ROC-AUC from **0.7199 to 0.7235**, while HistGradientBoosting changed from **0.7235 to 0.7232**. The differences are small and do not justify a claim of superiority. See [`05a_safe_feature_check.ipynb`](notebooks/us_only/05a_safe_feature_check.ipynb).

## Repository layout and study order

| Location | Purpose |
|---|---|
| `notebooks/00_load_data.ipynb` through `05b_pre_tuning_validation.ipynb` | Original broad-cohort data processing, feature work, baseline modeling, and pre-tuning validation |
| [`notebooks/us_only/01–05`](notebooks/us_only/) | Separate U.S.-candidate audit, cleaning, preprocessing, feature engineering, and baseline modeling |
| [`notebooks/us_only/04b_us_eda.ipynb`](notebooks/us_only/04b_us_eda.ipynb) | Additional U.S.-specific EDA with uncertainty intervals |
| [`notebooks/us_only/05a_safe_feature_check.ipynb`](notebooks/us_only/05a_safe_feature_check.ipynb) | Training-only feature comparison, without using held-out labels |
| `utils/` | Reusable feature engineering, filtering, Google Drive, and preprocessing helpers |
| `tests/` | Automated scope, feature, and target-encoding regression tests |
| [`docs/`](docs/) | Feature-policy, data-scope, and methodological decisions |

The **global notebooks remain separate and unchanged by the U.S.-only workflow**. Do not compare the global and U.S.-candidate results as if their feature sets or populations were identical.

## Reproduce the workflow

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -v
```

To run notebooks, install Jupyter if needed (`python3 -m pip install notebook`) and start it with `python3 -m notebook`.

**Data is not committed to GitHub.** Retrieve the Kaggle Kickstarter dataset or the team's shared processed files according to [`data/README.md`](data/README.md). The U.S.-only notebooks require the cleaned broad-cohort CSV and saved global train/test project-ID files. Follow [`docs/us_only_workflow.md`](docs/us_only_workflow.md) for prerequisites and run order; then run optional `04b` and `05a` as explained in [`docs/us_feature_extension.md`](docs/us_feature_extension.md).

Our existing [GitHub Actions workflow](.github/workflows/python-app.yml) runs Python unit tests and critical lint checks on relevant pushes and pull requests. It **does not** execute notebooks or train models; those require the separately managed dataset.

## Methodological safeguards and limitations

- Candidate predictions are intended for **campaign launch**. The conservative U.S. baseline excludes unverified `funded date`-derived timing, title flags, `renamed`, and scraped reward-level features.
- Learned feature statistics and categorical encodings are fitted inside training folds during cross-validation. The original target encoder was separately reviewed and tested for training/inference index collisions; see [`docs/model_validation.md`](docs/model_validation.md).
- The geographic proxy and historical availability of some fields still need source verification or Challenge Advisor review.
- Feature associations are **not causal effects**. Recommendations must not imply that changing a funding goal, category, or reward automatically changes a campaign's outcome.
- Final tuning and model selection must use training data only; the reserved U.S. test set is for final evaluation after selecting a configuration.

## Next steps

1. Confirm the U.S. scope and prediction-time feature policy with the team and Challenge Advisor.
2. Tune HistGradientBoosting and Logistic Regression (optionally Random Forest) and select the final model using training-only validation in `06`.
3. Evaluate the selected model once on the reserved test set and document all required classification metrics.
4. Complete interpretation/error analysis in `07`, then implement the recommendation prototype.

## Team

| Contributor | GitHub | Work to date |
|---|---|---|
| Anna Sehgal | [@annasehgal](https://github.com/annasehgal) | Data handling, cleaning, preprocessing, validation, U.S. scope workflow, and project documentation |
| Zasha Benites | [@zm-michelle](https://github.com/zm-michelle) | Feature engineering, EDA, visualization, and modeling collaboration |

## Data, license, and acknowledgements

- **Source:** [Kaggle Kickstarter Projects dataset](https://www.kaggle.com/parienza/kickstarter). Original dataset and third-party materials have their own terms; no raw or processed data is distributed with this release.
- **License:** See the repository's existing [custom `LICENSE`](LICENSE). Do not assume this is an MIT-licensed project.
- **Acknowledgements:** Break Through Tech AI Studio, State Street challenge sponsors, Challenge Advisor, and team collaborators.

Release notes for the baseline milestone are in [`docs/releases/v0.1.0.md`](docs/releases/v0.1.0.md).

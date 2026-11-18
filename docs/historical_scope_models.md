# Experimental historical location models

These two additional predictors support the historical-cohort use requested for the dashboard. They do not replace or modify U.S. HGB v1. No U.S. weights, feature definitions, threshold or metrics changed.

International means explicit named non-U.S. **listed locations**, not verified registration markets. Mixed uses the original global training cohort, including U.S., named non-U.S., unknown locations and territories. The cohorts overlap.

## Inputs and fitting

Both new models use duration, category, subcategory and a named listed-region group derived by the existing fixed geography classifier. Mixed maps both U.S. state proxies and explicit U.S. names to United States. Unknown/territory training locations are retained as an unresolved group. Public inputs require recognized U.S. or named non-U.S. locations; international mode accepts only named non-U.S. locations. Use full country names rather than ambiguous country/province abbreviations.

Funding goals and all derived financial features are excluded. No title, rewards, inferred calendar, updates, pledged funds, backers or comments enter these models. This resolves the monetary-input blocker for these limited baselines without inventing currency conversion. See [the source evidence](funding_currency_review.md). The four-input policy is specific to these new models; U.S. v1 still requires its original five inputs.

One fixed LogisticRegression(max_iter=2000) baseline is fitted per scope. This reuses an existing baseline setting without a hyperparameter search, model-family competition, threshold optimization or optional feature experiment. It is not a promotion of an earlier goal-based model's score. Preprocessing is a serialized ColumnTransformer with training-fold duration median imputation/scaling and categorical imputation/one-hot encoding (min_frequency=10, handle_unknown=ignore). Geography normalization is fixed and does not learn statistics.

Training preserves the saved official project-ID order and checks unique IDs, binary labels and cleaned-status agreement. International membership is checked against original global training IDs and the named-location classifier. No official split is recreated. The training files contain 1,427 international and 29,288 mixed rows; mixed comprises 26,945 U.S., 1,427 named non-U.S., 896 unresolved and 20 territory training rows. No reserved ID/label file is opened and no reserved score is computed.

Five-fold stratified training CV uses shuffle=True and random_state=42. Every fitting fold refits the complete preprocessing/model pipeline. Reported metrics pool out-of-fold probabilities at the fixed .5 threshold. After validation, each baseline is fitted on all its official training rows and serialized. Representative probabilities are checked exactly before/after serialization.

## Results and limits

| Scope | Accuracy | ROC-AUC | Failure recall | Always-majority accuracy |
|---|---:|---:|---:|---:|
| International historical | 60.41% | 0.5893 | 30.52% | 60.97% |
| Mixed historical | 62.23% | 0.6631 | 50.41% | 54.74% |

The international baseline has lower accuracy than always predicting successful. The UI explicitly warns against using it for campaign decisions. Both baselines are exploratory demonstrations, not selected production models or validated contemporary forecasts. These results are weaker than the frozen U.S. model, but different populations, inputs and evaluation procedures prevent a head-to-head ranking.

Current campaign plans can be entered, but random folds of old data do not evaluate present-day performance. Actual future-facing validation requires newer, appropriately licensed data, consistent input definitions, known currency if funding is added, and a time-based external evaluation. Existing reserved sets have historical exposure and remain approval-gated. No probabilities are calibrated here. Listed location, duration semantics and source snapshot timing remain limitations.

Fewer than 25 training examples for a listed region triggers a support warning, including zero for unseen regions. This is a display rule, not a statistical reliability guarantee. Unseen categories/subcategories also receive explicit fitted-fallback warnings. The models measure associations; they do not establish causal planning advice.

## Files and deployment

Artifacts prepared locally for team review, not automatically staged, committed or published:

- `models/international_historical_v1.joblib` and sibling `.metadata.json`
- `models/mixed_historical_v1.joblib` and sibling `.metadata.json`

The narrow ignore exceptions make these exact files reviewable without exposing other local artifacts. Confirm model/data publication terms before adding them to a public commit. No training rows or raw datasets are bundled. Metadata contains aggregate support, class metrics, input schema, environment and a training-ID digest.

Expected SHA-256:

```text
international: d6cbe690005c6bb8afff8bb99daf25d8962de9813ea27011bcfd9554bd9106b7
mixed:         f46014783e219c7a5f98175ce9d355e0c67a7b738dac872626459a4a5347349d
```

`utils/historical_prediction.py` pins these hashes, checks sibling metadata, runtime, class order and pipeline schema, and uses class-1 predict_proba at .5. Loading is cached separately by scope/hash in Streamlit. Missing or corrupt artifacts produce an error with no probability. The application never imports the training script or trains during reruns.

The new artifacts are about 10 KB each and use the existing pinned Python 3.12 runtime. No additional runtime packages or credentials are required. Streamlit needs both binary and metadata for each new mode in the deployed checkout or approved trusted storage. A code-only push will not supply uncommitted local artifacts.

Reproduction command, for a separately reviewed new training run:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1 .venv/bin/python scripts/prepare_historical_models.py
```

The script refuses to overwrite any existing historical artifact, verifies the U.S. frozen hash before and after, and only writes the two new historical model paths. Do not remove or replace frozen artifacts just to rerun it. Reproduction changes require a new reviewed artifact version/hash, never training on application startup.

For Music / Rock, Berlin, Germany, duration 30, direct probabilities are 0.7530055715237318 (international) and 0.7558369258217466 (mixed). UI displays 75.30% and 75.58%. Funding amount changes cannot affect either model because those inputs are absent. This is an inference-consistency example, not an accuracy evaluation.

## Verification and review

The current workspace passes all 92 tests. A checkout-shaped copy containing the proposed source/artifact files and no processed datasets runs 92 tests: 91 pass and the private-data integration test skips as expected. Critical lint and git diff --check pass. Real Streamlit AppTest submissions agree with direct calls for both new models; existing U.S. prediction checks still pass. No browser-based visual QA or public deployment was performed for these new modes.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1 /private/tmp/team1c-deploy-venv/bin/python -m unittest discover -s tests -v
/private/tmp/team1c-deploy-venv/bin/python -m flake8 app.py utils tests scripts/prepare_historical_models.py --extend-exclude=.venv --count --select=E9,F63,F7,F82 --show-source --statistics
git diff --check
```

Tests cover official training-ID ordering with synthetic data and no reserved-file access, serialization, exact direct/UI probabilities, class mapping, funding/outcome input exclusion, location/scope and numeric validation, unknown-region/category warnings, missing/corrupt artifact handling, switching modes without premature loading, and no training during inference.

Changed/new files in this addition: `.gitignore`, `README.md`, `app.py`, `docs/dashboard_model_interface.md`, `docs/funding_currency_review.md`, this document, `models/international_historical_v1.joblib`, `models/international_historical_v1.metadata.json`, `models/mixed_historical_v1.joblib`, `models/mixed_historical_v1.metadata.json`, `scripts/prepare_historical_models.py`, `utils/historical_prediction.py`, `tests/test_historical_prediction.py`, and `tests/test_streamlit_integration.py`. All files remain unstaged. No commit, push, release, artifact upload or deployment was performed.

# Kickstarter Crowdfunding Recommendation Engine: Team 1C

Break Through Tech AI Studio, Fall 2026, State Street challenge. This project does not use State Street data. See [the challenge brief](Challenge-Project-Overview.md) and the existing [custom license](LICENSE).

The current demonstration uses the frozen U.S. HistGradientBoosting v1 pipeline with threshold 0.5. Its inputs are goal, duration, category, subcategory and listed location. No additional tuning is needed for the dashboard.

| Development validation | Result |
|---|---:|
| Accuracy | 66.93% |
| ROC-AUC | 0.7239 |
| Failure recall | 55.72% |

These are training-only restricted nested comparison results, informed by prior tuning, on 26,945 official U.S. training campaigns. They are not independent test results. The 11,475 reserved campaigns include 1,486 IDs exposed during historical development. No new reserved evaluation was performed. Probabilities are uncalibrated; source launch snapshots, duration meaning and goal currency remain unverified. Listed state/DC is a geographic proxy. See [final selection](docs/final_model_selection.md) for complete provenance and preserved historical results.

The city-frequency HGB v2 candidate (67.70% development accuracy) remains experimental. International and mixed models are unavailable in the dashboard because input, currency and separate evaluation requirements are unresolved.

## Run the dashboard

The actual entrypoint is `app.py`. Use Python 3.12 (the model was created with 3.12.3):

```bash
python3.12 -m venv .venv-dashboard
source .venv-dashboard/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The runtime pins match the frozen model: scikit-learn 1.9.1, NumPy 2.5.3, pandas 3.0.5 and joblib 1.6.0, with Streamlit 1.67.0. Research notebooks use `requirements-reviewed.txt`; optional algorithm comparisons use `requirements-model-comparison.txt` when available in your checkout. The public app needs no Drive access, API keys, datasets or LLM.

Supply both `models/us_primary_model_v1.joblib` and its sibling `us_primary_model_v1.metadata.json` from the reviewed team artifact source. The loader checks the model hash, metadata identity, threshold, input order, runtime, class order and pipeline structure before serving results. Missing or incompatible artifacts make prediction unavailable; startup never retrains.

Frozen SHA-256:

```text
243bb0d53ac7b456e6b636c424bc8d0d74dbd67f014d5f8f175b7d4226bd94b6
```

Enter positive finite goals/durations, exact category names such as Music / Rock, and `TX` or `Austin, TX`. Fractional durations remain supported. Unseen category/subcategory strings use fitted training fallbacks; pairs are not validated. The app shows only the model estimate and its limitations.

## Deployment and recommendations

See [the integration and deployment handoff](docs/streamlit_integration_handoff.md) before deploying to Streamlit Community Cloud. Choose the reviewed branch, `app.py`, and Python 3.12. Root `requirements.txt` supplies the runtime. Artifact permissions and a clean host build must be verified first.

Important: PR #56 already added both joblib models to public Git history. Ignore rules do not untrack them. This review does not publish or remove either model. Confirm publication permission with the team. A versioned, approved GitHub release asset with the exact hash is the preferred future distribution method; protected storage is appropriate if redistribution is restricted. No download or startup training is implemented.

Historical category benchmarks, similar-campaign retrieval, and goal/duration scenarios exist in local research work, not in this dashboard. Their outputs must remain labeled historical comparisons or model responses to substituted inputs. They do not establish causal improvements. Local LLM summaries remain a plan; an LLM must never invent probabilities, statistics or model explanations.

## Repository and verification

`utils/us_prediction.py` is the saved-model interface. `utils/preprocessing.py` contains the serialized feature and preprocessing classes. `utils/us_modeling.py` reconstructs official training membership. `models/train_and_predict.py` is Zasha's separate experimental CatBoost workflow, not the selected HGB interface. Do not import its training script into Streamlit. Its limitations and a consolidation proposal are in the handoff.

The global and international notebooks remain historical/secondary work. Data is not required for dashboard prediction. See [data instructions](data/README.md), [U.S. workflow](docs/us_only_workflow.md), [environment guidance](docs/environment_reproducibility.md), and [reserved evaluation protocol](docs/us_reserved_test_protocol.md). Do not recreate splits or refit frozen artifacts while preparing deployment.

```bash
python -m pip install flake8
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v
python -m flake8 app.py utils tests models --extend-exclude=.venv --count --select=E9,F63,F7,F82 --show-source --statistics
```

Tests use synthetic inputs where possible. Real artifact/UI checks explicitly skip if their dependencies are absent. The processed-data integration test skips absent private data. CI runs tests and critical lint without training notebooks.

## Team

| Contributor | GitHub | Work to date |
|---|---|---|
| Anna Sehgal | [@annasehgal](https://github.com/annasehgal) | Data handling, cleaning, preprocessing, validation, U.S. scope workflow, and project documentation |
| Zasha Benites | [@zm-michelle](https://github.com/zm-michelle) | Feature engineering, EDA, visualization, modeling collaboration, online training and prediction workflow |
| Nitya Vobugari | [@nityavobugari18](https://github.com/nityavobugari18) | Initial Streamlit campaign advisor interface and saved U.S. model integration |

Source: [Kaggle Kickstarter Projects](https://www.kaggle.com/parienza/kickstarter). Dataset and third-party materials have their own terms. Acknowledgements: Break Through Tech AI Studio, State Street challenge sponsors, Challenge Advisor, and team collaborators. [Baseline release notes](docs/releases/v0.1.0.md) preserve the earlier milestone.

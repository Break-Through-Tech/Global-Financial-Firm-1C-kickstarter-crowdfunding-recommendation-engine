# Reproduce additional algorithm comparisons

Run from the repository root on `feat/launch-feature-experiments`. Keep the existing reviewed `.venv`, production requirements, official data splits and v1/v2 artifacts unchanged.

## Environment

The separate `.venv-model-comparison` installs only the additional packages and reads the reviewed `.venv` site-packages through `reviewed_base.pth`. This is an overlay environment, not a self-contained copy. It requires the reviewed base to remain present at the recorded location. It leaves that base untouched. Python 3.12.3, sklearn 1.9.1, NumPy 2.5.3, pandas 3.0.5, SciPy 1.18.1 and joblib 1.6.0 were retained. Extra packages: XGBoost 3.4.1, CatBoost 1.2.10, LightGBM 4.7.0, graphviz 0.21 and plotly 7.1.0. Full observed versions are in `additional_model_results/environment.txt`; additions are pinned separately in `requirements-model-comparison.txt`.

To recreate the overlay after the reviewed `.venv` is available:

```bash
.venv/bin/python -m venv .venv-model-comparison
.venv-model-comparison/bin/python - <<'PY'
from pathlib import Path
import site
base = Path('.venv/lib/python3.12/site-packages').resolve()
assert base.is_dir()
Path(site.getsitepackages()[0], 'reviewed_base.pth').write_text(str(base) + '\n')
PY
.venv-model-comparison/bin/python -m pip install -r requirements-model-comparison.txt
.venv-model-comparison/bin/python -m pip check
```

Installation/import/fit smoke tests and pip check passed on macOS arm64. Existing `/opt/homebrew/opt/libomp/lib/libomp.dylib` was available; no Homebrew or system-library changes were made. XGBoost and LightGBM document OpenMP requirements on macOS. If missing on another machine, address that prerequisite separately rather than changing production packages. [XGBoost installation](https://xgboost.readthedocs.io/en/stable/install.html), [LightGBM installation](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html)

CatBoost's list-valued `cat_features` is incompatible with sklearn 1.9.1 clone checks because get_params deep-copies it. The comparison uses an immutable tuple of the same category names, which passes cloning. CatBoost receives strings and handles categoricals natively; external one-hot/target encoding is not applied. Numeric NaNs are retained for CatBoost. Ordered boosting, seed 42, CPU and one thread are explicit. No private classes are patched. [CatBoost categorical handling](https://catboost.ai/docs/en/features/categorical-features), [CatBoost source](https://github.com/catboost/catboost/blob/master/catboost/python-package/catboost/core.py)

## Required local files and run order

Required: `data/Processed-Data/us_only/DSI_kickstarterscrape_dataset_cleaned.csv`, its saved `train_project_ids.csv`, and the existing v1/v2 joblib files with sibling metadata. No Drive credentials are needed. No reserved ID/label files are opened by the new comparison or recommendation scripts. The existing loader reconstructs only official training rows for modeling, checks training statuses/labels and verifies the established settled count.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-model-comparison/bin/python -m unittest discover -s tests -v
.venv-model-comparison/bin/python scripts/execute_algorithm_comparison.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/research_us_recommendations.py
.venv/bin/python -m flake8 utils tests scripts --extend-exclude=.venv,.venv-model-comparison --count --select=E9,F63,F7,F82 --show-source --statistics
```

The executor runs only `notebooks/06_us_algorithm_comparison.ipynb` in a fresh kernel using the invoking interpreter, serial numerical threads and repository-root working directory. Local Jupyter sockets must be allowed. Outputs are saved atomically only after successful execution; existing notebooks are not run. The measured comparison took 486.5 seconds on this machine, excluding environment setup; runtime varies. No model is automatically saved.

The previous extended-feature notebook does not need rerunning. Do not run `06_us_validation.ipynb`, old global workflows, construction notebooks or the reserved-evaluation notebook as prerequisites. They can overwrite artifacts or use obsolete procedures. See the [earlier workflow map](launch_experiments_reproducibility.md).

Generated files in `docs/additional_model_results/`: summary metrics, outer-fold metrics/settings/thresholds, inner option scores, paired differences, plot, environment/manifest, runtime and tests. `recommendation_example.txt` is a deterministic descriptive training-row example with self exclusion, benchmark counts, nearest neighbors, a v1 scenario and limitations. It is not a classifier feature source, validated retrieval metric or LLM output.

HGB v1 stays primary, city v2 stays experimental, and no new classifier passed the minimum guardrails relative to its matching HGB. Reward-only/ladder experiments remain exploratory. Global deployment remains unavailable while goal units and coverage are unresolved. Coordinate any application work with Zasha; no Streamlit changes are included.

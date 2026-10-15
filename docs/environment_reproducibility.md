# Reviewed environment and preprocessing artifacts

October 15, 2026: CPython **3.12.3**, NumPy **2.5.3**, pandas **3.0.5**,
SciPy **1.18.1**, scikit-learn **1.9.1**, matplotlib **3.11.2**, joblib **1.6.0**,
nbclient **0.11.0**, nbformat **5.11.1**, ipykernel **7.4.0**.
`requirements.txt` remains the original broad application dependency list.
`requirements-reviewed.txt` captures the complete installed environment, including
notebook dependencies, to constrain future installs rather than float versions.
No upgrade/downgrade was performed. This is a reviewed macOS environment snapshot,
not a claim that fresh installation or identical scores on every platform was tested.

In a separate environment with Python 3.12.3, install the snapshot using
`python -m pip install -r requirements-reviewed.txt`, then run the tests. Retain
this numerical stack for artifact serialization and notebook comparisons; moving
versions requires rechecking compatibility and scores. Do not monkeypatch private
scikit-learn classes or blindly downgrade to load an old pickle.

## Preserved originals and fresh compatible base artifacts

The original combined and international `base_preprocessor.joblib` objects
identify scikit-learn 1.6.1 and fail under 1.9.1 with `_RemainderColsList` missing.
The original U.S. artifact loads and reproduces its stored training transform.
All originals remain unchanged. Their compatible replacements are local and ignored:

```text
data/Processed-Data/compat/python-3.12.3_sklearn-1.9.1/base_preprocessor.joblib
data/Processed-Data/us_only/compat/python-3.12.3_sklearn-1.9.1/base_preprocessor.joblib
data/Processed-Data/international_only/compat/python-3.12.3_sklearn-1.9.1/base_preprocessor.joblib
```

Each has a sibling `manifest.json` with environment, training-input hashes,
artifact hash, generator hash, original load diagnostics and smoke-test result.
Generation fits only official training rows, matches the original 03 definitions,
checks saved matrix values and ordered column names, then loads and transforms
the fresh artifact. No reserved labels or split writes are involved.

```bash
.venv/bin/python docs/audit_evidence/compatible_artifacts.py
```

The generator reuses an existing versioned artifact after verifying its transform;
it does not replace originals or overwrite existing manifests. Use a new namespace
for a changed recipe. Only load trusted local/team artifacts.

These are **base preprocessing compatibility artifacts**, including reward `levels`,
not conservative production models. For 06 use raw official training rows and
`make_pipeline(..., include_timing=False, include_flags=False, include_reward=False,
exclude_renamed=True)`. Refitting the full preprocessing/estimator pipeline inside
CV remains required. A matrix/serialization match does not reproduce historical
model scores or validate source semantics.

Historical U.S./international modeling notebook snapshots and the old international
training-CV CSV are preserved locally under
`data/Processed-Data/remediation_2026-10-15/historical_outputs/`. They are not GitHub
artifacts. The six refreshed notebooks retain their model settings; score differences
are recorded in the remediation report. Combined historical notebooks are unchanged.

## Repeat verification safely

Run from repository root, activating `.venv` or using its interpreter explicitly:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python docs/audit_evidence/reconcile.py
MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python docs/audit_evidence/eda_and_base_matrices.py
MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python docs/audit_evidence/training_cv.py
MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python docs/audit_evidence/combined_cv.py
MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python docs/audit_evidence/execute_active_notebooks.py
```

The read-only audit CV runners execute current code in fresh namespaces, not full
Jupyter notebooks. The final command runs the six active notebooks in separate
kernels using the invoking interpreter and repository-root working directory. It
saves each notebook atomically only after full success, unchanged sources and no
cell errors. It backs up the international CV CSV, the sole notebook data write,
and prints a temporary evidence directory. A sandbox may require permission for
local Jupyter kernel sockets; the first restricted attempt failed without saving
any notebooks, then the authorized kernel runs succeeded.

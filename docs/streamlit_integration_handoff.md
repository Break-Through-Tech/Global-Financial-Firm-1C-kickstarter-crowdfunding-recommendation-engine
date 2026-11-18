# Streamlit integration handoff

Reviewed November 17, 2026. Branch `fix/streamlit-model-integration` starts at verified public main `d9caa1360fc7e8283c8201381fea8a60309b2d5f`. SSH fetch failed authentication; public HTTPS fetch confirmed the same revision as `origin/main`. Starting branch was `feat/launch-feature-experiments` at `b435a62`. Existing untracked research files were preserved and are outside this change. Nothing was committed, pushed, merged, published or deployed.

## Findings and integration changes

Nitya's initial interface (`678660d`) captured inputs without prediction. Her integration (`d9caa13`) correctly caches `USPredictor`, selects `models/us_primary_model_v1.joblib`, passes only the five approved inputs, and displays `predict_proba` column 1. The predictor verifies class order [0 failed, 1 successful], applies the saved .5 threshold and transforms without refitting. It never imports the online trainer or historical raw model. There were no dropdowns to validate: category/subcategory are text inputs. Unknown values use the fitted prior and goal-reference fallback; category/subcategory pairs are not checked. Case matters for category matching.

Necessary changes preserve Nitya's page and prediction flow: verify frozen hash/metadata/runtime/schema on first cached load; key the cache by frozen hash; remove required reward/date controls unused by v1; make campaign name optional; preserve fractional duration display; replace inactive navigation labels with explicit mode status; explain fallback and validation limitations; suppress internal exception details in unexpected failures. Input ValueErrors remain visible and contain validation guidance. Missing model/metadata produces an unavailable message and no probability. The app still uses positive numeric widget bounds plus the shared validator, without imposing new campaign-duration limits.

The public loader is a small wrapper around the unchanged generic `USPredictor` contract. It validates bytes before joblib deserialization and does not change features, weights, reported metrics or threshold. Model files are unchanged. The existing v1 metadata is made reviewable through a narrow ignore exception; its content is unchanged. v2 metadata stays ignored. The serialized feature builder, imputer, scaler, encoding maps and HGB are all included in v1.

Local v1 is 309,170 bytes and SHA-256 is exactly:
`243bb0d53ac7b456e6b636c424bc8d0d74dbd67f014d5f8f175b7d4226bd94b6`.

Its metadata records 26,945 official U.S. training rows, five approved inputs, eight transformed features, HGB, threshold .5 and the recorded runtime. Development accuracy 66.93%, AUC .7239 and failure recall 55.72% remain unchanged. v2's 67.70% development accuracy is experimental and is never loaded.

## Online pipeline audit

Reviewed actual PR #55 merge (`5f341e8`), issue #52 implementation (`f785dc8`), November 10 Drive-cleaning change (`aae841a`), and merge-resolution history. No remote trainer was executed and no Drive credentials were used.

| Question | Actual implementation |
|---|---|
| Data source | All scopes default to the same raw CSV from configured Drive folder/name, with `clean=True`; supplied `df` bypasses loading. This is not the official processed training loader. |
| Cleaning | HTML-unescapes three fields, keeps first duplicate project ID, infers launch from funded date minus duration, keeps settled status and positive goal/duration with non-null funded percentage, creates `success`. It is a separate cleaning recipe. |
| Scope | `us`: state/DC suffix proxy. `int`: every nonblank location outside that proxy, including unresolved places and territories, unlike the reviewed named-international cohort. `mixed`: all rows, including unknown locations. |
| Splits | No split, random or official, is created in this implementation. By default it fits all scoped cleaned rows, potentially including reserved IDs. Optional positional `train_mask` is caller-supplied, with no ID, provenance or disjointness checks. |
| Algorithm/selection | Always CatBoost, 600 iterations, depth 6, learning rate .05, seed 7. The `model` argument is unused. No CV, evaluation, best-model comparison or model selection is implemented despite the commit description. Imported metrics/other estimators are unused. Historical broad-cohort scores cannot establish separately evaluated deployment models. |
| Preprocessing safety | Imputation, scaling and one-hot encoding live inside the pipeline and would refit within folds if a caller used CV. There is no target encoder here and no CV in the script. Fold containment alone does not fix its default population leakage. |
| Inputs/features | Eight raw inputs: name, category, subcategory, location, goal, duration, reward levels, launch. Uses title, reward amounts and inferred calendar features with unresolved launch availability. It computes state/is-US but drops them from the model matrix. Incompatible with the five-input conservative HGB contract. |
| Training/loading | `train_online_model` trains a new model. `save_model` pickles the entire pipeline including preprocessing; `load_model` loads an existing trusted pickle with only a shallow type check. `predict_campaign` itself only calls predict_proba, assumes column 1, and has no class-order or input-domain check. |
| Import/startup | `from modeling import load, parse_tiers` fails because no `models/modeling.py` or importable root modeling module exists. `train_and_save.py` also trains at module import if imports are repaired. Its CLI example trains three scopes. None is called by Streamlit. |
| International/mixed | No separately evaluated artifacts, integrity metadata, approved currency normalization, robust scope validation or five-input compatibility demonstrated. Keep unavailable. |

Recommendation: keep Zasha's code as exploratory provenance, and route the public dashboard exclusively through `USPredictor` plus frozen identity checks. Coordinate a later small consolidation with Zasha: use one inference interface per approved artifact, reuse the official membership loader for any future offline training, guard executable training under `__main__`, and remove the obsolete modeling import only when its parser source is agreed. Do not rewrite or promote the experimental trainer during this integration task.

## Recommendations review

No benchmarks, retrieval, scenarios or LLM integration are rendered by Streamlit. There are no fabricated recommendation values or decorative action buttons. Existing untracked `utils/us_recommendation_research.py` implements category/subcategory training quantiles, robust-scaled log-goal/duration nearest neighbors with self-ID exclusion, and frozen-predictor goal/duration scenarios. Research documentation/example files identify official training provenance. The functions accept caller-supplied training rows and do not enforce membership themselves; any future UI must supply the official training loader and verify provenance. They are not ready-made public endpoints. Scenario differences describe model responses, not causal success improvements. Local LLM explanation remains a plan and must use verified structured facts without inventing numbers. No LLM was added.

## Deployment procedure and blockers

Simplest hosting option: Streamlit Community Cloud, public app, repository branch reviewed by the team, entrypoint `app.py`, Python 3.12 selected in advanced settings. Root requirements pins the actual inference dependencies. No environment variables or secrets are required for the current local-artifact app. It does not import Drive, request credentials, train, or download data. Preserve repository `utils` module paths for pickle compatibility.

Streamlit 1.50.0 available locally declares pandas<3 and is unsuitable as the reproducible deployment pin. Streamlit 1.67.0 declares pandas<4 and supports the frozen pandas 3.0.5 runtime. Do not resolve installation errors by downgrading model dependencies or retraining. Python 3.12 patch releases are allowed; different Python minor or recorded model-library versions fail closed. A Linux host build and real public-host request remain necessary before claiming hosted readiness.

Artifact discrepancy: PR #56 (`7254d7b`, changes from `b435a62`) added **both v1 and v2 joblibs** and accuracy result files to public Git. They are already tracked despite the prior assumption that models were ignored. Metadata was ignored and absent from main, so a fresh main checkout could not predict. The new narrow metadata exception fixes that prerequisite when reviewed/staged. New `*.joblib` ignore rules prevent future accidental additions but do not remove tracked binaries or history. The team must confirm publication rights, including training-data terms and any learned categorical information, for both already-public artifacts. No deletion or history rewrite was performed.

If the team approves keeping the existing tracked v1 artifact, deployment from a reviewed immutable commit is simplest: checkout supplies the exact bytes and reviewed metadata, and the loader independently enforces the hardcoded digest. No additional model publication is needed. Do not activate v2.

For future distribution outside Git, prefer an approved versioned GitHub release asset containing only v1 plus its reviewed metadata and recorded hash. A deployment/build step should fetch a fixed release/tag asset from the team-owned repository over HTTPS into the exact models path, verify the hardcoded digest before loading, and fail if missing or changed. Never use a user-provided URL, mutable latest asset, or untrusted pickle. Release upload and a downloader are deliberately not implemented or authorized here. Community Cloud requires that approved download setup or an artifact present in checkout; an arbitrary ignored local file will not be uploaded.

If public redistribution is restricted, use protected object storage with read-only server credentials managed in host secrets, or a container host with a securely provisioned artifact. Do not put a signed URL, storage token or Drive OAuth flow in the browser/repository. A private artifact source protects download access but does not resolve permission to offer its predictions publicly. Restore the approved same bytes, never refit on startup.

The artifact is small; one cached pipeline per process avoids repeated loads. Actual Linux startup and host memory limits must still be measured. No GPU, training dataset or local LLM is required.

Hosting references checked November 17: [entrypoint and Python selection](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy), [dependency discovery](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies), [repository layout](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization). These steps are a handoff, not an executed deployment.

## Public repository safety

Scanned tracked files and staged names for common key/token/private-key patterns, credential assignments, sensitive filenames, personal filesystem paths, model/data outputs and oversized files. No staged changes existed and no credential-pattern matches were detected. This bounded scan does not prove that every historical blob or notebook output is free of sensitive content. No secret values were printed. Credential filenames do not appear in the all-refs filename history search.

Local `.env`, `credentials.json`, `token_raw.json`, `token_processed.json`, processed data, reference/1A and reference/1B, virtual environments and caches remain ignored/untracked. Explicit ignores now cover `.env.*` except the example, `.streamlit/secrets.toml`, future joblibs and the documented dashboard venv. Broad existing JSON/data ignore rules remain intact. No tracked raw/processed datasets or reference repositories were found. The small tracked accuracy CSV/JSON are model results; their publication provenance should be reviewed with the models.

Existing notebook outputs contain personal filesystem paths at: `notebooks/00_load_data.ipynb`, `02_clean_data.ipynb`, `05_modeling_v3.ipynb`, `06_us_validation.ipynb`, `international_only/02_clean_data.ipynb`, `international_only/03_preprocess_data.ipynb`, `us_only/02_clean_data.ipynb`, `us_only/03_preprocess_data.ipynb` (paths relative to notebooks after the first). Review/redact outputs in a separate approved cleanup if privacy matters. No notebook output or project history was removed. No tracked file exceeds 1 MB. `Challenge-Project-Overview.md` is unchanged.

If a later audit finds an actual credential in history, revoke/rotate it first, then coordinate removal and any history cleanup with repository owners; do not merely add an ignore rule.

## Verification and review commands

The final validator also rejects comma-qualified locations with an empty city, such as `, CA`; supported bare state codes still work. No approved feature definition changed.

Testing used Python 3.12.3 and a fresh temporary virtual environment, independent of the reviewed notebook environment. The initial temporary environment reused reviewed packages; the final `/private/tmp/team1c-deploy-venv` was installed cleanly from root requirements plus flake8. `pip check` reports no broken requirements. Optional experimental algorithm libraries are absent from this minimal runtime; their existing conditional tests do not exercise those optional algorithms here. The complete suite also passed in the reviewed-package environment before the last empty-city validation check.

Exact final test/lint commands from the repository root:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1 /private/tmp/team1c-deploy-venv/bin/python -m unittest discover -s tests -v
/private/tmp/team1c-deploy-venv/bin/python -m flake8 app.py utils tests models --extend-exclude=.venv --count --select=E9,F63,F7,F82 --show-source --statistics
/private/tmp/team1c-deploy-venv/bin/python -m pip check
git diff --check
```

A separate checkout-shaped export in `/private/tmp/team1c-checkout-smoke` contains only tracked files plus the three proposed new files (metadata, this handoff and the new test). It excludes all existing untracked research and all processed datasets. Run the same unittest command there to reproduce the data-absent test result.

Streamlit AppTest exercises real form submissions, optional blank campaign name, both class outcomes, unknown subcategory, full city/state and state-only locations, invalid/missing location, bounded goal handling, missing model, class-order rejection, metadata absence/mismatch, tampered hash rejection before deserialization, cache reuse and no Pipeline/FeatureBuilder/TargetEncoder fitting during inference. Existing synthetic tests cover invalid numeric goals/durations/categories, serialization, probability equality and threshold behavior. Streamlit 1.67 rejects a zero-valued widget edit and keeps its previous valid value; the shared validator separately rejects zero. No browser-based visual QA was performed; AppTest is the end-to-end programmatic UI check.

Actual server smoke command executed:

```bash
/private/tmp/team1c-deploy-venv/bin/python /private/tmp/team1c_server_smoke.py
```

That temporary harness launched `python -m streamlit run app.py --server.headless=true --server.address=127.0.0.1 --server.port=8517 --browser.gatherUsageStats=false`, checked root and `/_stcore/health` (both HTTP 200), and stopped the server. Socket binding required sandbox escalation. This is a local execution check, not public deployment.

| Inputs (duration 30) | Direct success probability | UI | Outcome |
|---|---:|---:|---|
| Music / Rock, Austin, TX, goal 5000 | 0.5974844967523899 | 59.75% | Successful (1) |
| Art / Painting, CA, goal 12000 | 0.3170445766767746 | 31.70% | Failed (0) |
| Music / unseen test subcategory, Boston, ma, goal 3000 | 0.6689528147835481 | 66.90% | Successful (1) |

Direct load measured .016 seconds after imports; the inference-only process peaked at 180.7 MiB on this Mac. These are local observations, not whole-server memory guarantees or Linux benchmarks. The frozen artifact hash was checked again and remains unchanged.

Complete task file list:

- `.gitignore`: protect future artifacts/secrets and expose only approved v1 metadata for review.
- `README.md`: current model, limitations, runtime, credits and deployment overview.
- `app.py`: retain Nitya's interface and saved-model flow with the integration fixes above.
- `requirements.txt`: minimal pinned dashboard runtime; research environment remains separate.
- `utils/us_prediction.py`: frozen loader checks and empty-city validation.
- `tests/test_streamlit_integration.py`: new artifact/UI regression checks.
- `models/us_primary_model_v1.metadata.json`: existing local metadata, unchanged bytes, newly eligible for staging.
- `docs/streamlit_integration_handoff.md`: this audit and handoff.

Review and, only when satisfied, commit and push explicitly. These commands intentionally exclude the pre-existing untracked research files:

```bash
git branch --show-current
git status --short
git diff --check
git diff -- .gitignore README.md app.py requirements.txt utils/us_prediction.py
cat tests/test_streamlit_integration.py
cat docs/streamlit_integration_handoff.md
cat models/us_primary_model_v1.metadata.json
git add -- .gitignore README.md app.py requirements.txt utils/us_prediction.py tests/test_streamlit_integration.py models/us_primary_model_v1.metadata.json docs/streamlit_integration_handoff.md
git diff --cached --stat
git diff --cached
git commit -m "Fix frozen US model integration and document Streamlit deployment"
git push -u origin fix/streamlit-model-integration
```

SSH authentication must be repaired before that origin push, or use your team's approved authenticated HTTPS setup. Do not stage all untracked files blindly. No command above has been run to stage, commit or push.

Team actions: confirm permissions for the already-public models and notebook dataset previews; review existing personal paths in notebook outputs; review this branch and v1 metadata; run a clean Linux/Community Cloud build from the approved commit; then authorize deployment through your hosting account. If choosing external artifact storage, approve and provision a versioned trusted source first. Coordinate consolidation with Zasha without promoting her experimental models. Preserve the existing reserved-evaluation approval process.

Final results: **80 tests passed** in the current workspace using the clean runtime. The checkout-shaped export ran **68 tests: 67 passed, 1 expected skip** for absent processed training data. Its Streamlit and real frozen-artifact tests passed. Critical lint reported **0** findings, `pip check` found no broken requirements, and `git diff --check` passed. The index remains empty.

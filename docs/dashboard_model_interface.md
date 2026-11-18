# Model selector proposal for Zasha

Documentation only. No Streamlit source or existing prediction helper is changed.

| Contract | U.S. primary | Global exploratory |
|---|---|---|
| Status | Available with reviewed environment and local artifacts | Unavailable; selector disabled |
| Artifact | `models/us_primary_model_v1.joblib` plus `us_primary_model_v1.metadata.json` | No trained/validated artifact; do not point to U.S. or base preprocessing joblibs |
| Required fields | goal, duration, category, subcategory, location | Not finalized; requires goal with documented common currency basis, planned duration, category/subcategory and explicit location/coverage policy |
| Coverage | Historical listed state/DC suffix proxy, 26,945 training rows | Would use official global training IDs, including U.S.; not a separate international-only cohort |
| Output | Existing `success_probability`, `predicted_class` (1 success), `classification`, `threshold` | None until independently trained and validated |

Current usage: instantiate `USPredictor(path)` once and reuse `.predict(campaign)`. Cache model loading in the app with an artifact-version/hash key. Import repository `utils` classes before trusted joblib loading, use Python 3.12.3 and the reviewed pinned environment, verify joblib SHA-256 against metadata and expected frozen hash, and verify the metadata version. `USPredictor` itself checks threshold range and class order, but does not enforce environment/hash equality. Report unavailable if either artifact or compatible runtime is missing. Do not silently retrain during app startup.

Goal and duration must be positive finite numbers; bools and numeric strings are rejected. Fractional days are supported. Location accepts any of the 50 state codes or DC, or `City, ST`; surrounding whitespace and terminal state case are normalized. Unsupported/unknown locations and territories are rejected by the current interface. Category and subcategory must be nonempty strings; unseen values use the training prior. The observed support is 13 categories and 49 subcategories, listed in `launch_feature_results/input_review.txt`. A mismatched category/subcategory pair is not currently rejected. Propose dropdowns based on training taxonomy and flag out-of-support values; coordinate enforcement changes first.

Preserve existing output keys. A proposed UI-level envelope can add `model_id`, `model_scope`, `training_cohort`, `validation_provenance` and `warnings`. This is a proposal, not a new helper API. Never route global requests through the U.S. pipeline. A future global artifact needs its own metadata, validator, currency policy, supported-location list and training-only evidence before enabling the option.

Show limitations near predictions: historical 2009–2012 source, unverified launch snapshots/duration/currency, listed location proxy, unseen-category fallback, historical holdout exposure and development-only validation. Probability is a model estimate, not a guarantee or causal recommendation. Reward/title experiment gains must not appear as deployed-model performance. Historical comparison charts must identify team, cohort, split, prediction point and allowed inputs; U.S. and global cohorts overlap.

Team decisions: resolve source currency and launch snapshots, agree global coverage, decide artifact distribution, and coordinate any changes to inputs/output envelope with Zasha. Existing held-out evaluation approval requirements remain in force.

A separate evaluated candidate now exists locally at `models/us_candidate_v2.joblib` with sibling metadata. It adds fold-fitted listed-city frequency and passes the development guardrails, but remains experimental. Its required location is a full listed city/state string; state-only input cannot recover its new feature. Do not enable it through the existing U.S. option until Zasha agrees the full-city validation, normalization/fallback and model selector contract. It is not a global model.

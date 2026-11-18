# Dashboard model interface

| Mode | Artifact | User inputs | Output status |
|---|---|---|---|
| U.S. prediction | `models/us_primary_model_v1.joblib` plus sibling metadata | goal, duration, category, subcategory, location | Frozen primary HGB v1, threshold .5 |
| International (experimental) | `models/international_historical_v1.joblib` plus sibling metadata | duration, category, subcategory, named non-U.S. location | Experimental historical listed-location estimate |
| Mixed (experimental) | `models/mixed_historical_v1.joblib` plus sibling metadata | duration, category, subcategory, recognized U.S. or named non-U.S. location | Experimental historical combined-cohort estimate |

Each mode has its own cached loader, pinned binary SHA-256 and metadata identity. No mode trains during inference. No international request is routed through U.S. HGB. Class order is [failed=0, successful=1]; class-1 predict_proba supplies success_probability, with predicted_class/classification from the fixed .5 threshold. Generic U.S. USPredictor remains unchanged; the app uses load_primary_predictor for its frozen checks.

The two new models deliberately exclude funding amounts because cross-market units remain unverified. They use fixed Logistic Regression baselines with five-fold training-only validation and complete fold-fitted preprocessing. See [training, metrics, artifacts and limitations](historical_scope_models.md) and [funding-unit evidence](funding_currency_review.md). Existing goal-based international baselines and the online CatBoost research code remain separate historical work; their scores are not assigned to these models.

U.S. inputs keep the established contract: positive finite numeric goal/duration (fractional days allowed); nonempty category/subcategory; state/DC code or nonempty City, ST. Named international locations do not pass its state validator. Bare state codes are normalized before inference. Unseen categorical values use the existing fitted encoding/reference fallbacks. Category/subcategory pairs are not validated.

Experimental inputs require positive finite duration, nonempty category/subcategory, and a location recognized by the existing conservative geography classifier. Use full country/region names, such as Berlin, Germany or Toronto, Canada; Toronto, ON is unresolved and rejected. International excludes U.S. locations, unknowns and territories. Mixed accepts U.S. state codes, state-suffix locations, explicit U.S. names and explicit named non-U.S. locations. Its training population includes unresolved/territory locations, but the public form does not infer those unknown memberships.

Experimental pipelines receive duration, category, subcategory and normalized country_region. U.S. state proxies map to United States for the mixed baseline. Unknown regions/categories use learned encoder fallbacks; fewer than 25 training rows for a region and unseen categories trigger warnings. This support rule does not establish reliability. Metadata input_order describes the serialized normalized frame; raw_input_order describes the form contract.

Show the historical-experimental warning before input and near results. Current plans may be entered, but no mode has verified prospective performance. The international baseline's accuracy is below the majority baseline and must be labeled for demonstration, not campaign decisions. Different-cohort scores are not a head-to-head comparison. No LLM, causal recommendations, live currency conversion or new U.S. model is activated. HGB v2 remains experimental and unselected.

All new artifacts require publication review before deployment. Missing binary/metadata or an incompatible runtime must return unavailable, without a fabricated prediction or fallback to another scope. Preserve contributor credit in repository documentation; implementation/contributor history is omitted from the public form.

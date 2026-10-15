# 1C international-only exploration — conservative extension

**Status:** optional cohort; not the primary State Street challenge benchmark.  Parth confirmed the team's plan to report a **U.S.-focused primary evaluation** alongside optional broader/international analysis. This branch of work should not delay the shared `06` U.S. model selection.

## What the geography labels actually mean

The historical Kaggle Kickstarter CSV used by the team contains **`location` text but no verified country-of-registration or country-of-campaign column**. Earlier U.S. notebooks used a transparent `City, ST` state/DC suffix proxy, identifying 42,175 all-status U.S. candidates out of 45,815 globally cleaned records. **The remaining 3,640 were not all international**: 1,318 had missing location, and others were ambiguous, insufficiently specified, or genuinely outside the U.S.

For the international extension, `utils/international_scope.py` assigns five non-overlapping scopes:

| Scope | Sufficient evidence | Example | Eligible for international model? |
|---|---|---|---|
| `us_candidate` | Comma-qualified terminal two-letter U.S. state/DC code | `Chicago, IL` | No |
| `explicit_us` | Explicit United States/USA name or full state name outside proxy | `Portland, USA` | No |
| `us_territory` | Explicit U.S. territory | `San Juan, Puerto Rico` | No |
| **`explicit_non_us`** | Recognized **named** country/region as complete suffix (possibly containing commas) | `Berlin, Germany`, `Toronto, Canada`, `Taipei, Taiwan` | **Yes, provisionally** |
| `unknown` | Missing, ambiguous, unrecognized, or only abbreviated country/province | `Toronto, ON`, `Berlin, BE`, `Tbilisi, Georgia`, `NaN` | **No** |

We intentionally **do not use two-letter ISO country codes**, since `CA`, `IN`, `DE`, `GA` etc. collide with U.S. state abbreviations. We do not guess from city names. Names such as `Georgia` and `Jersey` can refer to more than one geographic unit, so they are held for review. English ISO-3166 country/region names and selected aliases are bundled in `utils/international_scope.py` so the code does not require internet access or a new dependency. This linguistic ISO grouping does not imply political recognition. U.S. territories are excluded, not reclassified as foreign.

**Critical limitations of this cohort:**

- **Location is not nationality or official country.** A creator may list a city or location that differs from their registered country or project origin. A listing may have been edited after campaign launch.
- **Selection bias:** requiring an explicit country name excludes some genuine international records and could favor countries with a particular naming convention. International success estimates **do not generalize to all non-U.S. campaigns**.
- **Unknown membership stays unknown:** no imputation of country, no inferred membership based solely on failing the U.S. filter, and no use of questionable lat/long/geocoding without advisor approval.
- **Currency comparability is unverified.** A `$1,000`-like numeric `goal` may not mean the same thing across markets if its original currency/normalization is uncertain. We report this limitation and use **within-country reference medians** for exploratory relative-goal features rather than pooled global category medians. Even within-country goal references are not proof of currency equivalence.
- **Small samples and sparsity:** rare country/category combinations may overfit; minimum support and training-fold fallbacks are explicit. A smaller country-specific test sample can have wide uncertainty.
- **Historical feature timing:** `funded date` may not represent a deadline; post-launch updates, pledged amount, backers, and comments must never enter a launch-time model. Title/reward values may be post-launch snapshots, so the conservative baseline excludes them.
- **Recommendations are not causal.** A country/category/goal association does not show that changing those fields will increase success. Any future Streamlit explanation must distinguish predictive associations from interventions.

## Repository design and run order

The original international extension introduced the following files. The October 15 remediation also fixes shared utilities and refreshes active training outputs:

```text
utils/international_scope.py
utils/international_features.py
notebooks/international_only/
    01_location_audit.ipynb
    02_clean_data.ipynb
    03_preprocess_data.ipynb
    04_feature_engineering.ipynb
    05_modeling.ipynb
tests/test_international_scope.py
tests/test_international_features.py
docs/international_only_workflow.md
```

**Prerequisites:** current shared preprocessing utilities and the **existing official global split** must be present. Activate the team's Python environment and run the tests:

```bash
source .venv/bin/activate
python3 -m unittest discover -s tests -v
python3 -m jupyter notebook
```

For existing local data, run only 04 then 05. The full preparation order below is documentation, not authorization to overwrite official splits. The first requires:

```text
data/Processed-Data/DSI_kickstarterscrape_dataset_cleaned.csv
```

The third additionally requires the existing original global IDs:

```text
data/Processed-Data/train_project_ids.csv
data/Processed-Data/test_project_ids.csv
```

| Notebook | Task and checks | Output |
|---|---|---|
| `01_location_audit` | Count explicit countries and U.S./unknown exclusions, show high-frequency ambiguous strings; **do not analyze outcomes** | Displays only |
| `02_clean_data` | Reuse global cleaning, select explicit named non-U.S. rows, audit uniqueness/schema, save per-ID evidence and unknown frequency review | Local international cleaned CSV + audit manifests |
| `03_preprocess_data` | Restrict original global train/test IDs to settled international records; assert disjointness, verify saved labels; fit a **base** preprocessor on training only | Seven local preprocessing/split artifacts |
| `04_feature_engineering` | Training-only country/category/duration EDA with Wilson intervals and sample sizes; inspect conservative features and missingness | Displays only |
| `05_modeling` | Five-fold training-only CV for dummy and five ML classifiers; refit every learned feature and encoder per CV fold | Training CV results under international data folder |

Outputs are saved under **`data/Processed-Data/international_only/`**, already inside the repository's ignored processed-data folder. Do not commit generated private/shared datasets or model outputs.

### Why reuse the global official split?

The U.S. workflow already restricts the global original train/test project ID lists. If we randomly re-split international records, some original test campaign might leak into an auxiliary international training model, compromising later evaluation of the global/combined experiments. We therefore **intersect rather than re-split**. Validation and model comparison use only `train_project_ids.csv`; the reserved international test labels are saved separately but must not be opened to choose models.

The original global and derived U.S./international membership remains stable. Any proposal to alter those memberships requires separate human review and is out of scope.

### Country feature engineering

`InternationalFeatureBuilder` inherits the shared, conservative `FeatureBuilder`, but **drops its U.S.-oriented `state` feature** and adds a named `country_region` field, not a mislabeled `state`. It also omits the global `goal_vs_cat` and `goal_vs_subcat` columns, because pooling reference goals across countries without established currency equivalence is questionable. Instead it adds:

- `goal_vs_country`: log1p(goal) relative to the **training-fold country median**; fallback to global training-fold median for unseen or rare countries.
- `goal_vs_country_cat`: log1p(goal) relative to a **training-fold country-and-category median** when at least ten examples support it, otherwise the country reference.

`log_goal`, `goal_per_day`, and `duration` remain for comparison with the conservative U.S. candidate baseline, though their financial meaning across countries still depends on source semantics. When fitting the pipeline with CV, these country medians, imputers, scalers, and categorical encoder are all learned **only from the current fitting fold**. They do not use outcomes or validation/test rows. `country_region` may represent country/territory ISO names, **not registered campaign nationality**.

### Why a conservative model first?

Run the current international conservative policy (`include_timing=False`, `include_flags=False`, `include_reward=False`, `exclude_renamed=True`) before attempting any expanded features. This preserves the project's **launch-time prediction** objective. If source metadata later proves that `funded date` is a pre-specified deadline and scraped titles/rewards reflect launch-time snapshots, treat expanded features as a **separately justified experiment** using training CV. Do not enable features merely because they raise accuracy.

`05_modeling.ipynb` uses one-hot encoding for all models as a consistent, conservative small-sample baseline. Category/country cross-fitting and target encoding remain available through `make_international_pipeline(..., categorical_strategy="target")` for controlled *training-only* experiments if Zasha agrees, but are not required for initial international analysis.

## Handoff and evaluation rules

1. **Primary priority:** U.S.-only `06` training-only tuning and selection. Held-out scoring must wait for agreement on the historically exposed evaluation protocol.
2. **Optional international:** review labeled cohort size, unknown-location exclusions, country imbalances, currency semantics, model CV uncertainty; only then consider a separate `06` evaluation.
3. **Fair cross-cohort science:** a U.S.-trained model and global-trained model can be compared **on the same original U.S. test IDs**, if training rules/feature policy are compatible. International-only model metrics on its **different** test population are descriptive, not a head-to-head U.S. competition.
4. **Do not select models based on held-out test outcomes**, and do not leak future information through dates or historical encodings.
5. Review and document decisions with Zasha and Parth before reporting definitive claims about international generalization.

## October 15 corrections and evaluation boundary

Corrected reporting partitions all 45,815 rows as 42,175 U.S. state/DC proxy,
1 explicit full-state U.S. location outside proxy, 28 U.S. territories,
2,279 explicit non-U.S., and 1,332 unresolved. The two `Virgin Islands, U.S.`
records are territories; Charlottesville is explicit U.S. evidence. Neither
correction expands modeling membership. Bare `CA` is unknown; `City, CA` matches
the frozen proxy. Georgia/Congo and truncated strings remain unresolved.

The 2,064 settled international IDs, 1,427 training and 637 reserved, remain
unchanged. Training has 125 named regions; 113 have fewer than 25 campaigns;
438 of 467 country/category groups have fewer than 10. Within-country goal
references and pooled fallbacks do not establish comparable currency units.

Historical raw training included 84 current international reserved IDs.
These test data are historically exposed even though current splits are disjoint.
No final scores until the team agrees on the [evaluation protocol](evaluation_decisions.md).
Use 04 then 05 for training-only verification; 01 is read-only scope reporting,
02/03 write cohort/split artifacts and must not be routinely rerun in this patch.
The upload helper omits cohort subdirectories. Use the reviewed environment and
namespaced compatible base artifacts described in [environment guidance](environment_reproducibility.md).

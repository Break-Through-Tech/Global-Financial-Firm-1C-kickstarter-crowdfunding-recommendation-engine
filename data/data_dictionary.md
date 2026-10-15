# Source-field and prediction-time data dictionary (draft)

Dataset described in `Challenge-Project-Overview.md`: Kickstarter `DSI_kickstarterscrape_dataset.csv`, 17 original columns. This file records **current code usage and unresolved metadata**, not a verified original data-provider dictionary.

| Field | Current use | Launch-time decision |
|---|---|---|
| `project id` | Unique identifier; official split membership | ID only, excluded as direct feature. |
| `name` | Current campaign title; engineered title/rename indicators | Unverified historical snapshot. |
| `url` | ID/creator/slug extraction | No direct predictive input; derived creator/slug may encode later changes. |
| `category` | Category label and goal comparisons, target encoding | Assumed launch-available; verify stability. |
| `subcategory` | Subcategory label and goal comparisons, target encoding | Assumed launch-available; verify stability. |
| `location` | Location presence, state parsing and encoding | Do not infer country merely from state suffix. |
| `status` | Successful/failed target | **Never input**. |
| `goal` | Funding target; log and relative goal features | Assumed launch-available; verify source currency handling. |
| `pledged` | Funding received | **Exclude: post-launch.** |
| `funded percentage` | Funding progress | **Exclude: post-launch.** |
| `backers` | Contributions / backer count | **Exclude: post-launch.** |
| `funded date` | Parsed by `add_base_columns` as end timestamp | **Unresolved**: is this deadline or date funding threshold was met? Must verify from original source metadata. |
| `levels` | Count of reward levels in base preprocessing | Excluded from conservative U.S./international 05; retained only in historical base matrices. |
| `reward levels` | Parsed pledge amounts and ladder metrics | Unverified historical snapshot. |
| `duration` | Campaign length; reconstructs launch date from `funded date` | Assumed launch-available if calculated against true deadline. |
| `updates` | Campaign update count | **Exclude: post-launch.** |
| `comments` | Comment count | **Exclude: post-launch.** |

**Scope:** U.S. primary scope is advisor-confirmed; the existing comma-qualified state/DC suffix cohort is a geographic proxy, not registration country. Combined and explicitly identifiable non-U.S. listed locations are secondary experiments. Unknowns and territories remain outside both derivative modeling cohorts. International currency comparability and sparse country/category coverage remain unresolved.

All retained goal, duration, category, subcategory and location values assume availability and meaning around launch; historical scrape snapshots do not verify that assumption. Fractional durations are preserved. `funded date` remains unverified and reconstructed timing is excluded from conservative models. See `docs/evaluation_decisions.md`: current holdouts have historical development exposure; no prospective-generalization claim or final held-out score is authorized yet.

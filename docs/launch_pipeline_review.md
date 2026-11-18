# Launch feature pipeline review

Reviewed November 10, 2026 on `feat/launch-feature-experiments`, based on verified remote main `fd7ca9744a6beee2e18c7b63d62f2ba1aa364571`. The worktree was clean. SSH fetch failed authentication; HTTPS `ls-remote` confirmed main. The prior branch and its local ignore-rule commit remain intact. Nothing was committed, pushed or changed on main.

## Current pipeline

| Stage | Finding |
|---|---|
| Global cleaning | `02_clean_data` normalizes HTML/whitespace/categories, keeps the first duplicate project ID, checks rewards and numeric validity, and saves 45,815 rows. October 15's exact raw-to-clean replay passed. It is construction history, not a notebook to rerun over frozen files. Current raw/clean column headers have no currency or launch-snapshot field. |
| Global preprocessing | `03_preprocess_data` established the official seed-42 70/30 settled split: 29,288 training and 12,553 reserved. Base matrices include `levels`; they are not the conservative modeling inputs. Preserve these artifacts. |
| U.S. cohort | `us_only/02-03` uses a comma-qualified state/DC suffix proxy and ordered intersections of global memberships. Current checks confirm 26,945 training and 11,475 reserved IDs, uniqueness, disjointness, parent ordering and training-label agreement. No reserved target column was read. |
| International cohort | Explicitly named non-U.S. locations only: 1,427 training / 637 reserved. Unknowns and territories are separate. This cohort is selected and sparse. Current membership intersections pass. Global includes U.S.; they are not disjoint populations. |
| Feature engineering | HGB v1 transforms five inputs into `log_goal`, `goal_vs_subcat`, `goal_vs_cat`, `goal_per_day`, `duration`, `te_category`, `te_subcategory`, `te_state`. All learned state is inside each training-fold pipeline. |
| Category references | Medians are medians of log1p(goal), learned per category/subcategory. Unknown groups fall back to the median of subcategory medians, not the row-weighted overall goal median. Preserve this established definition. |
| Target encoding | Smoothed success rates, smoothing 50, five shuffled KFold splits at seed 42 in fit_transform. Each encoding uses the fitting subfold's prior. Inference uses full fitted maps with prior fallback, even for colliding DataFrame indices. Training and inference differ intentionally through cross-fitting. Existing regression tests cover this. |
| Location | Training builder extracts uppercase two-letter suffix; inference strips/uppercases a supported state/DC and constructs a comma-qualified location. Real U.S. training rows satisfy the proxy. Bare state-code input is accepted by the interface but normalized before the builder. This is listed location, not verified registration country. |
| Duration | Fractional days are retained; 14,130 training rows are fractional. Range 1–91.96. Planned versus elapsed/scraped duration remains unverified. Do not impose today's campaign-length limits or silently round. |
| Baselines/tuning | Active U.S. 05 has conservative baselines. Model-specific 06 notebooks contain training-only tuning; ordinary tuning CV is selection evidence. No new hyperparameter search here. |
| Validation/interpretability | `06_us_validation` contains restricted nested family comparisons, primary confusion matrix, Brier/calibration and supported group errors. `07_interpretability` provides training-only interpretation; importance/association is not a causal intervention. Existing results are preserved. |
| Ablation | `05a_safe_feature_check` jointly tested goal/state, round-goal and duration flags. `06_us_feature_audit` and accuracy experiments already informed selection. They are preserved; this work does not repeat those features. Global V3 top-12 selection used full-training supervised ranking before CV and remains exploratory. |
| Final selection | Original tuned HGB at .5 retained: .6693 accuracy, .7239 AUC, .7158 success F1, .5572 failure recall in the prior training-fold comparison. The +.003 accuracy threshold is a minimum guardrail, not proof. |
| Serialization | v1 joblib and metadata are local/ignored. New notebook verifies exact refitted-baseline probabilities against saved v1 and representative prediction-interface calls without changing either artifact. Old global/international base artifacts have known version incompatibility; versioned compatible base artifacts are not deployable global models. |
| Prediction | `utils/us_prediction.py` loads joblib plus sibling metadata; accepts the five approved inputs; uses metadata threshold and checks class order. It does not enforce the metadata hash or dependency versions itself. Deployment must check these externally. |

No new implementation defect was demonstrated that warrants replacing established results. Historical raw EDA/original modeling exposed 1,630 current global reserved IDs, including 1,486 U.S. and 84 named-international. This finding comes from the preserved October 15 audit, not a new reserved-label analysis. Official memberships are currently disjoint. The reserved evaluation remains historically exposed and approval-gated; no score was computed here.

## Reference ideas

References are local ZIP downloads extracted into ignored `reference/1A/` and `reference/1B/`. No reference code is imported or executed. Findings below refer to the downloaded snapshots, not independently reproduced team metrics.

| Idea and reference | Team 1C status | Decision |
|---|---|---|
| 1A feature-selection notebook: goal, duration, category, subcategory, state | Already active, with engineered goal references | Preserve v1 definitions. |
| 1A reward analysis and XGBoost notebooks: `levels` | Available in base preprocessing and historical reward builder; excluded from v1 | Test parsed count separately, exploratory only. |
| 1B `src/features.py`: tier amounts, entry amount/goal, goal/tier | Historical `reward_features` already covers counts, endpoints, median and entry/goal; absent from v1 | Test amount and relative-amount groups separately. No backer or quantity features. |
| 1B title length/words, digits, uppercase, punctuation | Historical flag builder has length/words/allcaps/exclamation; excluded from v1. Ratio and digit presence add definitions | Simple length and style groups; no LLM/text model. Snapshot edits unresolved. |
| 1A city experiment; 1B fold-fitted listed-city frequency | Missing from v1; state already active | Test compact city/state frequency only. It measures dataset frequency, not city size or completed history. |
| 1B relative subcategory goal | Concept already active as difference of logs against fold-fitted medians | No duplicate competing definition without a new hypothesis. |
| 1B rounding and duration flags | Already tested in 05a | Do not repeat. |
| Timing from funded date minus duration | Historical builder supports it; source semantics unresolved | Reject as a verified launch calendar. |
| Updates in 1A final selection and 1B final model | Both teams' final feature sets use actual campaign updates | Reject. Planned updates cannot substitute for labels measured with final actual updates. |
| Pledged, funded percentage, backers, comments | Outcomes/engagement | Reject for pre-launch prediction. |
| Creator history, social/video/text assets | No verified timed history or suitable local inputs | Defer; don't infer from future campaigns or scraped outcomes. |

1A final selection records .814 accuracy with updates versus .703 without, on its own 80/20 cohort (30,803 / 7,701). 1B's model card records .8155 with updates versus .7135 launch-only, on 30,702 / 7,676. `m3_updates_effect.csv` similarly shows about .10 accuracy gains for LR/RF. Their reported differences do not establish superiority over 1C: prediction point, feature validity, cohorts, splits and selection procedures differ. Statements in their notebooks about statistical ties or causation are not adopted here. Team 1B constructs matrices/imputations outside its model CV; we retain our fold-contained preprocessing rather than copy that implementation.

## Source audit and global decision

See `launch_feature_results/input_review.txt` for current counts and ordered-ID digests. Among official U.S. training rows, 28 missing reward strings parse as empty lists; both repository parser definitions agree for every row and parsed counts equal `levels`. Raw and cleaned reward strings match after joining by training ID. This supports parsing consistency, not launch-time validity. Titles have no missing values but 366 contain replacement characters. Normalization and historical edits can affect character counts. Dollar symbols alone do not establish a currency or conversion basis.

No local evidence verifies launch reward configuration or title snapshots. Both remain exploratory regardless of gains. The raw/clean schema supplies no currency conversion metadata, while the global cohort contains U.S., named international and unresolved locations. Pooled goal amounts and relative goal references therefore have unresolved comparability. Do not deploy a global model or label unknown locations as international. A separate conservative global experiment is deferred until monetary semantics and supported-location policy are defensible. This task creates no global model or synthetic global predictions from v1.

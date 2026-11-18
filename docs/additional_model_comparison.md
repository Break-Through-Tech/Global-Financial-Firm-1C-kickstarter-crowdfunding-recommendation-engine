# Additional U.S. model comparison

November 10, 2026. Retain HGB v1 as primary and preserve the separate city-frequency HGB v2. No additional algorithm reaches the minimum +.003 pooled accuracy gain over the HGB with the same feature set. No new candidate was saved. This is restricted training-only development evidence, not an independent or prospective evaluation.

## Comparison

Population: the same 26,945 official U.S. training IDs. Three outer stratified folds, seed 20261027; two inner folds, seed 100 + fold. Every fit learns goal references and preprocessing only from its fitting rows. The official split files are untouched and no reserved ID/label files were opened by the new scripts.

HGB keeps its frozen settings. Each additional algorithm has two settings, selected by inner pooled AUC. A separate inner threshold choice considers .45/.50/.55 with a failure-recall constraint relative to inner HGB. Fixed-.5 results below separate ranking/model changes from threshold changes. HGB/XGBoost/LightGBM share the same engineered inputs and cross-fitted target encoding. CatBoost uses the same numeric definitions with native categorical strings and Ordered boosting, without duplicate external target/one-hot encodings. This compares complete pipelines with different categorical handling. [CatBoost categorical documentation](https://catboost.ai/docs/en/features/categorical-features)

| Feature set | Algorithm | Accuracy | ROC-AUC | Success F1 | Failure F1 | Failure recall | Brier | Fit/predict sec |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| conservative | HGB | 0.6693 | 0.7239 | 0.7158 | 0.6047 | 0.5572 | 0.2096 | 0.374/0.063 |
| conservative | XGBoost | 0.6711 | 0.7244 | 0.7182 | 0.6052 | 0.5554 | 0.2094 | 0.253/0.037 |
| conservative | CatBoost | 0.6713 | 0.7240 | 0.7212 | 0.5996 | 0.5423 | 0.2107 | 7.610/0.033 |
| conservative | LightGBM | 0.6705 | 0.7235 | 0.7167 | 0.6062 | 0.5588 | 0.2098 | 0.363/0.122 |
| city | HGB | 0.6770 | 0.7323 | 0.7212 | 0.6163 | 0.5716 | 0.2068 | 0.402/0.066 |
| city | XGBoost | 0.6767 | 0.7330 | 0.7215 | 0.6146 | 0.5681 | 0.2065 | 0.277/0.044 |
| city | CatBoost | 0.6770 | 0.7328 | 0.7237 | 0.6112 | 0.5595 | 0.2075 | 7.947/0.022 |
| city | LightGBM | 0.6793 | 0.7326 | 0.7226 | 0.6201 | 0.5767 | 0.2067 | 0.390/0.128 |
| reward_count_exploratory | HGB | 0.7015 | 0.7597 | 0.7440 | 0.6423 | 0.5904 | 0.1959 | 0.745/0.237 |
| reward_count_exploratory | XGBoost | 0.6998 | 0.7605 | 0.7433 | 0.6386 | 0.5844 | 0.1957 | 0.572/0.171 |
| reward_count_exploratory | CatBoost | 0.6996 | 0.7605 | 0.7441 | 0.6364 | 0.5792 | 0.1964 | 10.163/0.183 |
| reward_count_exploratory | LightGBM | 0.7002 | 0.7596 | 0.7433 | 0.6395 | 0.5861 | 0.1960 | 0.678/0.273 |
| reward_ladder_exploratory | HGB | 0.7117 | 0.7706 | 0.7525 | 0.6548 | 0.6025 | 0.1916 | 0.836/0.240 |
| reward_ladder_exploratory | XGBoost | 0.7102 | 0.7706 | 0.7527 | 0.6500 | 0.5929 | 0.1917 | 0.654/0.176 |
| reward_ladder_exploratory | CatBoost | 0.7100 | 0.7716 | 0.7537 | 0.6473 | 0.5863 | 0.1921 | 12.053/0.178 |
| reward_ladder_exploratory | LightGBM | 0.7109 | 0.7711 | 0.7510 | 0.6553 | 0.6055 | 0.1914 | 0.788/0.299 |

Fit/predict times are means per outer fit/validation partition. Total comparison time including inner fitting was 486.5 seconds. CatBoost took roughly 7.6–12.1 seconds per outer fit versus HGB .37–.84 seconds here. Full success/failure precision/recall and confusion counts for both threshold policies are in [summary.csv](additional_model_results/summary.csv).

Conservative XGBoost gains .00178 accuracy; CatBoost gains .00193 at .5 but loses .01488 failure recall, beyond the .01 tolerance. Conservative LightGBM gains .00115. With city frequency, LightGBM gains .00230, below .003, despite improving accuracy and failure recall in all folds. XGBoost and CatBoost city versions do not improve pooled accuracy over matching HGB. No external algorithm improves reward-set pooled accuracy over matching HGB. This modest grid does not prove other settings cannot help.

## Paired validation and thresholds

Paired fixed-.5 differences versus corresponding HGB:

| Feature set | Algorithm | Accuracy delta | Accuracy-positive folds | AUC delta | Failure-recall delta |
|---|---|---:|---:|---:|---:|
| conservative | XGBoost | +0.00178 | 2/3 | +0.00042 | -0.00180 |
| conservative | CatBoost | +0.00193 | 3/3 | +0.00007 | -0.01488 |
| conservative | LightGBM | +0.00115 | 2/3 | -0.00042 | +0.00164 |
| city | XGBoost | -0.00037 | 1/3 | +0.00067 | -0.00352 |
| city | CatBoost | -0.00004 | 1/3 | +0.00043 | -0.01210 |
| city | LightGBM | +0.00230 | 3/3 | +0.00023 | +0.00507 |
| reward_count_exploratory | XGBoost | -0.00171 | 1/3 | +0.00074 | -0.00597 |
| reward_count_exploratory | CatBoost | -0.00193 | 1/3 | +0.00076 | -0.01120 |
| reward_count_exploratory | LightGBM | -0.00137 | 0/3 | -0.00011 | -0.00433 |
| reward_ladder_exploratory | XGBoost | -0.00156 | 1/3 | -0.00002 | -0.00965 |
| reward_ladder_exploratory | CatBoost | -0.00178 | 1/3 | +0.00101 | -0.01619 |
| reward_ladder_exploratory | LightGBM | -0.00089 | 0/3 | +0.00050 | +0.00303 |

Inner thresholds stayed .5 for XGBoost and LightGBM. CatBoost selected .55 in all conservative folds, .5/.5/.55 for city, and .5/.5/.55 for each reward set. Inner threshold selection changed conservative CatBoost to .6662 accuracy and .6481 failure recall; city CatBoost to .6759 and .5935. AUC/Brier are unchanged by threshold. Increased failure recall did not create an accuracy improvement. Fold-level failures of the inner constraint on outer rows are recorded, not concealed.

The nested family selector chose XGBoost only in conservative fold 1, CatBoost only in city fold 2, and HGB elsewhere. Pooled selector accuracy was .6697 for conservative and .6769 for city, versus .6693/.6770 for fixed HGB. Both reward procedures retained HGB in every fold. Family/settings/threshold choices all used inner data only. The +.003 accuracy, at most .002 AUC loss and .01 failure-recall loss guardrails are minimum practical checks, not proof of superiority. Prior tuning and repeated inspection of these outer folds remain selection bias. Overlapping folds do not supply independent replications or confidence intervals.

Confusion counts at .5 (failed=0, successful=1):

| Feature set | Algorithm | TN | FP | FN | TP |
|---|---|---:|---:|---:|---:|
| conservative | HGB | 6814 | 5415 | 3495 | 11221 |
| conservative | XGBoost | 6792 | 5437 | 3425 | 11291 |
| conservative | CatBoost | 6632 | 5597 | 3261 | 11455 |
| conservative | LightGBM | 6834 | 5395 | 3484 | 11232 |
| city | HGB | 6990 | 5239 | 3463 | 11253 |
| city | XGBoost | 6947 | 5282 | 3430 | 11286 |
| city | CatBoost | 6842 | 5387 | 3316 | 11400 |
| city | LightGBM | 7052 | 5177 | 3463 | 11253 |
| reward_count_exploratory | HGB | 7220 | 5009 | 3033 | 11683 |
| reward_count_exploratory | XGBoost | 7147 | 5082 | 3006 | 11710 |
| reward_count_exploratory | CatBoost | 7083 | 5146 | 2948 | 11768 |
| reward_count_exploratory | LightGBM | 7167 | 5062 | 3017 | 11699 |
| reward_ladder_exploratory | HGB | 7368 | 4861 | 2906 | 11810 |
| reward_ladder_exploratory | XGBoost | 7250 | 4979 | 2830 | 11886 |
| reward_ladder_exploratory | CatBoost | 7170 | 5059 | 2756 | 11960 |
| reward_ladder_exploratory | LightGBM | 7405 | 4824 | 2967 | 11749 |

## Rewards, recommendations and global status

Reward count and the full ladder are separate exploratory comparisons. The ladder combines count, min/max/median log amounts and their log difference from goal. These features contain no reward quantities or backer selections. HGB ladder accuracy .7117 is a descriptive development result, not an eligible replacement.

2012 first-party sources document adding reward tiers while campaigns were live. Thus even historical count and amount distributions cannot be assumed to be launch snapshots. Our schema has no per-tier creation/revision history. Keep rewards exploratory until row-level initial snapshots and units are verified. [Kickstarter July 2012 report](https://www.kickstarter.com/blog/this-week-in-kickstarter-31), [V20 Companion creator FAQ](https://www.kickstarter.com/projects/200664283/v20-companion-deluxe-edition/faqs). Detailed provenance distinctions and references are in [the research note](additional_model_research.md).

Recommendation methods are separate from classification. The executable research example uses approved training rows only. Nearest neighbors use robust-scaled log goal/duration, with category/subcategory filters and self-ID exclusion. Outcomes/engagement do not determine distances. Benchmarks report all-campaign goal/duration quartiles and counts; a subcategory requires 30 rows, otherwise a supported parent category is explicitly used. Unknown categories or unsupported parents return unavailable. No geography distance or inferred country is used; comparisons span supported U.S. listed states. State-specific matching remains a future option requiring enough support. These 30-row rules are design choices, not statistical guarantees.

`recommendation_example.txt` contains one computed training-row example with provenance hashes, benchmarks, five neighbor IDs/distances, frozen-v1 predictions and a +20% goal scenario. Scenario outputs are model responses, not causal effects or recommended optimum settings. The object can later support local LLM explanations, but no LLM is invoked, raw campaign text is excluded, and explanation claims must be checked against the supplied facts. Retrieval relevance and downstream decision quality have not been validated.

Additional algorithms do not resolve global monetary units, historical snapshots or location coverage. Global includes U.S.; unknown locations are not verified international campaigns. No global model was trained or deployed. Before enabling global in Streamlit, resolve units/conversion and location policy, validate a separate conservative pipeline using official global training IDs, verify serialization, define its own validator/provenance contract and coordinate with Zasha. No Streamlit application code changed.

## Reproduction and checks

See [exact environment setup and commands](additional_model_reproduction.md). `.venv-model-comparison` overlays the reviewed base with XGBoost 3.4.1, CatBoost 1.2.10 and LightGBM 4.7.0. Original `.venv` and pinned production files are unchanged. Existing OpenMP was used. Dependency check and import/fit smoke checks passed. The new notebook ran to completion in a fresh kernel and saved all tables/figure. Full suite: 69 passed, no failures/errors/skips, with optional algorithms exercised in the overlay. Critical lint and whitespace checks pass.

Changes: new `utils/us_algorithm_comparison.py`, `utils/us_recommendation_research.py`, `tests/test_algorithm_comparison.py`, `notebooks/06_us_algorithm_comparison.ipynb`, `scripts/execute_algorithm_comparison.py`, `scripts/research_us_recommendations.py`, separate requirements, this report, research/reproduction notes and `docs/additional_model_results/`. `.gitignore` adds only the comparison environment rule on top of prior work. Existing uncommitted feature experiments are preserved.

No commit, merge or push. Remaining decisions: obtain reliable launch snapshots and funding units; review full-city inputs before activating existing v2; validate retrieval quality and support rules; agree future artifact distribution and global coverage. No classifier activation or causal recommendation is justified by this comparison.

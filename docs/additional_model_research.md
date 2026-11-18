# Reward eligibility and recommendation research

Reviewed November 10, 2026. This note separates historical launch evidence from plausible planning inputs. No reserved labels were read, reference code executed, model fitted or application changed for this research.

## Reward eligibility

Creators can decide an initial reward ladder before launch, so intended tier count, minimum/maximum/median amounts and amounts relative to goal are plausible planning inputs. That does not establish that this dataset's `levels` and `reward levels` describe the initial ladder. Kickstarter's launch checklist asks creators to review reward tiers before publishing and notes that tiers become uneditable after someone pledges. [Kickstarter launch-day checklist](https://d3mlfyygrfdi2i.cloudfront.net/LaunchDayChecklist.pdf?ref=tipsblog)

There is direct first-party evidence from the historical period that reward configurations changed during campaigns. Kickstarter's July 13, 2012 blog describes Black Moth Super Rainbow adding rewards during its campaign. A creator's February 21–23, 2012 V20 Companion FAQ explains that new pledge tiers had been added after earlier reward wording was written, and that a selected tier could not be edited. These are counterexamples to assuming an end-of-campaign tier list must equal its launch list. Neither source proves which rows in our dataset changed. [Kickstarter's July 2012 report](https://www.kickstarter.com/blog/this-week-in-kickstarter-31), [V20 Companion creator FAQ](https://www.kickstarter.com/projects/200664283/v20-companion-deluxe-edition/faqs)

The current Kickstarter help page, updated August 6, 2026, permits new rewards and editing rewards without backers while a project is live; selected tiers are restricted. It also permits live title/category edits. This corroborates the general snapshot concern, but current rules are not proof of the complete 2009–2012 policy or of any individual scraped value. [Kickstarter editing rules](https://help.kickstarter.com/en-us/articles/16236599-can-a-project-be-edited-after-launching)

The local source schema has a count (`levels`), a dollar-prefixed amount string (`reward levels`), status/outcome fields and a `funded date`, but no per-tier creation timestamp, reward revision history, launch-snapshot timestamp or documented currency conversion. The repository's source dictionary explicitly treats these meanings as unresolved. Prior training-source checks found parser/count agreement across all 26,945 U.S. training rows and 28 missing reward strings, but those checks cannot identify initial versus later tiers. These are local observations recorded in [the source dictionary](../data/data_dictionary.md) and [the training-source audit](launch_feature_results/input_review.txt).

The repository points to the [Parienza Kaggle listing](https://www.kaggle.com/datasets/parienza/kickstarter). The public page returned no readable metadata through the research tool. No original provider statement establishing launch-only rewards was obtained. A file name, plausible planning interpretation or agreement with the live page is insufficient evidence of launch-time provenance.

Decision: keep every reward experiment exploratory, including count-only models. Adding tiers can change count and the amount distribution even if already-selected amounts cannot be edited. Do not substitute a creator's planned ladder for a model trained on potentially final ladders and claim the same validation performance. Exclude backer selections, stock consumption, pledged money, updates and other activity. A future eligible dataset needs per-project launch time, an auditable initial reward snapshot captured by that time, tier creation/revision evidence, count/amount parsing rules and verified units. A later scrape with no history cannot reconstruct this reliably.

## Recommendation methods, separate from classification

The following are implementation proposals based on the approved historical training population, not validated claims about recommendation quality. All indexes, scaling statistics and benchmark tables must use the 26,945 official U.S. training IDs only. During evaluation, fit them again inside each fitting fold. Do not add their outcome summaries to classifier inputs.

| Method | Feasibility and proposed safeguards |
|---|---|
| Similar-campaign retrieval | Feasible as descriptive retrieval using log goal and planned duration, with category/subcategory filtering. Fit numeric scaling on training rows; use simple `NearestNeighbors` and return distance plus explicit matching rules. Exclude a query campaign's own project ID when evaluating an existing training row. Do not use success targets, pledged funds or engagement to choose neighbors. Neighbors need not imply comparable campaign quality. |
| Goal/duration benchmarks | Feasible as descriptive training quantiles with row count, cohort, units assumption and fallback level. Show median and interquartile range across successful and failed campaigns together; any outcome-conditioned comparison must be separately labeled. A historical common value is not an optimal setting or a success guarantee. |
| Scenario comparisons | Feasible using one frozen predictor and validated user-selected changes to goal/duration. Report original/changed inputs, model version, predicted probabilities and their difference. Keep category/location fixed unless the user supplies a meaningful change. Flag out-of-support values and avoid extrapolated optimization. These are model responses to input substitutions, not estimated causal effects. |
| Structured summary for a future local LLM | Feasible without training an LLM. Build a small deterministic object from validated inputs, model output, benchmark counts/quantiles, retrieved IDs/distances and limitations. The LLM would explain those supplied facts, not calculate probabilities, invent evidence or give causal advice. Keep raw campaign text outside instruction fields and validate any generated factual claims against the object. |

Nearest-neighbor search is supported by scikit-learn's unsupervised neighbor interface; distance and representation choices determine which observations count as similar. It is a retrieval mechanism, not evidence that an input intervention improves outcomes. [scikit-learn neighbors documentation](https://scikit-learn.org/stable/modules/neighbors.html)

A proposed starting support rule is at least 30 training campaigns per subcategory for a benchmark; otherwise use its parent category with an explicit fallback message. If the parent also has fewer than 30, return insufficient support instead of inventing a specific comparison. This threshold is a team design choice, not a statistically justified universal cutoff. Display actual counts and intervals/quantiles. If showing a success rate, also require at least 10 observed examples of each outcome; suppress it when support is smaller. Do not imply those small-sample checks remove selection bias or dataset shift.

For retrieval, return fewer than the requested k if the eligible group is small, rather than silently adding unrelated categories. Unseen subcategories can fall back only to a known parent supplied/validated by the taxonomy; unseen categories should return unavailable descriptive benchmarks/retrieval. The classifier may still use its existing encoding fallback, but the UI must distinguish that prediction from a supported historical comparison. For existing-row evaluation, excluding the same project is necessary even when repeated titles or creators remain.

Geography is a listed-location proxy. Do not infer registration country from a city name, treat city frequency as population or merge unknown places into verified international campaigns. Prefer a state-filtered comparison only when the state is supported and the filtered sample meets the support rule. Otherwise broaden within the U.S. cohort explicitly, or return no comparable sample. Currency assumptions must accompany every monetary benchmark. Do not pool international monetary values merely to increase sample size.

A future explanation object can use this shape:

```json
{
  "schema_version": 1,
  "model_id": "us_primary_model_v1",
  "cohort": "official U.S. listed-state/DC training proxy",
  "inputs": {"goal": 5000, "duration": 30, "category": "Art"},
  "prediction": {"success_probability": "computed by frozen pipeline"},
  "benchmark": {"group": "computed group", "n": "observed count", "fallback": "explicit level"},
  "neighbors": [{"project_id": "training ID", "distance": "computed distance"}],
  "limitations": ["historical sample", "units assumption", "associations are not causal"]
}
```

The placeholders above describe a proposal; they are not computed results or a production schema. In a working implementation use numeric types for computed values, a complete validated input set, artifact/cohort fingerprints, method/feature names and data/collection provenance. Do not let an LLM upgrade reward-experiment results into a launch-valid prediction claim.

## Global feasibility and dashboard requirements

Additional algorithms do not fix missing currency semantics, snapshot provenance or uncertain coverage. The global official training population includes U.S. campaigns; it is not disjoint from the U.S. training population. Retain the existing global IDs, and keep unknown locations distinct from verified named international locations. Existing counts and limitations are documented in [the pipeline review](launch_pipeline_review.md).

Keep the Streamlit global option unavailable until the team has documented common goal units or a reliable currency/time conversion recipe; a supported-location and unknown-location policy; a separate conservative global pipeline trained on official global training IDs with fold-fitted preprocessing; training-only nested comparison and failure/calibration checks; versioned artifact/metadata with serialization verification; and a separate input validator/output provenance contract. Broad global averages do not establish performance for sparse countries. If an eventual global model is enabled, identify its overlapping cohort and exploratory status and never route its requests through a U.S. artifact. Coordinate any interface change through Zasha. [Existing dashboard proposal](dashboard_model_interface.md)

# Funding-goal currency review

Reviewed November 17, 2026. Scope: the repository's `DSI_kickstarterscrape_dataset.csv`, linked to Kaggle `parienza/kickstarter`. No reserved labels were read, no model was trained, and no artifact was published.

The source history strongly suggests a pre-UK Kickstarter corpus. It does not yet verify the exact CSV's monetary semantics or support claims about today's international campaign markets. Do not infer exchange-rate conversion from a dollar sign, a listed country, or matching row counts.

## Evidence

| Source | Verified claim | Limit |
|---|---|---|
| [Kaggle dataset card](https://www.kaggle.com/datasets/parienza/kickstarter) and [public metadata API](https://www.kaggle.com/api/v1/datasets/view/parienza/kickstarter) | Metadata identifies `parienza/kickstarter`, version 1, initially released December 27, 2017. The API description and subtitle are empty. | No monetary dictionary, scraper provenance, conversion procedure, original snapshot date, or exact upstream file fingerprint was supplied by the retrieved metadata. The 2017 upload date is not a campaign or scrape date. |
| [Jeanne Pi's AppsBlogger article, republished with permission](https://venturebeat.com/entrepreneur/kickstarter-failures-revealed-what-can-you-learn-from-them-infographic) | Collector reports 45,815 projects scraped as of June 2, 2012. | This establishes the collector's corpus, not a byte-level identity with the local CSV. |
| [Mitra and Gilbert, author-hosted CSCW paper](https://eegilbert.org/papers/mitra.cscw14.kickstarter.pdf), Method | The researchers used AppsBlogger's 45,815 project URLs covering launches through June 2, 2012, and scraped pages again on August 3, 2012. | Their later page scrape is a separate dataset. It corroborates the original URL corpus, not the exact Kaggle file's fields, units, or snapshot. |
| [Kickstarter, October 31, 2012 launch announcement](https://www.kickstarter.com/blog/kickstarter-in-the-uk-0) | Kickstarter opened to UK projects on that date. It explicitly states that UK projects are listed in pounds sterling and outside-UK backers see an approximate USD conversion before pledging. | Displaying a conversion to backers does not establish that this CSV stored converted USD goals. |
| [Kickstarter's five-year UK retrospective](https://www.kickstarter.com/blog/celebrating-five-years-of-kickstarter-in-the-uk) | Kickstarter describes the UK launch as its first international expansion. | International listed project locations before that date are not proof of international registered campaign markets. |

The [2020 study of a 45,815-row public corpus](https://www.mdpi.com/2071-1050/12/2/599) is useful secondary context, but its source is a retired Google Fusion Table, not a currency specification. Its stated date range also differs from the collector's account. It should not settle exact lineage or monetary units.

The similarly named [Kaggle `kemical/kickstarter-projects`](https://www.kaggle.com/kemical/kickstarter-projects/data) is a different, later dataset. Its documented `usd_goal_real` and exchange-rate conversions cannot be transferred to the repository's CSV.

## Interpretation for the app

A pre-October-2012 corpus is consistent with projects using the original U.S. portal even when their listed activity location is abroad. This is a historical hypothesis supported by the source timeline, not verified row-level currency normalization. An "international" location subset must remain described as **explicit named non-U.S. listed locations**, not as all international registered Kickstarter campaigns.

The repository's [dictionary](../data/data_dictionary.md) has no verified currency field, and the [international workflow](international_only_workflow.md) explicitly distinguishes listed location from registration country. Country-relative goal medians do not solve unknown units: they retain raw goal and goal/day and can fall back to pooled medians.

Before enabling goal-based international or mixed probabilities, establish the exact source lineage and funding-unit interpretation, then save and evaluate each separate pipeline using official training membership only. Record any adopted unit assumption in metadata and the user-facing input label. Accepting arbitrary local-currency amounts would not be supported by the current evidence.

Offline training-only research may use an explicit, versioned monetary assumption to examine feasibility, but that is not verification of public prediction inputs. Keep the frozen U.S. pipeline unchanged. A future currency conversion must specify source currency, conversion date and rate source; no inferred conversion was introduced here.

The shortest next source check is an original scraper/data dictionary or a row-level match to archived original project metadata with explicit currency. Matching the AppsBlogger row count and a pre-UK date range alone is insufficient to claim that every local goal is normalized USD.

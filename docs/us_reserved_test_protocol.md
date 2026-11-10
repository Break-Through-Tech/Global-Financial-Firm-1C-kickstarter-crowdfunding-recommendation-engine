# Proposed U.S. reserved-test protocol

Status: awaiting explicit approval from Anna and Zasha, ideally with advisor input. This document and the prepared notebook do not authorize test scoring. No reserved labels have been opened in this milestone.

Review the October 15 audit, its remediation report, `evaluation_decisions.md`, and `final_model_selection.md`. The reserved partition contains 11,475 campaigns. The audit identified 1,486 IDs in historical raw training, leaving 9,989 not known to have that exposure. The latter group is not established as independent: other exploration and shared development history may affect it.

Before approval, freeze the locally saved `models/us_primary_model_v1.joblib` and its metadata, including artifact SHA-256, exact feature policy, HGB settings, and threshold .5. Record both approvers, approval date and the frozen hash. Do not run training or accuracy notebooks after this freeze. Use the environment recorded in the metadata.

After explicit approval, set the approval fields in `notebooks/08_us_reserved_evaluation.ipynb` and run it once from a fresh kernel. The notebook must verify the approved hash before loading labels and stop if an existing result file is present. Read reserved IDs in saved order and verify their count, uniqueness and disjointness from training. Read only the five approved inputs from cleaned data. Compare the saved reserved target column against `y_test.csv` in the same order before scoring.

Reconstruct historical raw-training membership with the audited raw snapshot and the original seed-42 split, without fitting any model. Assert exactly 1,486 reserved IDs are exposed. Report metrics for all 11,475, the 1,486 historically exposed, and the 9,989 not-known-exposed campaigns. Do not pool the latter into a claim of pristine independence. These groups differ in membership and may have different case mix; their score difference does not estimate exposure-caused optimism.

Compute accuracy, ROC-AUC, precision/recall/F1/support for both classes, failure recall, Brier score, confusion matrix and false-positive/false-negative counts. Retain threshold .5 and the frozen pipeline. Save one local result JSON with cohort sizes, metrics, hash, approval record, and limitations. Do not choose features, adjust thresholds, calibrate or retrain from these results. Unexpected inconsistencies stop the run and require review, not a new test-driven modeling round.

Report the result as retrospective evaluation on historically exposed data. External or prospective evaluation would support stronger generalization claims. Goal currency, duration meaning, snapshot provenance, launch availability and listed-location scope remain unresolved.

Approval record to complete after explicit team agreement:

- Anna approval and date: pending
- Zasha approval and date: pending
- Approved frozen artifact SHA-256: pending
- Execution date and operator: pending

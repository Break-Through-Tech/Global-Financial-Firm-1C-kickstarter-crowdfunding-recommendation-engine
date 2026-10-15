# Proposed evaluation policy — October 15, 2026

Status: proposed for Anna and Zasha's agreement, ideally with Parth Rana's input.
Parth has confirmed U.S. primary scope. This does not approve an evaluation
protocol or resolve historical holdout exposure.

1. Develop the primary model using the existing **26,945 official U.S. training
   IDs and labels**. Retain all official split memberships, labels and ordering.
   Combined/global and named-international results remain secondary experiments.
2. Freeze the conservative policy: goal, duration, category, subcategory and listed
   location/state proxy; fold-fitted engineered references only. Exclude outcomes,
   engagement, title/slug/renamed, rewards/levels and reconstructed timing. Even
   retained fields rely on unverified launch-availability and source-meaning assumptions.
3. Refit all learned processing within each training fold. Keep feature selection,
   hyperparameter and threshold selection inside validation boundaries. Use nested
   CV when estimating performance of a selection procedure. Existing training CV
   is development evidence and has already informed candidate selection.
4. Label reserved data **historically exposed**: raw 01 exploration and original
   05 modeling trained on 1,630 current combined test IDs (1,486 U.S.; 84
   international). Current official train/test membership is disjoint. No measured
   estimate of exposure-related optimism is available.
5. Do not calculate or publish final held-out scores until Anna and Zasha agree
   on the protocol, ideally with advisor input. Prespecify model choice, thresholds,
   metrics and reporting limitations using training data. A later approved score
   is retrospective evaluation on historically exposed data, not a pristine holdout.
6. Rerandomizing the same historically explored dataset is not a guaranteed remedy
   and is not part of this patch. Seek a genuinely independent external or prospective
   evaluation if available and separately authorized. Do not claim prospective
   generalization based only on this historical dataset.

Training-only 06 collaboration can proceed now under these boundaries. No 06/07,
final production fitting or final held-out scoring was performed in remediation.
See [validation provenance](model_validation.md), [handoff](06_handoff.md) and
[verification](workflow_audit_remediation_2026-10-15.md).

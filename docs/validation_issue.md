# Validate prediction-time features and preprocessing before tuning

## Objective
Review potential look-ahead variables, test the target encoder's training-versus-inference behavior, and compare sensitivity of baseline models using only the official training partition.

## Scope
- Confirm the meaning of `funded date` with source documentation and whether title/reward values represent launch-time snapshots.
- Discuss `renamed` with feature-code owner; conservatively exclude via `exclude_renamed=True` in an experiment without changing the shared feature generator.
- Fix and test the target encoder's index-collision behavior and OOF fold-specific prior.
- Record the U.S.-only scope decision from the challenge brief; do not filter based on unreliable two-letter suffix alone.
- Run training-only 3-fold sensitivity checks and an optional chronological holdout check.
- Update `docs/model_validation.md` with evidence, changes, and results; keep `X_test` untouched.

## Acceptance criteria
- [ ] `python3 -m unittest discover -s tests -v` passes.
- [ ] Teammate reviews `utils/preprocessing.py` changes.
- [ ] Validation notebook runs with ignored local processed data; figures and scores are documented.
- [ ] Source-field and U.S. scope questions are resolved or explicitly blocked/documented.
- [ ] No held-out test metrics are computed or used for selection.

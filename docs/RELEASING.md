# Publishing the v0.1.0 GitHub Release

The `docs/releases/v0.1.0.md` file is a **draft** until the release is actually created on GitHub. Tag the reviewed **main** commit, not an unmerged feature branch.

1. Merge the reviewed `feat/US-data` pull request into `main`.
2. Verify the Actions **Python CI** check is green and the release notes accurately describe the merged code.
3. From a clean local repository with the [GitHub CLI](https://cli.github.com/) installed and authenticated, run:

```bash
git switch main
git pull --ff-only origin main
git status
gh auth status
gh release create v0.1.0 \
  --target main \
  --title "v0.1.0 - U.S. Baselines" \
  --notes-file docs/releases/v0.1.0.md
```

The command creates a tag and publishes a GitHub Release. Check whether `v0.1.0` already exists before running it; don't create a duplicate. The GitHub CLI requires permissions to publish a release in the organization repository. If you lack those permissions, ask a maintainer to create the release using the prepared notes.

Do not attach third-party raw/processed campaign data, local `.env` files, Drive tokens, or cached model artifacts. This fixed historical dataset does not need a scheduled data-ingestion workflow.

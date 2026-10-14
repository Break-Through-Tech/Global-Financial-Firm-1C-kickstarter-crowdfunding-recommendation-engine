"""Auditable U.S. state-suffix proxy for the historical Kickstarter CSV.

The source has `location` strings, not a verified `country` column. A
matching U.S. state/DC suffix is a candidate U.S. location, not proof of
country. Do not claim a verified U.S.-only cohort without source/advisor
approval. No global data or official global split is modified here.
"""

from pathlib import Path
import pandas as pd

US_STATE_CODES = frozenset("""
AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS
MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI
WY DC
""".split())


def project_root(start=None):
    """Find the repository from root, notebooks/, or notebooks/us_only/."""
    start = Path.cwd() if start is None else Path(start)
    for folder in (start, *start.parents):
        if (folder / "utils" / "preprocessing.py").is_file():
            return folder
    raise FileNotFoundError("Run within the Kickstarter repository checkout")


def state_suffix(locations):
    """Extract uppercase two-letter terminal location suffixes."""
    return (locations.astype("string").str.strip().str.upper()
            .str.extract(r",\s*([A-Z]{2})\s*$", expand=False))


def us_state_proxy_mask(df):
    """Provisional geographic filter; does not establish nationality."""
    if "location" not in df.columns:
        raise KeyError("Expected a 'location' column")
    return state_suffix(df["location"]).isin(US_STATE_CODES).fillna(False).astype(bool)


def us_location_audit(df):
    """Return counts and a reviewable list of excluded terminal suffixes."""
    suffix = state_suffix(df["location"])
    mask = us_state_proxy_mask(df)
    summary = pd.DataFrame([{
        "all_rows": len(df),
        "candidate_us_rows": int(mask.sum()),
        "excluded_or_uncertain_rows": int((~mask).sum()),
        "missing_locations": int(df["location"].isna().sum()),
        "no_two_letter_suffix": int(suffix.isna().sum()),
    }])
    excluded = (suffix[~mask].fillna("<none>")
                .value_counts(dropna=False).rename_axis("suffix")
                .reset_index(name="n"))
    return summary, excluded


def align_to_global_split(us_clean, global_train_ids, global_test_ids):
    """Restrict original untouched global holdout, avoiding a fresh split.

    This preserves the existing global test boundary and permits future
    global-vs-U.S. evaluations on identical U.S. test campaigns.
    """
    if not us_clean["project id"].is_unique:
        raise ValueError("U.S. cleaned project IDs are not unique")
    if not global_train_ids["project id"].is_unique or not global_test_ids["project id"].is_unique:
        raise ValueError("Original split IDs must be unique")
    train_set = set(global_train_ids["project id"])
    test_set = set(global_test_ids["project id"])
    if train_set & test_set:
        raise ValueError("Global train/test IDs overlap")
    settled = us_clean[us_clean["status"].isin(["successful", "failed"])].copy()
    us_ids = set(settled["project id"])
    if not us_ids.issubset(train_set | test_set):
        raise ValueError("U.S. settled campaigns absent from original global split; regenerate global artifacts first")
    # Keep the global split order so comparisons can reuse the exact rows.
    kept_train = global_train_ids[global_train_ids["project id"].isin(us_ids)].copy()
    kept_test = global_test_ids[global_test_ids["project id"].isin(us_ids)].copy()
    if len(kept_train) + len(kept_test) != len(settled):
        raise AssertionError("U.S. partition failed to cover every settled campaign")
    status = settled.set_index("project id")["status"]
    for ids in (kept_train, kept_test):
        expected = status.loc[ids["project id"]].eq("successful").astype(int).to_numpy()
        if "target" in ids and not (ids["target"].astype(int).to_numpy() == expected).all():
            raise ValueError("Saved target disagrees with campaign status")
        ids["target"] = expected
    return kept_train.reset_index(drop=True), kept_test.reset_index(drop=True)

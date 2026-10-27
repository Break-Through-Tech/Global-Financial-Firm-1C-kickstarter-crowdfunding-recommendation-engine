"""Shared training-only helpers for the official U.S. modeling workflow.

The loader intentionally reads only the U.S. cleaned rows and saved training
IDs. It never opens the reserved IDs or labels. Model preprocessing is built
inside each estimator pipeline so cross-validation refits all learned state.
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from utils.preprocessing import drop_leaks, make_pipeline

SEED = 42
EXPECTED_TRAIN_ROWS = 26_945
EXPECTED_SETTLED_ROWS = 38_420
CONSERVATIVE_POLICY = dict(
    include_timing=False,
    include_flags=False,
    include_reward=False,
    exclude_renamed=True,
)
OUTCOME_COLUMNS = frozenset(
    ["status", "target", "pledged", "funded percentage", "backers", "comments", "updates"]
)
ACTIVE_COLUMNS = frozenset(["goal", "duration", "category", "subcategory", "location"])


def repository_root(start=None):
    start = Path.cwd() if start is None else Path(start)
    for folder in (start, *start.parents):
        if (folder / "utils" / "preprocessing.py").is_file():
            return folder
    raise FileNotFoundError("Run from inside the Kickstarter repository")


def load_us_training_data(root=None):
    """Load official U.S. training rows by saved project-ID membership only."""
    root = repository_root(root)
    us_dir = root / "data" / "Processed-Data" / "us_only"
    clean_path = us_dir / "DSI_kickstarterscrape_dataset_cleaned.csv"
    train_path = us_dir / "train_project_ids.csv"
    for path in (clean_path, train_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing official U.S. training artifact: {path}")

    clean = pd.read_csv(clean_path)
    train_ids = pd.read_csv(train_path)
    required_id_columns = {"project id", "target"}
    if not required_id_columns.issubset(train_ids.columns):
        raise AssertionError(f"Training IDs must contain {required_id_columns}")
    if not clean["project id"].is_unique:
        raise AssertionError("Cleaned U.S. project IDs are not unique")
    if not train_ids["project id"].is_unique:
        raise AssertionError("Official U.S. training project IDs are not unique")
    if not set(train_ids["project id"]).issubset(set(clean["project id"])):
        raise AssertionError("An official U.S. training ID is missing from cleaned rows")

    rows = clean.set_index("project id", drop=False).loc[train_ids["project id"]]
    rows = rows.reset_index(drop=True)
    y = train_ids["target"].astype(int).reset_index(drop=True)
    status_y = rows["status"].eq("successful").astype(int)
    if not np.array_equal(status_y.to_numpy(), y.to_numpy()):
        raise AssertionError("Saved training labels disagree with cleaned status")
    if len(rows) != EXPECTED_TRAIN_ROWS:
        raise AssertionError(f"Expected {EXPECTED_TRAIN_ROWS} U.S. training rows, found {len(rows)}")
    if len(set(rows["project id"])) != len(rows):
        raise AssertionError("Training project IDs are not unique after reconstruction")
    settled = clean[clean["status"].isin(["successful", "failed"])]
    if len(settled) != EXPECTED_SETTLED_ROWS:
        raise AssertionError(f"Expected {EXPECTED_SETTLED_ROWS} settled U.S. rows, found {len(settled)}")
    if set(rows["project id"]) != set(train_ids["project id"]):
        raise AssertionError("Training population membership changed during reconstruction")
    if set(y.unique()) != {0, 1}:
        raise AssertionError("Training labels must be binary 0/1")

    X = drop_leaks(rows.drop(columns=["status"]))
    if OUTCOME_COLUMNS.intersection(X.columns):
        raise AssertionError("Outcome-derived columns entered X: " + repr(sorted(OUTCOME_COLUMNS.intersection(X.columns))))
    if not ACTIVE_COLUMNS.issubset(X.columns):
        raise AssertionError("Missing conservative active columns")
    return X, y, rows["project id"].copy()


def development_cv():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def scoring():
    return {"roc_auc": "roc_auc", "accuracy": "accuracy", "f1": "f1"}


def conservative_pipeline(model, categorical_strategy):
    return make_pipeline(model, categorical_strategy=categorical_strategy, **CONSERVATIVE_POLICY)


def run_randomized_search(X, y, model, categorical_strategy, distributions, n_iter=15):
    """Run an ordinary training-CV search and return its search plus trial log.

    The returned best score is a tuning estimate, not an unbiased generalization
    estimate, because the same CV results select the hyperparameters.
    """
    search = RandomizedSearchCV(
        conservative_pipeline(model, categorical_strategy),
        param_distributions=distributions,
        n_iter=n_iter,
        scoring=scoring(),
        cv=development_cv(),
        n_jobs=1,
        random_state=SEED,
        refit="roc_auc",
        return_train_score=True,
        error_score="raise",
    )
    started = time.perf_counter()
    search.fit(X, y)
    elapsed = time.perf_counter() - started
    cv = pd.DataFrame(search.cv_results_)
    trials = cv[["rank_test_roc_auc", "mean_test_roc_auc", "std_test_roc_auc", "mean_train_roc_auc", "mean_test_accuracy", "mean_test_f1", "mean_fit_time", "params"]].copy()
    trials = trials.rename(columns={
        "rank_test_score": "trial_rank",
        "rank_test_roc_auc": "trial_rank",
        "mean_test_roc_auc": "cv_roc_auc",
        "std_test_roc_auc": "cv_roc_auc_sd",
        "mean_train_roc_auc": "train_roc_auc",
        "mean_test_accuracy": "cv_accuracy",
        "mean_test_f1": "cv_f1",
        "mean_fit_time": "mean_fit_seconds",
    })
    trials["overfit_gap"] = trials["train_roc_auc"] - trials["cv_roc_auc"]
    trials["elapsed_seconds"] = elapsed
    return search, trials.sort_values("cv_roc_auc", ascending=False).reset_index(drop=True)


def search_result(search, trials, model_name):
    """Return a compact, JSON/table-friendly summary for docs and notebooks."""
    best_row = trials.iloc[0]
    return {
        "model": model_name,
        "cohort": "official U.S. training rows",
        "feature_policy": "conservative: no timing, flags, rewards, or renamed",
        "cv": "5-fold StratifiedKFold shuffle=True random_state=42",
        "search_iterations": int(len(trials)),
        "estimate": "ordinary tuning CV; optimistic selection estimate",
        "cv_roc_auc": float(search.best_score_),
        "cv_roc_auc_sd": float(best_row["cv_roc_auc_sd"]),
        "train_roc_auc": float(best_row["train_roc_auc"]),
        "overfit_gap": float(best_row["overfit_gap"]),
        "accuracy": float(search.cv_results_["mean_test_accuracy"][search.best_index_]),
        "f1": float(search.cv_results_["mean_test_f1"][search.best_index_]),
        "fit_seconds": float(best_row["mean_fit_seconds"]),
        "elapsed_seconds": float(best_row["elapsed_seconds"]),
        "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
    }

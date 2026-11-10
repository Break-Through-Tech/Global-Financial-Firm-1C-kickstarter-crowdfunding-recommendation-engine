
import sys
import warnings
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parent))
from modeling import load, parse_tiers
from utils.us_scope import state_suffix, us_state_proxy_mask  # noqa: E402
from catboost import CatBoostClassifier
from utils.get_google_drive_data import load_csv_from_drive
from utils.google_drive_client import get_drive_client

RAW_COLUMNS = ["name", "category", "subcategory", "location", "goal", "duration",
               "reward levels", "launch"]

NUMERIC_FEATURES = [
    "log_goal", "duration", "n_tiers", "log_min_tier", "log_max_tier", "log_median_tier",
    "tier_spread", "entry_vs_goal", "launch_month_sin", "launch_month_cos",
    "name_len", "name_words", "name_allcaps", "name_excl", "has_location",
]
CATEGORICAL_FEATURES = ["category", "subcategory"]
SEED=7
MODEL_DIR = Path(__file__).resolve().parent / "trained_models"

def obtain_clean_location(df):
    """
    Verify location is present
    Determine state suffix
    Determine whether it belongs to the US
    """
    loc = df["location"]
    nonblank = loc.astype("string").str.strip().fillna("") != ""
    has = (loc.notna() & nonblank).astype(bool)
    state = state_suffix(loc)
    is_candidate = us_state_proxy_mask(df).astype(bool)
    return has.astype(int), state, is_candidate

def scope_mask(df, scope):

    has, _, is_candidate = obtain_clean_location(df)
    has, is_candidate = has.astype(bool).values, is_candidate.values
    if scope == "us":
        return is_candidate
    if scope == "int":
        return has & ~is_candidate
    return np.ones(len(df), dtype=bool)

class OnlineFeatureBuilder(BaseEstimator, TransformerMixin):
    # using standard scikit-learn conventions
    def __init__(self, scope="mixed"):
        self.scope = scope

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = pd.DataFrame(X).copy()
        idx = df.index

        goal = pd.to_numeric(df["goal"], errors="coerce")
        duration = pd.to_numeric(df["duration"], errors="coerce")
        
        tiers = df["reward levels"].map(parse_tiers)

        def tier_stat(fn):
            return tiers.map(lambda t: fn(t) if t else np.nan).astype(float)
        
        # feature eng for tiers
        log_min = np.log1p(tier_stat(min))
        log_max = np.log1p(tier_stat(max))
        log_med = np.log1p(tier_stat(np.median))
        log_goal = np.log1p(goal)

        # preprocessing
        launch = pd.to_datetime(df["launch"], utc=True)
        month = launch.dt.month

        nm = df["name"].fillna("").astype(str)

        has_loc, state, is_candidate = obtain_clean_location(df)

        F = pd.DataFrame({
            "log_goal": log_goal,
            "duration": duration,
            "n_tiers": tiers.map(len).astype(float),
            "log_min_tier": log_min,
            "log_max_tier": log_max,
            "log_median_tier": log_med,
            "tier_spread": log_max - log_min,
            "entry_vs_goal": log_min - log_goal,
            "launch_month_sin": np.sin(2 * np.pi * month / 12),
            "launch_month_cos": np.cos(2 * np.pi * month / 12),
            "name_len": nm.str.len(),
            "name_words": nm.str.split().map(len),
            "name_allcaps": (nm.str.upper() == nm).astype(int),
            "name_excl": nm.str.count("!").clip(0, 3),
            "has_location": has_loc.values,
            "category": df["category"].fillna("__NA__").astype(str),
            "subcategory": df["subcategory"].fillna("__NA__").astype(str),
            "state": state.fillna("__NA__").astype(str).values,
            "is_us_candidate": is_candidate.astype(int).values,
        }, index=idx)
        return F
    
def make_online_pipeline(scope="us", model="boosting"):
    if scope not in ["us", "mixed", "int"]:
        raise ValueError("Please enter a valid scope")
    
    numeric = list(NUMERIC_FEATURES)
    categorical = list(CATEGORICAL_FEATURES)

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20,
                                  sparse_output=False), categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
        
    clf = CatBoostClassifier(iterations=600, learning_rate=0.05, depth=6,
                                 random_seed=SEED, verbose=0, allow_writing_files=False)
    return Pipeline([
        ("features", OnlineFeatureBuilder(scope=scope)),
        ("preprocess", preprocess),
        ("model", clf),
    ])

def train_online_model(
        scope="mixed",
        df=None,
        train_mask=None,
    ):
    if df is None:
        service = get_drive_client()
        df = load_csv_from_drive(service, clean=True)

    mask = scope_mask(df, scope)

    # do you want train/test dataset
    if train_mask is not None:
        mask = mask & np.asarray(train_mask, dtype=bool)

    pipe = make_online_pipeline(scope=scope)
    pipe.fit(df.loc[mask, RAW_COLUMNS], df.loc[mask, "success"].astype(int).values)
    return pipe

def save_model(pipe, path=None):
    """
    Pickle a fitted pipeline to disk and return the path.
 
    """
    if path is None:
        scope = pipe.named_steps["features"].scope
        clf = type(pipe.named_steps["model"]).__name__.lower()
        path = MODEL_DIR / f"online_{scope}_{clf}.pkl"
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "wb") as fh:
        pickle.dump(pipe, fh, protocol=pickle.HIGHEST_PROTOCOL)
    
    return path

def predict_campaign(pipe, features):
    X = pd.DataFrame([features]) if isinstance(features, dict) else features
    return pipe.predict_proba(X[RAW_COLUMNS])[:, 1]

def load_model(path):
 
    with open(path, "rb") as fh:
        pipe = pickle.load(fh)

    if not isinstance(pipe, Pipeline) or "features" not in pipe.named_steps:
        raise ValueError(f"{path} does not contain an online prediction pipeline")
    
    return pipe

if __name__ == "__main__":
    example = {
        "name": "Indie Album", "category": "Music", "subcategory": "Rock",
        "location": "Austin, TX", "goal": 5000, "duration": 30,
        "reward levels": "$10,$25,$100,$500", "launch": "2026-03-01",
    }
    for s in ("us", "mixed", "int"):
        p = predict_campaign(train_online_model(scope=s), example)[0]
        print(f"  {s:14s} P(funded) = {p:.3f}")
import pandas as pd
import numpy as np
import re

LEAK_COLS = ["pledged", "backers", "funded percentage", "updates", "comments"]


def wilson(k, n, z=1.96):
    "95% Wilson score interval for k successes in n trials. Works on scalars or arrays."
    k, n = np.asarray(k, dtype=float), np.asarray(n, dtype=float)
    p = k / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return centre - half, centre + half


def rate_table(data, by, target="target", min_n=50):
    "Success rate per group, with a Wilson CI and the gap from the overall rate."
    y = data[target]
    groups = data[by] if isinstance(by, str) else pd.Series(by, index=data.index)
    g = y.groupby(groups, observed=True).agg(n="size", successes="sum")
    g["rate"] = g["successes"] / g["n"]
    g["ci_lo"], g["ci_hi"] = wilson(g["successes"], g["n"])
    base = y.mean()
    g["vs_base"] = g["rate"] - base
    g = g[g["n"] >= min_n]
    g.attrs["base"] = base
    return g

def rate_plot(ax, table, title, labels=None, color="#2c6f52", rotate=0, base=None):
    "Bar chart of a rate_table: one bar per group, CI error bars, dashed overall-rate line."
    if base is None:
        base = table.attrs.get("base", (table["rate"] * table["n"]).sum() / table["n"].sum())
    x = np.arange(len(table))
    ax.bar(x, table["rate"], color=color, alpha=0.85)
    ax.errorbar(x, table["rate"],
                yerr=[table["rate"] - table["ci_lo"], table["ci_hi"] - table["rate"]],
                fmt="none", ecolor="k", capsize=3, lw=1)
    ax.axhline(base, color="#8a9590", ls="--", lw=1, label=f"overall {base:.1%}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels if labels is not None else table.index,
                       rotation=rotate, ha="right" if rotate else "center", fontsize=8.5)
    for i, n in enumerate(table["n"]):
        ax.text(i, 0.015, f"n={n:,}", ha="center", fontsize=7, color="white", rotation=90)
    ax.set_ylim(0, 1); ax.set_ylabel("success rate"); ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)


def parse_tiers(s):
    "'$25,$50,$1,000' -> [25.0, 50.0, 1000.0]. Split on $, not comma."
    out = []
    for chunk in str(s).split("$"):
        chunk = chunk.strip().rstrip(",").replace(",", "")
        if chunk:
            try:
                v = float(chunk)
            except ValueError:
                continue
            if np.isfinite(v) and v > 0:   # float("nan") parses fine, so check
                out.append(v)
    return out


def add_base_columns(df, parse_dates=True, parse_rewards=True, parse_url=True):
    "Parse the raw columns the other builders need."
    out = df.copy()

    # Deadline semantics are not verified for this dataset. Conservative
    # launch-time models must not construct features from this field.
    if parse_dates:
        out["end"] = pd.to_datetime(out["funded date"], format="mixed", utc=True)
        out["launch"] = out["end"] - pd.to_timedelta(out["duration"], unit="D")

    if parse_rewards:
        tiers = out["reward levels"].map(parse_tiers)
        out["tiers"] = tiers
        out["min_tier"] = tiers.map(lambda t: min(t) if t else np.nan)
        out["max_tier"] = tiers.map(lambda t: max(t) if t else np.nan)
        out["median_tier"] = tiers.map(lambda t: np.median(t) if t else np.nan)
        out["n_tiers"] = tiers.map(len)

    if parse_url:
        parts = out["url"].astype(str).str.extract(r"projects/([^/]+)/([^/?#]+)")
        out["creator"] = parts[0]
        out["slug"] = parts[1]

    out["state"] = out["location"].astype(str).str.extract(r",\s*([A-Z]{2})\s*$")[0]
    return out


def ask_features(df, subcat_median=None, cat_median=None, include_reward=True):
    "Size of the ask, absolute and relative to similar projects."
    f = pd.DataFrame(index=df.index)
    lg = np.log1p(df["goal"])
    f["log_goal"] = lg

    # pass the medians back in for test data so they are only fitted once
    if subcat_median is None:
        subcat_median = lg.groupby(df["subcategory"]).median()
    if cat_median is None:
        cat_median = lg.groupby(df["category"]).median()

    overall = subcat_median.median()
    f["goal_vs_subcat"] = lg - df["subcategory"].map(subcat_median).fillna(overall)
    f["goal_vs_cat"] = lg - df["category"].map(cat_median).fillna(overall)

    if include_reward:
        f["goal_per_tier"] = lg - np.log1p(df["n_tiers"])
    f["goal_per_day"] = lg - np.log1p(df["duration"])
    if include_reward:
        f["backers_needed"] = lg - np.log1p(df["median_tier"])
    return f, subcat_median, cat_median


def reward_features(df):
    "Shape of the reward ladder, not just its endpoints."
    f = pd.DataFrame(index=df.index)
    tiers = df["tiers"]

    f["n_tiers"] = df["n_tiers"]
    f["log_min_tier"] = np.log1p(df["min_tier"])
    f["log_max_tier"] = np.log1p(df["max_tier"])
    f["log_median_tier"] = np.log1p(df["median_tier"])
    f["tier_spread"] = f["log_max_tier"] - f["log_min_tier"]

    f["entry_vs_goal"] = f["log_min_tier"] - np.log1p(df["goal"])

    # average multiplier from one tier to the next
    f["geo_step"] = tiers.map(
        lambda t: (max(t) / max(min(t), 1)) ** (1 / (len(t) - 1)) if len(t) > 1 else np.nan
    )
    f["n_cheap"] = tiers.map(lambda t: sum(x < 25 for x in t) if t else np.nan)
    f["share_under_50"] = tiers.map(lambda t: np.mean([x < 50 for x in t]) if t else np.nan)
    return f


def creator_history(df):
    """Creator's past results, using only what was known on launch day.

    A past campaign counts only if it ended before this one launched. Grouping
    by creator and averaging the target instead leaks, since 92% of creators
    have a single campaign. Needs a target column, so training rows only.
    """
    f = pd.DataFrame(index=df.index, dtype=float)
    prior_n = np.zeros(len(df))
    prior_s = np.zeros(len(df))
    prior_launches = np.zeros(len(df))

    pos = {ix: i for i, ix in enumerate(df.index)}
    for _, g in df.groupby("creator"):
        if len(g) == 1:
            continue
        launch, end = g["launch"].values, g["end"].values
        won = g["target"].astype(int).values
        for i, ix in enumerate(g.index):
            done = end < launch[i]
            j = pos[ix]
            prior_n[j] = done.sum()
            prior_s[j] = won[done].sum()
            prior_launches[j] = (launch < launch[i]).sum()

    f["prior_completed"] = prior_n
    f["prior_successes"] = prior_s
    f["prior_launches"] = prior_launches
    f["is_repeat"] = (prior_launches > 0).astype(int)
    f["prior_rate"] = np.where(prior_n > 0, prior_s / np.maximum(prior_n, 1), np.nan)
    return f


def effort_flags(df):
    "Small signs of how much thought went into the campaign setup."
    f = pd.DataFrame(index=df.index)
    goal = df["goal"]
    dur = df["duration"].round()
    name = df["name"].astype(str)

    f["goal_round_1k"] = (goal % 1000 == 0).astype(int)
    f["goal_odd"] = (goal % 100 != 0).astype(int)
    f["duration_default_30"] = (dur == 30).astype(int)
    f["duration_default_60"] = (dur == 60).astype(int)

    f["has_location"] = df["location"].notna().astype(int)
    f["custom_creator"] = (~df["creator"].astype(str).str.fullmatch(r"\d+")).astype(int)

    # the slug is the original title, so a mismatch means they renamed it later
    slug_now = name.map(lambda s: re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-"))
    cut = df["slug"].astype(str).str.len().clip(upper=50)
    f["renamed"] = [
        int(a[:n] != b[:n]) for a, b, n in zip(slug_now, df["slug"].astype(str), cut)
    ]

    f["title_len"] = name.str.len()
    f["title_words"] = name.str.split().map(len)
    f["title_has_subtitle"] = name.str.contains(r"[:\-–—]\s").astype(int)
    f["title_allcaps"] = (name.str.upper() == name).astype(int)
    f["title_excl"] = name.str.contains("!").astype(int)
    return f


def timing_features(df):
    "When the campaign launched. Month is cyclical so it gets sin and cos."
    f = pd.DataFrame(index=df.index)
    launch = df["launch"]
    month = launch.dt.month
    f["launch_month_sin"] = np.sin(2 * np.pi * month / 12)
    f["launch_month_cos"] = np.cos(2 * np.pi * month / 12)
    f["launch_dow"] = launch.dt.dayofweek
    f["launch_weekend"] = (f["launch_dow"] >= 5).astype(int)
    f["duration"] = df["duration"]
    return f

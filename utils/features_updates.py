import numpy as np
import pandas as pd

LEAK_COLS = ["pledged", "backers", "funded percentage", "updates", "comments"]


def wilson(k, n, z=1.96):
    """95% Wilson score interval for k successes in n trials."""
    k, n = np.asarray(k, dtype=float), np.asarray(n, dtype=float)
    p = k / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return centre - half, centre + half


def rate_table(data, by, target="target", min_n=50):
    """Success rate per group, with a Wilson CI and gap from the overall rate."""
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
    """Bar chart of a rate_table with CI bars and an overall-rate reference line."""
    if base is None:
        base = table.attrs.get("base", (table["rate"] * table["n"]).sum() / table["n"].sum())

    x = np.arange(len(table))
    ax.bar(x, table["rate"], color=color, alpha=0.85)
    ax.errorbar(x, table["rate"], yerr=[table["rate"] - table["ci_lo"], table["ci_hi"] - table["rate"]], fmt="none", ecolor="k", capsize=3, lw=1)
    ax.axhline(base, color="#8a9590", ls="--", lw=1, label=f"overall {base:.1%}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels if labels is not None else table.index, rotation=rotate, ha="right" if rotate else "center", fontsize=8.5)

    for i, n in enumerate(table["n"]):
        ax.text(i, 0.015, f"n={n:,}", ha="center", fontsize=7, color="white", rotation=90)

    ax.set_ylim(0, 1)
    ax.set_ylabel("success rate")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)


def parse_tiers(s):
    """Parse '$25,$50,$1,000' into [25.0, 50.0, 1000.0]."""
    out = []
    for chunk in str(s).split("$"):
        chunk = chunk.strip().rstrip(",").replace(",", "")
        if not chunk:
            continue
        try:
            value = float(chunk)
        except ValueError:
            continue
        if np.isfinite(value) and value > 0:
            out.append(value)
    return out


def add_base_columns(df):
    """Parse raw columns needed by the feature builders."""
    out = df.copy()
    out["end"] = pd.to_datetime(out["funded date"], format="mixed", utc=True)
    out["launch"] = out["end"] - pd.to_timedelta(out["duration"], unit="D")

    tiers = out["reward levels"].map(parse_tiers)
    out["tiers"] = tiers
    out["min_tier"] = tiers.map(lambda t: min(t) if t else np.nan)
    out["max_tier"] = tiers.map(lambda t: max(t) if t else np.nan)
    out["median_tier"] = tiers.map(lambda t: np.median(t) if t else np.nan)
    out["n_tiers"] = tiers.map(len)

    parts = out["url"].astype(str).str.extract(r"projects/([^/]+)/([^/?#]+)")
    out["creator"] = parts[0]
    out["slug"] = parts[1]
    out["state"] = out["location"].astype("string").str.extract(r",\s*([A-Za-z]{2})\s*$", expand=False).str.upper()
    return out


def ask_features(df, subcat_median=None, cat_median=None):
    """Represent campaign ask size absolutely and relative to similar projects."""
    f = pd.DataFrame(index=df.index)
    lg = np.log1p(df["goal"])
    f["log_goal"] = lg

    if subcat_median is None:
        subcat_median = lg.groupby(df["subcategory"]).median()
    if cat_median is None:

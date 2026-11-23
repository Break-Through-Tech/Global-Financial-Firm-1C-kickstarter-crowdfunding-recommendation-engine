import math

import numpy as np
import pandas as pd
from catboost import Pool
from langchain.chat_models import init_chat_model

from models.train_and_predict import CATEGORICAL_FEATURES, RAW_COLUMNS

DEFAULT_GEMINI_MODEL = "gemini-flash-lite-latest"

FEATURE_LABELS = {
    "log_goal": "Funding goal",
    "duration": "Campaign length",
    "n_tiers": "Number of reward tiers",
    "log_min_tier": "Cheapest reward",
    "log_max_tier": "Most expensive reward",
    "log_median_tier": "Typical (median) reward",
    "tier_spread": "Range between cheapest and priciest reward",
    "entry_vs_goal": "Cheapest reward compared to goal",
    "launch_month": "Launch month",
    "name_len": "Campaign name length",
    "name_words": "Words in campaign name",
    "name_allcaps": "Name written in ALL CAPS",
    "name_excl": "Exclamation marks in name",
    "has_location": "Location provided",
    "category": "Category",
    "subcategory": "Subcategory",
}

SYSTEM_PROMPT = """You are a friendly Kickstarter campaign coach. A creator just ran their campaign idea
through a CatBoost model trained on about 38,000 U.S. Kickstarter campaigns from 2009 to 2012.
Explain what the prediction means in plain language, using only the numbers you are given.

A positive push moved the prediction toward success and a negative push moved it toward failure.
Overall importance is a separate ranking of how much the model relies on each feature across all
campaigns. Never mix a feature's push with another feature's importance, and quote numbers exactly.

Write in Markdown with these parts:
**Bottom line** - one or two sentences on the estimate.
**What is working for you** - the strongest positive drivers.
**What is holding it back** - the strongest negative drivers.
**Things to try** - two or three specific, realistic changes tied to the drivers.

Keep it under 250 words. The model has around 70% accuracy, so never promise an outcome,
and mention once that the data is historical."""


def source_feature(column_name):
    if column_name.startswith("launch_month"):
        return "launch_month"
    for categorical_feature in CATEGORICAL_FEATURES:
        if column_name.startswith(f"{categorical_feature}_"):
            return categorical_feature
    return column_name


def is_missing(value):
    return value is None or (isinstance(value, float) and math.isnan(value))


def describe_value(feature, engineered_row, campaign):
    value = engineered_row.get(feature)
    if feature != "launch_month" and is_missing(value):
        return "not provided"

    match feature:
        case "log_goal":
            return f"${campaign['goal']:,.0f}"
        case "duration":
            return f"{campaign['duration']:g} days"
        case "n_tiers":
            return f"{value:.0f} tiers"
        case "log_min_tier" | "log_max_tier" | "log_median_tier":
            return f"${np.expm1(value):,.0f}"
        case "tier_spread":
            return f"priciest tier is about {np.exp(value):,.0f}x the cheapest"
        case "entry_vs_goal":
            return f"cheapest reward is {np.exp(value):.2%} of the goal"
        case "launch_month":
            return pd.Timestamp(campaign["launch"]).strftime("%B")
        case "name_len":
            return f"{value:.0f} characters"
        case "name_words":
            return f"{value:.0f} words"
        case "name_allcaps" | "has_location":
            return "yes" if value else "no"
        case "name_excl":
            return f"{value:.0f}"
        case _:
            return str(value)


def explain_prediction(pipeline, campaign):
    campaign_frame = pd.DataFrame([campaign])[RAW_COLUMNS]
    engineered = pipeline.named_steps["features"].transform(campaign_frame)
    preprocess = pipeline.named_steps["preprocess"]
    catboost_model = pipeline.named_steps["model"]

    model_input = preprocess.transform(engineered)
    source_features = [source_feature(name) for name in preprocess.get_feature_names_out()]

    shap_values = catboost_model.get_feature_importance(Pool(model_input), type="ShapValues")[0][:-1]
    push_by_feature = pd.Series(shap_values, index=source_features).groupby(level=0).sum()
    overall_importance = (pd.Series(catboost_model.get_feature_importance(), index=source_features)
                          .groupby(level=0).sum())

    engineered_row = engineered.iloc[0].to_dict()
    drivers = pd.DataFrame({
        "feature": push_by_feature.index,
        "label": [FEATURE_LABELS.get(feature, feature) for feature in push_by_feature.index],
        "value": [describe_value(feature, engineered_row, campaign) for feature in push_by_feature.index],
        "push": push_by_feature.values,
        "overall_importance": overall_importance.reindex(push_by_feature.index).values,
    })
    return drivers.reindex(drivers["push"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def build_llm(api_key, model_name=DEFAULT_GEMINI_MODEL):
    return init_chat_model(model_name, model_provider="google_genai", api_key=api_key,
                           temperature=0.2, max_retries=3)


def describe_campaign(campaign):
    readable_fields = {
        "name": "Name",
        "category": "Category",
        "subcategory": "Subcategory",
        "location": "Location",
        "goal": "Funding goal ($)",
        "duration": "Duration (days)",
        "reward levels": "Reward levels",
        "launch": "Launch date",
    }
    return "\n".join(f"- {label}: {campaign[field]}" for field, label in readable_fields.items()
                     if not is_missing(campaign.get(field)) and campaign.get(field) != "")


def describe_drivers(drivers, how_many=10):
    return "\n".join(f"- {row.label} = {row.value}: push {row.push:+.3f}"
                     for row in drivers.head(how_many).itertuples())


def describe_overall_importance(drivers, how_many=5):
    most_relied_on = drivers.sort_values("overall_importance", ascending=False).head(how_many)
    return "\n".join(f"{rank}. {row.label}: {row.overall_importance:.1f}%"
                     for rank, row in enumerate(most_relied_on.itertuples(), start=1))


def interpretation_messages(campaign, probability, threshold, drivers=None, scope_note=None):
    outcome = "likely to succeed" if probability >= threshold else "likely to fall short"
    user_prompt = (f"Campaign inputs:\n{describe_campaign(campaign)}\n\n"
                   f"Predicted probability of reaching the goal: {probability:.1%} "
                   f"(threshold {threshold:.0%}, so the model says it is {outcome}).\n\n")

    if drivers is not None:
        user_prompt += (f"Biggest drivers for this campaign (SHAP values in log-odds):\n{describe_drivers(drivers)}\n\n"
                        f"Features the model relies on most across all campaigns:\n"
                        f"{describe_overall_importance(drivers)}")
    else:
        user_prompt += "Per-feature drivers are not available for this model, so focus on the inputs."

    if scope_note:
        user_prompt += f"\n\nNote: {scope_note}"

    return [("system", SYSTEM_PROMPT), ("human", user_prompt)]


def stream_interpretation(llm, messages):
    for chunk in llm.stream(messages):
        yield str(chunk.text)


import os
import time
from datetime import date
from pathlib import Path

import streamlit as st
from dotenv import dotenv_values
from models.train_and_predict import load_model, predict_campaign, save_model, train_online_model
from utils.campaign_interpretation import (DEFAULT_GEMINI_MODEL, build_llm, explain_prediction,
                                           interpretation_messages, stream_interpretation)
from utils.historical_prediction import (MODEL_HASHES, historical_campaign_frame,
                                         load_historical_predictor, predict_historical)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Kickstarter Campaign Advisor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

ONLINE_MODEL_PATH = Path(__file__).parent / "models" / "trained_models" / "online_us_catboostclassifier.pkl"
SUCCESS_THRESHOLD = 0.5
ENV_FILE = Path(__file__).parent / ".env"


@st.cache_resource(show_spinner="Loading the CatBoost model...")
def load_online_predictor():
    try:
        return load_model(ONLINE_MODEL_PATH)
    except FileNotFoundError:
        pass
    except Exception as error:
        st.warning(f"Saved model could not be loaded ({type(error).__name__}), retraining it now.")

    pipeline = train_online_model(scope="us")
    save_model(pipeline, ONLINE_MODEL_PATH)
    return pipeline


def read_setting(name, default=None):
    return dotenv_values(ENV_FILE).get(name) or os.environ.get(name) or default


@st.cache_resource
def load_llm(api_key, model_name):
    return build_llm(api_key, model_name)


def show_ai_interpretation(campaign, probability, threshold, drivers=None, scope_note=None):
    st.divider()
    st.markdown("### AI Interpretation")

    api_key = read_setting("API_KEY")
    if not api_key:
        st.info("Add your Gemini key as API_KEY in the .env file to see an AI interpretation here.")
        return

    model_name = read_setting("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    messages = interpretation_messages(campaign, probability, threshold, drivers, scope_note)

    try:
        write_interpretation(load_llm(api_key, model_name), messages)
        st.caption(f"Written by {model_name} from the prediction and feature importance above. "
                   "Treat it as a starting point, not advice.")
    except Exception as error:
        st.warning(f"The AI interpretation is unavailable right now ({type(error).__name__}: {error}).")


def write_interpretation(llm, messages, attempts=3):
    answer_box = st.empty()
    for attempt in range(1, attempts + 1):
        try:
            with answer_box.container(border=True):
                st.write_stream(stream_interpretation(llm, messages))
            return
        except Exception:
            answer_box.empty()
            if attempt == attempts:
                raise
            time.sleep(2 * attempt)


@st.cache_resource
def load_scope_predictor(scope, artifact_hash):
    path = Path(__file__).parent / 'models' / f'{scope}_historical_v1.joblib'
    return load_historical_predictor(path, scope)


# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------

st.markdown("""
<style>
    .stApp,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p,
    .stApp input, .stApp textarea, .stApp button, .stApp label,
    .stApp [data-testid="stMarkdownContainer"],
    .stApp [data-testid="stMetric"],
    .stApp [data-testid="stMetricValue"] > div {
        font-family: Arial, Helvetica, sans-serif;
    }

    .stApp {
        background-color: var(--background-color, #111827);
        color: var(--text-color, #F3F4F6);
    }

    [data-testid="stSidebar"] {
        background-color: var(--secondary-background-color, #1F2937);
    }

    .main-title {
        font-size: 2rem;
        font-weight: 600;
        color: var(--text-color, #F3F4F6);
        letter-spacing: normal;
        line-height: 1.25;
        margin-bottom: 5px;
    }

    .subtitle {
        color: var(--text-color, #F3F4F6);
        opacity: 0.8;
        font-size: 1rem;
        line-height: 1.5;
        margin-bottom: 25px;
    }

    .section-title {
        color: var(--text-color, #F3F4F6);
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 12px;
    }

    .info-card {
        background-color: var(--secondary-background-color, #1F2937);
        padding: 22px;
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.3);
        margin-bottom: 15px;
    }

    .small-label {
        color: var(--text-color, #F3F4F6);
        opacity: 0.8;
        font-size: 0.85rem;
    }

    div.stButton > button[kind="primary"],
    div.stFormSubmitButton > button {
        background-color: #2457D6;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px;
        font-weight: 600;
    }

    div.stButton > button[kind="primary"]:hover,
    div.stFormSubmitButton > button:hover {
        background-color: #1743B5;
        color: white;
    }

    [data-testid="stForm"] {
        background-color: var(--secondary-background-color, #1F2937);
        padding: 24px;
        border: 1px solid rgba(148, 163, 184, 0.3);
        border-radius: 12px;
    }

    h1, h2, h3 {
        color: var(--text-color, #F3F4F6);
    }

    .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
        font-weight: 600;
        letter-spacing: normal;
    }

    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown("# Kickstarter")
    st.markdown("### Campaign Advisor")

    st.divider()

    mode = st.radio("Prediction mode", ["U.S. prediction", "International (experimental)", "Mixed (experimental)"])
    st.caption("International and mixed estimates use historical listed locations, not registration country.")

    st.divider()

    st.markdown("**Project information**")
    st.caption("Team 1C")
    st.caption("Break Through Tech AI Studio")
    st.caption("Fall 2026")

    st.divider()

    st.caption(
        "Helping creators make more informed "
        "campaign planning decisions."
    )


scope = {"U.S. prediction": None, "International (experimental)": "international",
         "Mixed (experimental)": "mixed"}[mode]

# --------------------------------------------------
# MAIN HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">Kickstarter Campaign Advisor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Understand your campaign characteristics using '
    'historical Kickstarter data and machine learning.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# --------------------------------------------------
# CAMPAIGN INPUT SECTION
# --------------------------------------------------

st.markdown(
    '<div class="section-title">Campaign Information</div>',
    unsafe_allow_html=True
)

st.write(
    "Enter your proposed campaign details to prepare "
    "a success prediction."
)

if scope is None:
    st.caption(
        "Use exact training category names, such as Music / Rock or Art / Painting. "
        "Unseen categories fall back to an 'other' bucket. Location accepts ST or City, ST. "
        "Reward levels are dollar amounts separated by commas, such as $10,$25,$100."
    )
    st.info(
        "CatBoost model on 2009–2012 U.S. campaigns. On a held-out 20% of campaigns it scored "
        "about 70.4% accuracy and 0.766 ROC-AUC. Probabilities are uncalibrated and the "
        "AI interpretation is generated from the model's own feature importance."
    )
else:
    st.warning(
        "Experimental historical estimate for demonstration. These models use 2009–2012 data "
        "and have not been validated on present-day campaigns. Funding amounts are excluded "
        "because source currency units are unverified. Sparse regions and unfamiliar categories "
        "can make estimates unreliable."
    )
    st.caption(
        "Enter duration, category, subcategory and listed location. Use a full country name, "
        "such as Toronto, Canada. Mixed mode also accepts U.S. state/DC codes. "
        "No campaign funding amount is used by either experimental model."
    )

with st.form("campaign_form"):

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Basic Details")

        name = st.text_input(
            "Campaign Name" if scope is None else "Campaign Name (optional, not used by model)",
            placeholder="e.g. Independent Music Album"
        )

        category = st.text_input(
            "Category",
            placeholder="e.g. Music"
        )

        subcategory = st.text_input(
            "Subcategory",
            placeholder="e.g. Rock"
        )

        location = st.text_input(
            "Location",
            placeholder="e.g. Austin, TX" if scope is None else "e.g. Berlin, Germany"
        )

    with col2:
        st.markdown("#### Campaign Goals")

        goal = None
        if scope is None:
            goal = st.number_input(
                "Funding Goal ($)",
                min_value=1.0,
                value=5000.0,
                step=500.0
            )

        duration = st.number_input(
            "Campaign Duration (days)",
            min_value=1.0,
            value=30.0,
            step=1.0
        )

        reward_levels = ""
        launch_date = date.today()
        if scope is None:
            reward_levels = st.text_input(
                "Reward Levels",
                placeholder="e.g. $10,$25,$100,$500"
            )
            launch_date = st.date_input("Launch Date", value=date.today())

    st.markdown("---")

    submitted = st.form_submit_button(
        "Analyze campaign",
        type="primary",
        use_container_width=True
    )


# --------------------------------------------------
# FORM SUBMISSION AND MODEL PREDICTION
# --------------------------------------------------

if submitted:

    if not all([
        category.strip(),
        subcategory.strip(),
        location.strip()
    ]):

        st.error(
            "Please complete all campaign information "
            "before continuing."
        )

    else:

        st.success("Campaign information successfully captured.")

        st.markdown("### Campaign Summary")

        metric1, metric2 = st.columns(2)

        with metric1:
            if scope is None:
                st.metric("Funding Goal", f"${goal:,.2f}")
            else:
                st.metric("Estimate scope", scope.capitalize())

        with metric2:
            st.metric("Campaign Duration", f"{duration:g} days")

        st.markdown("#### Campaign Details")

        summary1, = st.columns(1)

        with summary1:
            st.write("**Campaign:**", name)
            st.write("**Category:**", category)
            st.write("**Subcategory:**", subcategory)
            st.write("**Location:**", location)

        # The selected validator keeps only that model's approved inputs.
        campaign = {
            "name": name.strip(),
            "goal": goal,
            "duration": duration,
            "category": category.strip(),
            "subcategory": subcategory.strip(),
            "location": location.strip(),
            "reward levels": reward_levels.strip(),
            "launch": launch_date.isoformat() if scope is None else None,
        }
        drivers = None
        scope_note = None

        st.divider()
        st.markdown("### Campaign Success Prediction")

        try:
            with st.spinner("Analyzing your campaign..."):

                if scope is None:
                    online_predictor = load_online_predictor()
                    probability = float(predict_campaign(online_predictor, campaign)[0])
                    result = {
                        "success_probability": probability,
                        "classification": "successful" if probability >= SUCCESS_THRESHOLD else "failed",
                        "threshold": SUCCESS_THRESHOLD,
                    }
                    drivers = explain_prediction(online_predictor, campaign)
                else:
                    historical_campaign_frame(campaign, scope)
                    predictor = load_scope_predictor(scope, MODEL_HASHES[scope])
                    result = predict_historical(predictor, campaign, scope)

            probability = result["success_probability"]
            classification = result["classification"]

            prediction_col1, prediction_col2 = st.columns(2)

            with prediction_col1:
                st.metric(
                    "Probability of Success",
                    f"{probability:.2%}"
                )

            with prediction_col2:
                st.metric(
                    "Predicted Outcome",
                    classification.capitalize()
                )

            st.markdown("#### Success Probability")
            st.progress(probability)

            if scope is not None:
                st.write("Experimental classification at the displayed threshold:", classification)
            elif classification == "successful":
                st.success(
                    "The model predicts that this campaign "
                    "is likely to succeed."
                )
            else:
                st.warning(
                    "The model predicts that this campaign "
                    "is likely to fail."
                )

            st.caption(
                f"Classification threshold: "
                f"{result['threshold']:.0%}"
            )

            st.info("This is a model estimate from historical data, not a guarantee of campaign outcome.")
            if scope is not None:
                metadata = predictor[1]
                validation = metadata['validation_metrics']
                st.caption(
                    f"Experimental {scope} baseline: {metadata['training_rows']:,} training campaigns. "
                    f"Training-only five-fold CV accuracy {validation['accuracy']:.2%}, "
                    f"ROC-AUC {validation['roc_auc']:.4f}, "
                    f"failure recall {validation['failure_recall']:.2%}. "
                    f"Always-majority accuracy: {metadata['majority_class_accuracy']:.2%}. "
                    "Different cohort scores are not a head-to-head comparison."
                )
                for message in result['warnings']:
                    st.warning(message)
                scope_note = (f"This estimate comes from an experimental {scope} baseline model "
                              f"with {validation['accuracy']:.0%} cross-validated accuracy, "
                              "not the U.S. CatBoost model, and it ignores the funding goal.")

            if drivers is not None:
                st.markdown("#### What Moved This Prediction")
                top_drivers = drivers.head(8)
                st.bar_chart(top_drivers, x="label", y="push", horizontal=True, sort=False,
                             x_label="", y_label="Push toward success (+) or failure (-)")
                st.dataframe(
                    top_drivers[["label", "value", "push", "overall_importance"]].rename(columns={
                        "label": "Feature", "value": "Your campaign", "push": "Push",
                        "overall_importance": "Overall importance (%)"}),
                    hide_index=True, width="stretch")

            show_ai_interpretation(campaign, probability, result["threshold"], drivers, scope_note)

        except FileNotFoundError:
            st.error(
                "The trained model or its metadata file could "
                "not be found. Check the models folder."
            )

        except ValueError as error:
            st.error(f"Campaign or model validation failed: {error}")

        except Exception:
            st.error("Prediction unavailable. Ask the team to check the model and runtime.")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

footer1, footer2 = st.columns(2)

with footer1:
    st.caption("Kickstarter Campaign Advisor | Team 1C")

with footer2:
    st.caption(
        "Break Through Tech AI Studio | Fall 2026"
    )

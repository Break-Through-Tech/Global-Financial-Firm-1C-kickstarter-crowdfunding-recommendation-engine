
import streamlit as st
from pathlib import Path
from utils.us_prediction import load_primary_predictor, campaign_frame, PRIMARY_SHA256
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

MODEL_PATH = Path(__file__).parent / "models" / "us_primary_model_v1.joblib"

@st.cache_resource
def load_predictor(model_hash):
    return load_primary_predictor(MODEL_PATH)


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
        "Unseen category and subcategory values use training-fitted fallbacks; "
        "category pairs are not validated. Location accepts ST or City, ST."
    )
    st.info(
        "Development validation: accuracy 66.93%, ROC-AUC 0.7239, failure recall 55.72%. "
        "Historical 2009–2012 data; probabilities are uncalibrated. Location is a state/DC proxy. "
        "Launch snapshots, duration meaning and goal currency are unverified. The reserved set "
        "has historical exposure; these are not independent test results. "
        "Campaign-specific benchmarks, scenarios and LLM advice are not available in this dashboard."
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
            "Campaign Name (optional, not used by model)",
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
            "goal": goal,
            "duration": duration,
            "category": category.strip(),
            "subcategory": subcategory.strip(),
            "location": location.strip()
        }

        st.divider()
        st.markdown("### Campaign Success Prediction")

        try:
            with st.spinner("Analyzing your campaign..."):

                if scope is None:
                    campaign_frame(campaign)
                    predictor = load_predictor(PRIMARY_SHA256)
                    result = predictor.predict(campaign)
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

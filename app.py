
import streamlit as st
from pathlib import Path
from utils.us_prediction import load_primary_predictor, campaign_frame, PRIMARY_SHA256

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


# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------

st.markdown("""
<style>
    .stApp {
        background-color: #F5F7FB;
    }

    [data-testid="stSidebar"] {
        background-color: #101C35;
    }

    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }

    .main-title {
        font-size: 2.3rem;
        font-weight: 750;
        color: #14213D;
        letter-spacing: -1px;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #64748B;
        font-size: 1rem;
        margin-bottom: 25px;
    }

    .section-title {
        color: #14213D;
        font-size: 1.35rem;
        font-weight: 650;
        margin-bottom: 12px;
    }

    .info-card {
        background-color: white;
        padding: 22px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        margin-bottom: 15px;
    }

    .small-label {
        color: #64748B;
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
        background-color: white;
        padding: 24px;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
    }

    h1, h2, h3 {
        color: #14213D;
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
    st.markdown("# 🚀 Kickstarter")
    st.markdown("### Campaign Advisor")

    st.divider()

    st.markdown("**AVAILABLE MODE**")
    st.markdown("U.S. campaign predictor: HGB v1")
    st.caption("International and mixed modes are unavailable pending validation.")

    st.divider()

    st.markdown("**PROJECT INFORMATION**")
    st.caption("Team 1C")
    st.caption("Break Through Tech AI Studio")
    st.caption("Fall 2026")

    st.divider()

    st.caption(
        "Helping creators make more informed "
        "campaign planning decisions."
    )


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
    "Historical comparisons, scenarios and LLM advice are not available in this dashboard."
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
            placeholder="e.g. Austin, TX"
        )

    with col2:
        st.markdown("#### Campaign Goals")

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
        "Analyze My Campaign →",
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
            st.metric("Funding Goal", f"${goal:,.2f}")

        with metric2:
            st.metric("Campaign Duration", f"{duration:g} days")

        st.markdown("#### Campaign Details")

        summary1, = st.columns(1)

        with summary1:
            st.write("**Campaign:**", name)
            st.write("**Category:**", category)
            st.write("**Subcategory:**", subcategory)
            st.write("**Location:**", location)

        # Send only the five features required by the model.
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

                campaign_frame(campaign)
                predictor = load_predictor(PRIMARY_SHA256)
                result = predictor.predict(campaign)

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

            if classification == "successful":
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

            st.info(
                "This prediction is generated using the team's "
                "trained machine learning model and its five "
                "approved input features. It is an estimate based "
                "on historical campaign data, not a guarantee "
                "of the actual campaign outcome."
            )

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

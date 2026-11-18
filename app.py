
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

    mode = st.radio("View", ["U.S. prediction", "International research", "Mixed cohort research"])
    st.caption("U.S. predictions use HGB v1. Research views show recorded cohort evidence.")

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

if mode != "U.S. prediction":
    st.subheader(mode)
    st.info(
        "Historical research summary. This view does not generate a campaign prediction. "
        "Each cohort has different coverage and evaluation; scores are not a head-to-head comparison."
    )

    if mode == "International research":
        st.write(
            "This cohort includes explicit named non-U.S. locations, such as Berlin, Germany "
            "or Toronto, Canada. Unknown locations and U.S. territories are excluded. "
            "Listed location does not establish campaign registration country."
        )
        st.metric("Official training campaigns", "1,427")
        st.markdown("#### Recorded training-validation results")
        st.caption(
            "Five-fold stratified training CV with fold-fitted conservative features and one-hot encoding. "
            "These are saved baseline results, not a newly evaluated or deployed international model."
        )
        st.table([
            {"Model": "Random Forest", "Accuracy": "65.73%", "ROC-AUC": "0.6685"},
            {"Model": "Decision Tree", "Accuracy": "65.03%", "ROC-AUC": "0.6584"},
            {"Model": "Logistic Regression", "Accuracy": "64.19%", "ROC-AUC": "0.6573"},
            {"Model": "KNN", "Accuracy": "65.03%", "ROC-AUC": "0.6510"},
            {"Model": "HistGradientBoosting", "Accuracy": "61.60%", "ROC-AUC": "0.6405"},
            {"Model": "Dummy baseline", "Accuracy": "60.97%", "ROC-AUC": "0.5000"},
        ])
        st.warning(
            "Coverage is sparse: 113 of 125 training regions have fewer than 25 campaigns. "
            "Currency comparability remains unresolved, including within-country fallbacks. "
            "The reserved cohort has historical exposure. No independent test result is shown."
        )
        st.markdown(
            "[Baseline notebook](https://github.com/Break-Through-Tech/"
            "Global-Financial-Firm-1C-kickstarter-crowdfunding-recommendation-engine/"
            "blob/main/notebooks/international_only/05_modeling.ipynb) · "
            "[Cohort documentation](https://github.com/Break-Through-Tech/"
            "Global-Financial-Firm-1C-kickstarter-crowdfunding-recommendation-engine/"
            "blob/main/docs/international_only_workflow.md)"
        )
    else:
        st.write(
            "The mixed cohort is the original combined dataset. It includes the U.S. subset, "
            "named non-U.S. locations, and other or unresolved locations. "
            "It is not a separate population from the U.S. or international cohorts."
        )
        st.metric("Official training campaigns", "29,288")
        st.table([
            {"Training subset": "U.S. state/DC proxy", "Campaigns": 26945},
            {"Training subset": "Explicit named non-U.S.", "Campaigns": 1427},
            {"Training subset": "Other or unresolved locations", "Campaigns": 916},
        ])
        st.warning(
            "There is no approved saved mixed-cohort predictor. Global funding-goal currency "
            "units remain unverified. Earlier combined model scores used different feature "
            "policies and preprocessing, so they are not presented as deployable performance."
        )
        st.write(
            "Zasha's online workflow is a separate CatBoost research pipeline. It requires "
            "title, rewards and launch information in addition to the five U.S. inputs, "
            "and does not establish a validated mixed-cohort model."
        )
        st.markdown(
            "[Cohort and pipeline review](https://github.com/Break-Through-Tech/"
            "Global-Financial-Firm-1C-kickstarter-crowdfunding-recommendation-engine/"
            "blob/main/docs/launch_pipeline_review.md)"
        )

    st.caption(
        "Enabling predictions requires a separately saved and evaluated pipeline, "
        "verified funding units, supported input rules, and model integrity metadata. "
        "No model is trained or loaded by these research views."
    )
    st.caption("Kickstarter Campaign Advisor | Team 1C")
    st.stop()


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
    "Campaign-specific benchmarks, scenarios and LLM advice are not available in this dashboard."
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

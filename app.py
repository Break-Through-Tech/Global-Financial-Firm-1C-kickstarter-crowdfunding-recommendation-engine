
import streamlit as st

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

    st.markdown("**NAVIGATION**")
    st.markdown("🏠 Campaign Predictor")
    st.markdown("📊 Campaign Insights")
    st.markdown("ℹ️ About the Project")

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

with st.form("campaign_form"):

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Basic Details")

        name = st.text_input(
            "Campaign Name",
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

        reward_levels = st.text_input(
            "Reward Levels",
            placeholder="$10,$25,$50,$100"
        )

        launch = st.date_input(
            "Expected Launch Date"
        )

    st.markdown("---")

    submitted = st.form_submit_button(
        "Analyze My Campaign →",
        type="primary",
        use_container_width=True
    )


# --------------------------------------------------
# FORM SUBMISSION
# --------------------------------------------------

if submitted:

    if not all([
        name.strip(),
        category.strip(),
        subcategory.strip(),
        location.strip(),
        reward_levels.strip()
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
            st.metric(
                "Funding Goal",
                f"${goal:,.2f}"
            )

        with metric2:
            st.metric(
                "Campaign Duration",
                f"{duration:.0f} days"
            )

        st.markdown("#### Campaign Details")

        summary1, summary2 = st.columns(2)

        with summary1:
            st.write("**Campaign:**", name)
            st.write("**Category:**", category)
            st.write("**Subcategory:**", subcategory)
            st.write("**Location:**", location)

        with summary2:
            st.write("**Reward Levels:**", reward_levels)
            st.write("**Launch Date:**", launch)

        st.info(
            "The prediction engine will be connected after "
            "the team's approved model and feature policy "
            "are finalized. No prediction has been generated yet."
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

footer1, footer2 = st.columns(2)

with footer1:
    st.caption(
        "Kickstarter Campaign Advisor | Team 1C"
    )

with footer2:
    st.caption(
        "Break Through Tech AI Studio | Fall 2026"
    )

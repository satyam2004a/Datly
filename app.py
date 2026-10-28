import hashlib
import io

import streamlit as st
import pandas as pd

from src.data_profiler import profile_dataset

from src.anomaly_detection import (
    detect_anomalies,
    anomaly_summary,
    combined_anomaly_report,
    calculate_anomaly_score
)

from src.data_cleaning import clean_dataset

from src.quality_rules import run_quality_rules

from src.quality_score import (
    calculate_completeness_score,
    calculate_validity_score,
    calculate_uniqueness_score,
    calculate_anomaly_quality_score,
    calculate_overall_quality_score,
    get_quality_status
)

from src.recommendations import generate_recommendations

from src.quality_report import (
    generate_quality_report,
    generate_quality_report_html
)


# =========================================================
# PERFORMANCE SETTINGS
# =========================================================

# Never render huge DataFrames directly in Streamlit.
# The complete data remains available for download.
MAX_DISPLAY_ROWS = 500

# Maximum number of anomalies displayed in the dashboard.
MAX_ANOMALY_DISPLAY_ROWS = 500


# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Datly | Smart Data Quality",
    page_icon="🛡️",
    layout="wide"
)

# Datly uses Streamlit's native HTML renderer for custom UI.
if not hasattr(st, "html"):
    st.error("Datly requires Streamlit 1.37.0 or newer. Please upgrade with: pip install -U streamlit")
    st.stop()


# =========================================================
# THEME SELECTION
# =========================================================



# Theme selector
if "theme" not in st.session_state:
    st.session_state["theme"] = "Light"

theme = st.sidebar.selectbox(
    "🎨 Theme",
    ["Light", "Dark"],
    index=["Light", "Dark"].index(
        st.session_state["theme"]
    )
)

st.session_state["theme"] = theme

# Your CSS

# =========================================================
# LIGHT THEME
# =========================================================

if theme == "Light":

    st.markdown(
        """
        <style>
        :root {
            --datly-bg: #ffffff;
            --datly-card: #ffffff;
            --datly-heading: #0f172a;
            --datly-muted: #64748b;
            --datly-border: #e2e8f0;
            --datly-shadow: rgba(15, 23, 42, 0.06);
        }

        .stApp {
            background-color: #ffffff;
            color: #111827;
        }

        .stApp p,
        .stApp span,
        .stApp label {
            color: #111827;
        }

        .stApp h1,
        .stApp h2,
        .stApp h3,
        .stApp h4,
        .stApp h5,
        .stApp h6 {
            color: #0f172a !important;
        }

        section[data-testid="stSidebar"] {
            background-color: #f8fafc;
            border-right: 1px solid #e2e8f0;
        }

        section[data-testid="stSidebar"] * {
            color: #111827 !important;
        }

        div[data-testid="stMetric"] {
            background-color: #f1f5f9;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 12px;
        }

        div[data-testid="stMetric"] label {
            color: #64748b !important;
        }

        div[data-testid="stMetric"] div {
            color: #111827 !important;
        }

        div[data-testid="stDataFrame"] {
            background-color: #f8fafc !important;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
        }

        div[data-testid="stButton"] button,
        div[data-testid="stDownloadButton"] button {
            background-color: #ffffff;
            color: #111827 !important;
            border: 1px solid #d1d5db;
        }

        div[data-testid="stButton"] button:hover,
        div[data-testid="stDownloadButton"] button:hover {
            background-color: #f1f5f9;
            border-color: #94a3b8;
        }

        div[data-testid="stButton"] button p,
        div[data-testid="stDownloadButton"] button p {
            color: #111827 !important;
        }

        section[data-testid="stFileUploader"] {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
        }

        section[data-testid="stFileUploader"] * {
            color: #111827 !important;
        }

        input,
        textarea,
        select {
            background-color: #ffffff !important;
            color: #111827 !important;
            border-color: #d1d5db !important;
        }

        div[data-testid="stAlert"] {
            border-radius: 10px;
        }

        </style>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# DARK THEME
# =========================================================

else:

    st.markdown(
        """
        <style>
        :root {
            --datly-bg: #0F1115;
            --datly-card: #181B21;
            --datly-heading: #F3F4F6;
            --datly-muted: #9CA3AF;
            --datly-border: #30343D;
            --datly-shadow: rgba(0, 0, 0, 0.25);
        }

        .stApp {
            background-color: #0F1115;
            color: #E5E7EB;
        }

        .stApp p,
        .stApp span,
        .stApp label {
            color: #E5E7EB;
        }

        .stApp h1,
        .stApp h2,
        .stApp h3,
        .stApp h4,
        .stApp h5,
        .stApp h6 {
            color: #F3F4F6 !important;
        }

        section[data-testid="stSidebar"] {
            background-color: #15181E;
            border-right: 1px solid #2A2E36;
        }

        section[data-testid="stSidebar"] * {
            color: #E5E7EB !important;
        }

        div[data-testid="stMetric"] {
            background-color: #181B21;
            border: 1px solid #2D323B;
            border-radius: 12px;
            padding: 14px;
        }

        div[data-testid="stMetric"] label {
            color: #9CA3AF !important;
        }

        div[data-testid="stMetric"] div {
            color: #F3F4F6 !important;
        }

        div[data-testid="stDataFrame"] {
            background-color: #1E2229 !important;
            border: 1px solid #30343D;
            border-radius: 10px;
        }

        div[data-testid="stButton"] button,
        div[data-testid="stDownloadButton"] button {
            background-color: #20242B;
            color: #E5E7EB !important;
            border: 1px solid #3A404A;
        }

        div[data-testid="stButton"] button:hover,
        div[data-testid="stDownloadButton"] button:hover {
            background-color: #292E37;
            border-color: #555C68;
        }

        div[data-testid="stButton"] button p,
        div[data-testid="stDownloadButton"] button p {
            color: #E5E7EB !important;
        }

        section[data-testid="stFileUploader"] {
            background-color: #181B21 !important;
            border: 1px solid #30343D !important;
            border-radius: 12px;
            padding: 12px;
        }

        section[data-testid="stFileUploader"] > div {
            background-color: #2B3038 !important;
        }

        section[data-testid="stFileUploader"] section {
            background-color: #343A43 !important;
            border: 1px dashed #5B6470 !important;
            border-radius: 12px;
            padding: 14px;
        }

        section[data-testid="stFileUploader"] > div {
            background-color: #2B3038 !important;
        }
        section[data-testid="stFileUploader"] * {
            color: #111827 !important;
            -webkit-text-fill-color: #111827 !important;
        }

        input,
        textarea,
        select {
            background-color: #181B21 !important;
            color: #E5E7EB !important;
            border-color: #30343D !important;
        }

        div[data-testid="stAlert"] {
            border-radius: 10px;
        }

        </style>
        """,
        unsafe_allow_html=True
    )

# =========================================================
# CUSTOM CSS
# =========================================================

st.html(
    """
    <style>

    /* =====================================================
       MAIN APPLICATION
    ===================================================== */

    .main {
        background-color: var(--datly-bg);
    }


    /* =====================================================
       DATLY HERO
    ===================================================== */

    .hero {
        padding: 2.4rem 2.6rem;
        border-radius: 20px;
        background: linear-gradient(
            135deg,
            #0f172a 0%,
            #1e293b 55%,
            #334155 100%
        );
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 25px rgba(15, 23, 42, 0.12);
    }

    .hero-top {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 0.5rem;
    }

    .hero-icon {
        font-size: 2.6rem;
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -1px;
    }

    .hero-tagline {
        font-size: 1.35rem;
        font-weight: 600;
        color: #e2e8f0;
        margin: 0.2rem 0 0.6rem 0;
    }

    .hero-description {
        font-size: 1rem;
        color: #ffffff !important;
        margin: 0;
    }

    .hero-badge {
        display: inline-block;
        margin-top: 1rem;
        padding: 0.35rem 0.8rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.10);
        border: 1px solid rgba(255, 255, 255, 0.15);
        color: #e2e8f0;
        font-size: 0.82rem;
        font-weight: 600;
    }


    /* =====================================================
       SECTION HEADINGS
    ===================================================== */

    .section-title {
        font-size: 1.55rem;
        font-weight: 750;
        color: var(--datly-heading);
        margin-top: 1.5rem;
        margin-bottom: 0.2rem;
    }

    .section-description {
        color: var(--datly-muted);
        font-size: 0.95rem;
        margin-bottom: 1rem;
    }

    .section-divider {
        margin: 1.5rem 0 0.5rem 0;
        border-top: 1px solid var(--datly-border);
    }


    /* =====================================================
       METRIC CARDS
    ===================================================== */

    .metric-card {
        background: var(--datly-card);
        padding: 1.25rem 1rem;
        min-height: 105px;
        border-radius: 16px;
        border: 1px solid var(--datly-border);
        box-shadow: 0 4px 12px var(--datly-shadow);
        text-align: center;
        transition: 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 7px 18px var(--datly-shadow);
    }

    .metric-card h3 {
        margin: 0;
        color: var(--datly-muted);
        font-size: 0.82rem;
        font-weight: 650;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .metric-card h2 {
        margin: 0.45rem 0 0 0;
        font-size: 1.75rem;
        font-weight: 800;
        color: var(--datly-heading);
    }


    /* =====================================================
       QUALITY SCORE CARD
    ===================================================== */

    .quality-score-card {
        background: var(--datly-card);
        padding: 1.5rem;
        border-radius: 18px;
        border: 1px solid var(--datly-border);
        box-shadow: 0 5px 15px var(--datly-shadow);
        margin-bottom: 1rem;
    }

    .quality-score-label {
        color: var(--datly-muted);
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .quality-score-value {
        color: var(--datly-heading);
        font-size: 2.6rem;
        font-weight: 850;
        margin: 0.2rem 0 0.4rem 0;
    }

    .quality-score-status {
        color: var(--datly-muted);
        font-size: 0.95rem;
    }


    /* =====================================================
       BEFORE / AFTER CARD
    ===================================================== */

    .comparison-card {
        background: var(--datly-card);
        padding: 1.3rem;
        border-radius: 16px;
        border: 1px solid var(--datly-border);
        box-shadow: 0 4px 12px var(--datly-shadow);
    }

    .comparison-label {
        color: var(--datly-muted);
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .comparison-value {
        color: var(--datly-heading);
        font-size: 1.8rem;
        font-weight: 800;
        margin-top: 0.3rem;
    }


    /* =====================================================
       BUTTONS
    ===================================================== */

    .stButton > button {
        border-radius: 10px;
        font-weight: 650;
        min-height: 42px;
    }

    .stDownloadButton > button {
        border-radius: 10px;
        font-weight: 650;
        min-height: 42px;
    }


    /* =====================================================
       DATAFRAME
    ===================================================== */

    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }


    /* =====================================================
       STATUS BOX
    ===================================================== */

    .status-card {
        padding: 1rem 1.2rem;
        background: var(--datly-card);
        border-radius: 14px;
        border: 1px solid var(--datly-border);
        box-shadow: 0 3px 10px var(--datly-shadow);
        margin: 0.8rem 0;
    }


    /* =====================================================
       FOOTER
    ===================================================== */

    .datly-footer {
        margin-top: 3rem;
        padding: 1.5rem;
        text-align: center;
        color: var(--datly-muted);
        font-size: 0.85rem;
        border-top: 1px solid var(--datly-border);
    }

    .datly-footer strong {
        color: var(--datly-heading);
    }

        /* =====================================================
       RESPONSIVE DESIGN
       ===================================================== */

    @media (max-width: 768px) {

        /* Main Streamlit content */
        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            padding-top: 1rem !important;
        }

        /* Hero */
        .hero {
            padding: 1.5rem 1.2rem;
            border-radius: 16px;
            margin-bottom: 1.2rem;
        }

        .hero-top {
            gap: 10px;
        }

        .hero-icon {
            font-size: 2rem;
        }

        .hero-title {
            font-size: 2.2rem;
        }

        .hero-tagline {
            font-size: 1.05rem;
        }

        .hero-description {
            font-size: 0.9rem;
            line-height: 1.5;
        }

        /* Section headings */
        .section-title {
            font-size: 1.25rem;
        }

        .section-description {
            font-size: 0.85rem;
            line-height: 1.5;
        }

        /* Metric cards */
        .metric-card {
            padding: 1rem 0.7rem;
            min-height: 90px;
            margin-bottom: 0.7rem;
        }

        .metric-card h3 {
            font-size: 0.72rem;
        }

        .metric-card h2 {
            font-size: 1.4rem;
        }

        /* Quality score */
        .quality-score-card {
            padding: 1.2rem;
        }

        .quality-score-value {
            font-size: 2.1rem;
        }

        /* Comparison cards */
        .comparison-card {
            padding: 1rem;
        }

        /* Buttons */
        .stButton > button,
        .stDownloadButton > button {
            width: 100% !important;
            min-height: 44px;
            font-size: 15px !important;
        }

        /* Data tables */
        [data-testid="stDataFrame"] {
            width: 100% !important;
            overflow-x: auto !important;
        }

        /* Footer */
        .datly-footer {
            margin-top: 2rem;
            padding: 1rem;
            font-size: 0.75rem;
        }
    }

    </style>
    """
)


# =========================================================
# WELCOME SCREEN CSS
# =========================================================

st.markdown(
    """
    <style>

    /* ---------- MAIN WELCOME AREA ---------- */

    .datly-welcome {
        max-width: 1200px;
        margin: 2rem auto 2rem auto;
        text-align: center;
    }

    /* ---------- BIG ICON ---------- */

    .datly-welcome-icon {
        width: 80px;
        height: 80px;
        margin: 0 auto 0.8rem auto;

        display: flex;
        align-items: center;
        justify-content: center;

        border-radius: 35px;

        background: var(--datly-card);
        border: 1px solid var(--datly-border);

        box-shadow: 0 18px 50px var(--datly-shadow);

        font-size: 2.5rem;
    }

    /* ---------- BIG TITLE ---------- */

    .datly-welcome-title {
        font-size: 3rem;
        font-weight: 600;

        letter-spacing: -4px;
        line-height: 1;

        color: var(--datly-heading);

        margin-bottom: 0.5rem;
    }

    /* ---------- SUBTITLE ---------- */

    .datly-welcome-subtitle {
        max-width: 500px;

        margin: 0 auto;

        font-size: 1.15rem;
        font-weight: 400;

        line-height: 1.6;

        color: var(--datly-muted);
    }


    /* ---------- WELCOME CARDS ---------- */

    .datly-welcome-card {
        background: var(--datly-card);

        border: 1px solid var(--datly-border);

        border-radius: 30px;

        padding: 2rem 2rem;

        min-height: 250px;

        box-shadow: 0 18px 50px var(--datly-shadow);

        text-align: center;

        margin-bottom: 1.5rem;

        transition:
            transform 0.25s ease,
            box-shadow 0.25s ease;
    }


    .datly-welcome-card:hover {
        transform: translateY(-10px);

        box-shadow: 0 30px 65px var(--datly-shadow);
    }


    /* ---------- CARD ICON ---------- */

    .datly-welcome-card-icon {
        width: 75px;
        height: 75px;

        margin: 0 auto 1rem auto;

        display: flex;
        align-items: center;
        justify-content: center;

        border-radius: 20px;

        background: var(--datly-bg);

        border: 1px solid var(--datly-border);

        font-size: 2.8rem;
    }


    /* ---------- CARD TITLE ---------- */

    .datly-welcome-card-title {
        font-size: 2.1rem;

        font-weight: 900;

        color: var(--datly-heading);

        margin-bottom: 1.2rem;
    }


    /* ---------- CARD DESCRIPTION ---------- */

    .datly-welcome-card-text {
        max-width: 500px;

        margin: 0 auto 2.5rem auto;

        color: var(--datly-muted);

        font-size: 1.25rem;

        line-height: 1.8;
    }


    /* ---------- ABOUT PANEL ---------- */

    .datly-about {
        max-width: 1000px;

        margin: 3rem auto;

        padding: 3rem 3.5rem;

        background: var(--datly-card);

        border: 1px solid var(--datly-border);

        border-radius: 28px;

        box-shadow: 0 18px 50px var(--datly-shadow);

        text-align: left;
    }


    .datly-about h3 {
        color: var(--datly-heading);

        font-size: 1.6rem;

        margin-top: 1.8rem;

        margin-bottom: 1rem;
    }


    .datly-about h3:first-child {
        margin-top: 0;
    }


    .datly-about p,
    .datly-about li {
        color: var(--datly-muted);

        font-size: 1.1rem;

        line-height: 1.9;
    }

    /* =====================================================
   MOBILE WELCOME SCREEN
   ===================================================== */

@media (max-width: 768px) {

    .stButton > button {
        min-height: 36px !important;
        height: 36px !important;
        padding: 0.2rem 0.5rem !important;
        font-size: 12px !important;
    }

    /* Dataset profile metric cards */
    .metric-card {
        width: 100% !important;
        box-sizing: border-box;
        padding: 0.7rem 0.5rem !important;
    }

    .metric-card h3 {
        font-size: 0.7rem !important;
        white-space: normal !important;
    }

    .metric-card h2 {
        font-size: 1.3rem !important;
    }

    .datly-welcome {
        padding: 0.5rem 0.25rem;
        margin-bottom: 0.4rem;
    }

    .datly-welcome-icon {
        font-size: 2rem;
        margin-bottom: 0.2rem;
    }

    .datly-welcome-title {
        font-size: 1.7rem;
        line-height: 1.15;
        margin-bottom: 0.2rem;
    }

    .datly-welcome-subtitle {
        font-size: 0.78rem;
        line-height: 1.3;
        margin-bottom: 0.4rem;
    }

    .datly-welcome-card {
    padding: 0.45rem 0.6rem;
    margin-bottom: 0.25rem;
    }

    .datly-welcome-card-icon {
        font-size: 1.2rem;
        margin-bottom: 0.1rem;
    }

    .datly-welcome-card-title {
        font-size: 0.9rem;
        margin-bottom: 0.1rem;
    }

    .datly-welcome-card-text {
        font-size: 0.65rem;
        line-height: 1.2;
        margin-bottom: 0;
    }

    .datly-welcome-card-icon {
        font-size: 1.5rem;
        margin-bottom: 0.2rem;
    }

    .datly-welcome-card-title {
        font-size: 1rem;
        margin-bottom: 0.2rem;
    }

    .datly-welcome-card-text {
        font-size: 0.72rem;
        line-height: 1.3;
        margin-bottom: 0.2rem;
    }

    .datly-about {
        padding: 0.8rem;
        margin-top: 0.5rem;
    }

    .datly-about h3 {
        font-size: 1rem;
        margin-top: 0.5rem;
    }

    .datly-about p,
    .datly-about li {
        font-size: 0.75rem;
        line-height: 1.35;
    }

}

    /* Keep welcome screen compact on real mobile */
    html, body {
        overflow-x: hidden;
    }

    .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
    }

    .datly-welcome {
        margin-top: 0 !important;
       margin-bottom: 0.2rem !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def section_heading(title, description):

    st.html(
        f"""
        <div class="section-divider"></div>

        <div class="section-title">
            {title}
        </div>

        <div class="section-description">
            {description}
        </div>
        """

    )


def metric_card(label, value):

    st.html(
        f"""
        <div class="metric-card">

            <h3>{label}</h3>

            <h2>{value}</h2>

        </div>
        """
    )


def display_dataframe_preview(
    dataframe,
    max_rows=MAX_DISPLAY_ROWS
):
    """
    Display only a limited preview in Streamlit.

    The original DataFrame is not modified.
    """

    if dataframe is None:
        return

    total_rows = len(dataframe)

    if total_rows > max_rows:

        st.dataframe(
            dataframe.head(max_rows),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            f"Showing first {max_rows:,} of "
            f"{total_rows:,} rows. "
            "The complete dataset remains available "
            "for download."
        )

    else:

        st.dataframe(
            dataframe,
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# DATLY WELCOME SCREEN
# =========================================================

if "datly_started" not in st.session_state:
    st.session_state["datly_started"] = False

if "show_datly_about" not in st.session_state:
    st.session_state["show_datly_about"] = False


if not st.session_state["datly_started"]:

    st.html(
        """
        <div class="datly-welcome">

            <div class="datly-welcome-icon">
                🛡️
            </div>

            <div class="datly-welcome-title">
                Welcome to Datly
            </div>

            <div class="datly-welcome-subtitle">
                Smart Data Quality & Anomaly Detection Platform
            </div>

        </div>
        """,

    )


    col1, col2 = st.columns(2, gap="small")


    # -----------------------------------------------------
    # KNOW ABOUT DATLY
    # -----------------------------------------------------

    with col1:

        st.html(
            """
            <div class="datly-welcome-card">

                <div class="datly-welcome-card-icon">
                    📖
                </div>

                <div class="datly-welcome-card-title">
                    Know About Datly
                </div>

                <div class="datly-welcome-card-text">
                    Learn what Datly does and how it helps
                    you understand, clean, and improve
                    the quality of your data.
                </div>

            </div>
            """,

        )

        if st.button(
            "📖 Know About Datly",
            use_container_width=True
        ):

            st.session_state["show_datly_about"] = (
                not st.session_state["show_datly_about"]
            )

            st.rerun()


    # -----------------------------------------------------
    # START DATLY
    # -----------------------------------------------------

    with col2:

        st.html(
            """
            <div class="datly-welcome-card">

                <div class="datly-welcome-card-icon">
                    🚀
                </div>

                <div class="datly-welcome-card-title">
                    Let's Start with Datly
                </div>

                <div class="datly-welcome-card-text">
                    Upload your dataset and let Datly
                    analyze quality, detect anomalies,
                    clean data, and generate reports.
                </div>

            </div>
            """,

        )

        if st.button(
            "🚀 Let's Start with Datly",
            use_container_width=True
        ):

            st.session_state["datly_started"] = True
            st.rerun()


    # -----------------------------------------------------
    # ABOUT DATLY
    # -----------------------------------------------------

    if st.session_state["show_datly_about"]:

        st.html(
            """
            <div class="datly-about">

                <h3>🛡️ What is Datly?</h3>

                <p>
                    Datly is a data quality and anomaly detection
                    platform designed to help you understand the
                    condition of your dataset before using it
                    for analysis or machine learning.
                </p>

                <h3>✨ What can Datly do?</h3>

                <ul>
                    <li>📊 Profile your dataset</li>
                    <li>🔍 Check data quality</li>
                    <li>🚨 Detect unusual values and anomalies</li>
                    <li>🧹 Clean detected data issues</li>
                    <li>📈 Calculate quality scores</li>
                    <li>💡 Generate data-quality recommendations</li>
                    <li>📋 Generate professional quality reports</li>
                </ul>

                <p>
                    Simply upload a CSV file and Datly will guide
                    you through the data-quality analysis process.
                </p>

            </div>
            """,

        )


    st.stop()


# =========================================================
# HERO SECTION
# =========================================================

st.html(
    """
    <div class="hero">

        <div class="hero-top">

            <div class="hero-icon">
                🛡️
            </div>

            <div class="hero-title">
                Datly
            </div>

        </div>

        <div class="hero-tagline">
            Smart Data Quality &amp; Anomaly Detection
        </div>

        <p class="hero-description">
            Clean data. Detect anomalies. Measure quality.
            Generate professional data-quality reports.
        </p>

        <div class="hero-badge">
            Data Quality Intelligence Platform
        </div>

    </div>
    """
)




# =========================================================
# CSV UPLOAD
# =========================================================

section_heading(
    "📂 Upload Dataset",
    "Upload a CSV file to analyze, validate, detect anomalies, and clean."
)

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    key="dataset_uploader"
)


if uploaded_file is None:

    st.info(
        "👆 Upload a CSV file to start analyzing your dataset."
    )

    st.stop()


# =========================================================
# FILE ID
# =========================================================

file_bytes = uploaded_file.getvalue()

file_id = hashlib.md5(
    file_bytes
).hexdigest()


# =========================================================
# NEW FILE DETECTION
# =========================================================


is_new_file = (
    st.session_state.get("file_id") != file_id
)

if is_new_file:

    # Preserve the selected theme
    current_theme = st.session_state.get("theme", "Light")
    datly_started = st.session_state.get("datly_started", False)

    # Clear old dataset-specific state
    st.session_state.clear()

    # Restore theme and new file ID
    st.session_state["theme"] = current_theme
    st.session_state["datly_started"] = datly_started
    st.session_state["file_id"] = file_id
    # -----------------------------------------------------
    # READ CSV ONLY ONCE
    # -----------------------------------------------------

    try:

        df = pd.read_csv(
            io.BytesIO(file_bytes)
        )

    except Exception as error:

        st.error(
            "❌ The CSV file could not be read. "
            "Please check the file format. "
            f"Details: {error}"
        )

        st.stop()

    if df.shape[1] == 0:

        st.error(
            "❌ The uploaded dataset has no columns."
        )

        st.stop()


    print("\n========== DATASET DIAGNOSTICS ==========")

    print("\nCOLUMN NAMES:")
    print(df.columns.tolist())

    print("\nUNIQUE VALUES PER COLUMN:")
    print(df.nunique())

    print("\nDATASET STATISTICS:")
    print(df.describe(include="all").to_string())

    print("\n=========================================")


    if df.shape[0] == 0:

        st.warning(
            "⚠️ Empty Dataset"
        )

        st.info(
            "The uploaded CSV contains no data rows. "
            "Please upload a dataset containing at least one row."
        )

        st.stop()

    if df.isna().all().all():

        st.warning(
            "⚠️ Empty Dataset"
        )

        st.info(
            "The uploaded dataset contains rows, but all values are missing. "
            "Please upload a dataset containing actual data."
        )

        st.stop()


    # -----------------------------------------------------
    # STORE DATASET
    # -----------------------------------------------------

    st.session_state["df"] = df

    st.session_state["dataset_name"] = (
        uploaded_file.name
    )


    # -----------------------------------------------------
    # INITIAL PROFILE
    # -----------------------------------------------------

    st.session_state["profile"] = (
        profile_dataset(df)
    )


    # -----------------------------------------------------
    # QUALITY RULES
    # -----------------------------------------------------

    st.session_state["quality_results"] = (
        run_quality_rules(df)
    )


    # -----------------------------------------------------
    # INITIAL ANOMALY DETECTION
    # -----------------------------------------------------

    initial_results = detect_anomalies(df)

    st.session_state["initial_anomaly_results"] = (
        initial_results
    )

    st.session_state["initial_anomaly_summary"] = (
        anomaly_summary(initial_results)
    )

    st.session_state["initial_anomaly_report"] = (
        combined_anomaly_report(initial_results)
    )


    # -----------------------------------------------------
    # INITIAL QUALITY SCORE
    # -----------------------------------------------------

    initial_anomaly_report = (
        st.session_state["initial_anomaly_report"]
    )

    quality_results = (
        st.session_state["quality_results"]
    )

    raw_anomaly_penalty = calculate_anomaly_score(
        initial_anomaly_report,
        len(df)
    )

    completeness_score = (
        calculate_completeness_score(
            quality_results
        )
    )

    validity_score = (
        calculate_validity_score(
            quality_results
        )
    )

    uniqueness_score = (
        calculate_uniqueness_score(
            quality_results
        )
    )

    anomaly_quality_score = (
        calculate_anomaly_quality_score(
            raw_anomaly_penalty
        )
    )

    overall_quality_score = (
        calculate_overall_quality_score(
            completeness_score,
            validity_score,
            uniqueness_score,
            anomaly_quality_score
        )
    )

    quality_status = get_quality_status(
        overall_quality_score
    )


    st.session_state["completeness_score"] = (
        completeness_score
    )

    st.session_state["validity_score"] = (
        validity_score
    )

    st.session_state["uniqueness_score"] = (
        uniqueness_score
    )

    st.session_state["anomaly_quality_score"] = (
        anomaly_quality_score
    )

    st.session_state["overall_quality_score"] = (
        overall_quality_score
    )

    st.session_state["quality_status"] = (
        quality_status
    )


# =========================================================
# RESTORE CACHED DATA
# =========================================================

df = st.session_state["df"]

dataset_name = st.session_state[
    "dataset_name"
]

profile = st.session_state[
    "profile"
]

quality_results = st.session_state[
    "quality_results"
]

anomaly_summary_df = st.session_state[
    "initial_anomaly_summary"
]

anomaly_report = st.session_state[
    "initial_anomaly_report"
]

completeness_score = st.session_state[
    "completeness_score"
]

validity_score = st.session_state[
    "validity_score"
]

uniqueness_score = st.session_state[
    "uniqueness_score"
]

anomaly_quality_score = st.session_state[
    "anomaly_quality_score"
]

overall_quality_score = st.session_state[
    "overall_quality_score"
]

quality_status = st.session_state[
    "quality_status"
]


# =========================================================
# DATASET PREVIEW
# =========================================================

section_heading(
    "📋 Dataset Preview",
    "Explore a sample of the uploaded data before quality analysis."
)

st.caption(
    f"📁 **{dataset_name}**  ·  "
    f"📊 {len(df):,} rows  ·  "
    f"📐 {len(df.columns):,} columns"
)


if df.empty:

    st.warning(
        "⚠️ The dataset contains columns but no rows."
    )

else:

    display_dataframe_preview(df)


# =========================================================
# DATA EXPLORATION MENU
# =========================================================

section_heading(
    "🔍 Explore Your Dataset",
    "Choose an analysis to understand data quality, structure, and anomalies."
)

if "selected_analysis" not in st.session_state:
    st.session_state["selected_analysis"] = None


menu_col1, menu_col2, menu_col3 = st.columns(3)


with menu_col1:

    if st.button(
        "🔎 Dataset Profile",
        use_container_width=True
    ):
        st.session_state["selected_analysis"] = "profile"


with menu_col2:

    if st.button(
        "📋 Column Profile",
        use_container_width=True
    ):
        st.session_state["selected_analysis"] = "columns"


with menu_col3:

    if st.button(
        "📐 Numerical Statistics",
        use_container_width=True
    ):
        st.session_state["selected_analysis"] = "statistics"

st.write("")

menu_col4, menu_col5, menu_col6 = st.columns(3)


with menu_col4:

    if st.button(
        "🛡️ Quality Rules",
        use_container_width=True
    ):
        st.session_state["selected_analysis"] = "quality"


with menu_col5:

    if st.button(
        "🚨 Anomaly Detection",
        use_container_width=True
    ):
        st.session_state["selected_analysis"] = "anomalies"


with menu_col6:

    if st.button(
        "📊 Quality Score",
        use_container_width=True
    ):
        st.session_state["selected_analysis"] = "score"


# =========================================================
# DATASET PROFILE
# =========================================================

if st.session_state["selected_analysis"] == "profile":

    section_heading(
        "🔎 Dataset Profile",
        "Understand the structure and characteristics of your dataset."
    )

    profile_col1, profile_col2, profile_col3, profile_col4 = (
        st.columns(4)
    )

    with profile_col1:
        metric_card(
            "Total Rows",
            f"{profile['rows']:,}"
        )

    with profile_col2:
        metric_card(
            "Total Columns",
            f"{profile['columns']:,}"
        )

    with profile_col3:
        metric_card(
            "Numerical",
            f"{len(profile['numerical_columns']):,}"
        )

    with profile_col4:
        metric_card(
            "Categorical",
            f"{len(profile['categorical_columns']):,}"
        )



# =========================================================
# COLUMN PROFILE
# =========================================================

if st.session_state["selected_analysis"] == "columns":

    section_heading(
        "📋 Column Profile",
        "Detailed information about each column in the dataset."
    )

    column_profile_df = profile["column_profile"]

    if column_profile_df is None or column_profile_df.empty:

        st.info(
            "ℹ️ No column profile information is available."
        )

    else:

        display_dataframe_preview(
            column_profile_df
        )


# =========================================================
# NUMERICAL STATISTICS
# =========================================================

if st.session_state["selected_analysis"] == "statistics":

    if profile["numerical_statistics"]:

        section_heading(
            "📐 Numerical Statistics",
            "Statistical summary of numerical columns."
        )

        statistics_df = (
            pd.DataFrame(
                profile["numerical_statistics"]
            )
            .T
            .reset_index()
            .rename(
                columns={
                    "index": "Column"
                }
            )
        )

        display_dataframe_preview(
            statistics_df
        )

    else:

        st.info(
            "No numerical columns are available for statistical analysis."
        )


# =========================================================
# DATA QUALITY RULES
# =========================================================

if st.session_state["selected_analysis"] == "quality":

    section_heading(
        "🛡️ Data Quality Rules",
        "Check completeness, validity, uniqueness, and other quality issues."
    )


    rule_table = quality_results[
        "rule_table"
    ]


    display_rule_table = rule_table[
        [
            "rule",
            "category",
            "count",
            "rate",
            "severity",
            "details"
        ]
    ].rename(
        columns={
            "rule": "Rule",
            "category": "Category",
            "count": "Issues",
            "rate": "Rate %",
            "severity": "Severity",
            "details": "Details"
        }
    )


    st.dataframe(
        display_rule_table,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# INITIAL ANOMALY DETECTION
# =========================================================

if st.session_state["selected_analysis"] == "anomalies":

    section_heading(
        "🚨 Anomaly Detection",
        "Detect unusual or suspicious values in the dataset."
    )

    if anomaly_summary_df.empty:

        st.success(
            "✅ Datly found no anomaly categories in this dataset."
        )

    else:

        anomaly_col1, anomaly_col2, anomaly_col3 = (
            st.columns(3)
        )

        total_anomalies = len(anomaly_report)

        if "severity" in anomaly_report.columns:

            high_anomalies = (
                anomaly_report["severity"]
                .astype(str)
                .str.lower()
                .eq("high")
                .sum()
            )

            critical_anomalies = (
                anomaly_report["severity"]
                .astype(str)
                .str.lower()
                .eq("critical")
                .sum()
            )

        else:

            high_anomalies = 0
            critical_anomalies = 0

        with anomaly_col1:

            metric_card(
                "Total Anomalies",
                f"{total_anomalies:,}"
            )

        with anomaly_col2:

            metric_card(
                "High Severity",
                f"{high_anomalies:,}"
            )

        with anomaly_col3:

            metric_card(
                "Critical",
                f"{critical_anomalies:,}"
            )

        st.markdown(
            "#### 📋 Anomaly Summary"
        )

        display_dataframe_preview(
            anomaly_summary_df
        )


    # =====================================================
    # INITIAL ANOMALY REPORT
    # =====================================================

    if anomaly_report.empty:

        st.success(
            "✅ No anomalies detected!"
        )

    else:

        st.markdown(
            "#### 🔎 Detected Anomalies"
        )

        display_columns = [
            "row_index",
            "column",
            "value",
            "z_score",
            "anomaly_method",
            "severity",
            "individual_score"
        ]

        display_columns = [
            column
            for column in display_columns
            if column in anomaly_report.columns
        ]

        anomaly_display = anomaly_report[
            display_columns
        ]

        display_dataframe_preview(
            anomaly_display,
            MAX_ANOMALY_DISPLAY_ROWS
        )


# =========================================================
# INITIAL QUALITY SCORE
# =========================================================

if st.session_state["selected_analysis"] == "score":

    section_heading(
        "📊 Initial Quality Score",
        "Measure the overall quality of the uploaded dataset."
    )

    # =====================================================
    # QUALITY SCORE SUMMARY
    # =====================================================

    st.html(
        """
        <div class="quality-score-card">

            <div class="quality-score-label">
                Overall Dataset Quality
            </div>

            <div class="quality-score-value">
        """
        + f"{overall_quality_score:.2f}"
        + """
                <span style="
                    font-size: 1.1rem;
                    font-weight: 600;
                    color: #64748b;
                ">
                    / 100
                </span>
            </div>

            <div class="quality-score-status">
                Current dataset status:
                <strong>
        """
        + str(quality_status)
        + """
                </strong>
            </div>

        </div>
        """
    )


    # =====================================================
    # QUALITY DIMENSIONS
    # =====================================================

    score_col1, score_col2, score_col3, score_col4 = (
        st.columns(4)
    )

    with score_col1:

        metric_card(
            "Completeness",
            f"{completeness_score:.2f}"
        )

    with score_col2:

        metric_card(
            "Validity",
            f"{validity_score:.2f}"
        )

    with score_col3:

        metric_card(
            "Uniqueness",
            f"{uniqueness_score:.2f}"
        )

    with score_col4:

        metric_card(
            "Anomaly Quality",
            f"{anomaly_quality_score:.2f}"
        )


    # =====================================================
    # QUALITY PROGRESS
    # =====================================================

    st.progress(
        min(
            max(
                overall_quality_score / 100,
                0.0
            ),
            1.0
        ),
        text=(
            f"Quality Score: "
            f"{overall_quality_score:.2f}/100"
        )
    )

    st.caption(
        "Datly combines completeness, validity, uniqueness, "
        "and anomaly quality to calculate the overall score."
    )


# =========================================================
# DATA CLEANING
# =========================================================

section_heading(
    "🧹 Data Cleaning",
    "Iteratively clean detected issues and recheck the dataset after each pass."
)


if st.button(
    "🧹 Clean Dataset",
    type="primary",
    use_container_width=True
):

    current_df = df.copy()

    current_anomaly_report = (
        anomaly_report.copy()
    )

    all_cleaning_reports = []

    iterations_completed = 0

    max_iterations = 5

    cleaning_progress = st.progress(
        0,
        text="🧹 Preparing dataset cleaning..."
    )

    cleaning_status = st.empty()


    # -----------------------------------------------------
    # NO ANOMALIES
    # -----------------------------------------------------

    if (
        current_anomaly_report.empty
        and not current_df.duplicated().any()
    ):

        cleaning_progress.progress(
            100,
            text="✅ No cleaning required — 100%"
        )

        cleaning_status.success(
            "✅ Dataset is already clean  — no cleaning required."
        )


    # -----------------------------------------------------
    # ITERATIVE CLEANING
    # -----------------------------------------------------

    else:

        for iteration in range(
            max_iterations
        ):

            iteration_number = (
                iteration + 1
            )

            if (
                current_anomaly_report.empty
                and not current_df.duplicated().any()
            ):

                break


            iteration_start = int(
                (
                    iteration
                    / max_iterations
                )
                * 100
            )

            cleaning_progress.progress(
                iteration_start,
                text=(
                    f"🧹 Starting cleaning iteration "
                    f"{iteration_number}/{max_iterations} "
                    f"— {iteration_start}%"
                )
            )

            cleaning_status.info(
                f"🔍 Processing cleaning iteration "
                f"{iteration_number} of "
                f"{max_iterations}..."
            )


            preparing_progress = min(
                iteration_start + 8,
                95
            )

            cleaning_progress.progress(
                preparing_progress,
                text=(
                    f"📋 Preparing cleaning actions "
                    f"— {preparing_progress}%"
                )
            )

            cleaning_status.info(
                f"📋 Preparing cleaning actions "
                f"for iteration {iteration_number}..."
            )


            cleaning_progress_value = min(
                iteration_start + 20,
                95
            )

            cleaning_progress.progress(
                cleaning_progress_value,
                text=(
                    f"🧹 Applying cleaning actions "
                    f"— {cleaning_progress_value}%"
                )
            )

            cleaning_status.info(
                f"🧹 Cleaning dataset "
                f"(iteration {iteration_number})..."
            )


            cleaned_df, cleaning_report = (
                clean_dataset(
                    current_df,
                    current_anomaly_report
                )
            )


            iterations_completed += 1


            if cleaned_df.equals(current_df):

                if not cleaning_report.empty:
                    if "iteration" in cleaning_report.columns:
                        cleaning_report["iteration"] = iteration_number
                    else:
                        cleaning_report.insert(
                            0, "iteration", iteration_number
                        )

                    all_cleaning_reports.append(cleaning_report)

                cleaning_status.info(
                    "ℹ️ No further automatic changes are needed. "
                    "Remaining anomalies are preserved for review. "
                )
                break



            report_progress = min(
                iteration_start + 35,
                95
            )

            cleaning_progress.progress(
                report_progress,
                text=(
                    f"📝 Recording cleaning results "
                    f"— {report_progress}%"
                )
            )


            if not cleaning_report.empty:

                if "iteration" in cleaning_report.columns:

                    cleaning_report["iteration"] = (
                        iteration_number
                    )

                else:

                    cleaning_report.insert(
                        0,
                        "iteration",
                        iteration_number
                    )

                all_cleaning_reports.append(
                    cleaning_report
                )


            recheck_progress = min(
                iteration_start + 50,
                95
            )

            cleaning_progress.progress(
                recheck_progress,
                text=(
                    f"🔎 Rechecking dataset "
                    f"— {recheck_progress}%"
                )
            )

            cleaning_status.info(
                "🔎 Detecting remaining anomalies..."
            )


            cleaned_results = detect_anomalies(
                cleaned_df
            )


            analysis_progress = min(
                iteration_start + 65,
                95
            )

            cleaning_progress.progress(
                analysis_progress,
                text=(
                    f"📊 Updating anomaly analysis "
                    f"— {analysis_progress}%"
                )
            )


            new_anomaly_report = (
                combined_anomaly_report(
                    cleaned_results
                )
            )


            current_df = cleaned_df

            current_anomaly_report = (
                new_anomaly_report
            )


            remaining_anomalies = len(
                current_anomaly_report
            )


            iteration_complete_progress = min(
                iteration_start + 80,
                95
            )

            cleaning_progress.progress(
                iteration_complete_progress,
                text=(
                    f"🔄 Iteration "
                    f"{iteration_number}/{max_iterations} "
                    f"complete — "
                    f"{iteration_complete_progress}%"
                )
            )

            cleaning_status.info(
                f"🔄 Iteration {iteration_number} "
                f"complete — "
                f"{remaining_anomalies} "
                f"anomalies remaining."
            )


            if (
                current_anomaly_report.empty
                and not current_df.duplicated().any()
            ):

                cleaning_progress.progress(
                    95,
                    text="🔎 Final validation — 95%"
                )

                cleaning_status.success(
                    f"✅ All anomalies removed after "
                    f"{iterations_completed} "
                    f"iteration(s)."
                )

                break


            if iteration_number == max_iterations:

                cleaning_progress.progress(
                    95,
                    text="⚠️ Final validation — 95%"
                )

                cleaning_status.warning(
                    f"⚠️ Cleaning stopped after "
                    f"{max_iterations} iterations. "
                    f"{remaining_anomalies} "
                    f"anomalies remain."
                )


    # =====================================================
    # FINAL DATA
    # =====================================================

    final_df = current_df

    final_anomaly_report = (
        current_anomaly_report
    )


    # =====================================================
    # FINAL QUALITY VALIDATION
    # =====================================================

    cleaning_progress.progress(
        96,
        text="🔍 Running final quality validation — 96%"
    )

    cleaning_status.info(
        "🔍 Running final quality validation..."
    )


    final_anomaly_penalty = calculate_anomaly_score(
        final_anomaly_report,
        len(final_df)
    )


    final_quality_results = run_quality_rules(
        final_df
    )


    cleaning_progress.progress(
        97,
        text="📊 Calculating final quality score — 97%"
    )


    final_completeness_score = (
        calculate_completeness_score(
            final_quality_results
        )
    )

    final_validity_score = (
        calculate_validity_score(
            final_quality_results
        )
    )

    final_uniqueness_score = (
        calculate_uniqueness_score(
            final_quality_results
        )
    )

    final_anomaly_quality_score = (
        calculate_anomaly_quality_score(
            final_anomaly_penalty
        )
    )

    final_quality_score = (
        calculate_overall_quality_score(
            final_completeness_score,
            final_validity_score,
            final_uniqueness_score,
            final_anomaly_quality_score
        )
    )


    # =====================================================
    # COMBINE CLEANING REPORTS
    # =====================================================

    cleaning_progress.progress(
        98,
        text="📝 Preparing final cleaning report — 98%"
    )


    if all_cleaning_reports:

        final_cleaning_report = pd.concat(
            all_cleaning_reports,
            ignore_index=True
        )

    else:

        final_cleaning_report = (
            pd.DataFrame()
        )


    # =====================================================
    # SAVE RESULTS
    # =====================================================

    st.session_state[
        "cleaned_df"
    ] = final_df

    st.session_state[
        "cleaning_report"
    ] = final_cleaning_report

    st.session_state[
        "cleaned_quality_score"
    ] = final_quality_score

    st.session_state[
        "final_completeness_score"
    ] = final_completeness_score

    st.session_state[
        "final_validity_score"
    ] = final_validity_score

    st.session_state[
        "final_uniqueness_score"
    ] = final_uniqueness_score

    st.session_state[
        "final_anomaly_quality_score"
    ] = final_anomaly_quality_score

    st.session_state[
        "before_anomaly_count"
    ] = len(anomaly_report)

    st.session_state[
        "after_anomaly_count"
    ] = len(final_anomaly_report)

    st.session_state[
        "cleaning_iterations"
    ] = iterations_completed

    st.session_state[
        "cleaned_quality_results"
    ] = final_quality_results

    st.session_state[
        "final_anomaly_report"
    ] = final_anomaly_report


    # Clear old report because dataset state changed.

    st.session_state.pop(
        "quality_report",
        None
    )

    st.session_state.pop(
        "quality_report_html",
        None
    )

    st.session_state.pop(
        "final_status",
        None
    )

    st.session_state.pop(
        "recommendations",
        None
    )

    st.session_state.pop(
        "cleaned_csv_data",
        None
    )


    # =====================================================
    # FINAL CLEANING STATUS
    # =====================================================

    if final_anomaly_report.empty:

        cleaning_progress.progress(
            100,
            text="✅ Cleaning complete — 100%"
        )

        cleaning_status.success(
            "🎉 Dataset successfully cleaned! "
            "No anomalies remain."
        )

    else:

        cleaning_progress.progress(
            100,
            text="⚠️ Cleaning process finished — 100%"
        )

        cleaning_status.warning(
            f"⚠️ Cleaning completed, but "
            f"{len(final_anomaly_report)} "
            f"anomaly remain."
            f"Review the anomaly report for details."
        )


# =========================================================
# POST-CLEANING RESULTS
# =========================================================

section_heading(
    "✨ Post-Cleaning Results",
    "Review the cleaned dataset, quality improvements, and remaining anomalies."
)

if "cleaned_df" in st.session_state:

    cleaned_df = st.session_state.get("cleaned_df")
    cleaning_report = st.session_state.get(
        "cleaning_report",
        pd.DataFrame()
    )
    cleaned_quality_score = st.session_state["cleaned_quality_score"]

    before_anomaly_count = st.session_state["before_anomaly_count"]
    after_anomaly_count = st.session_state["after_anomaly_count"]
    cleaning_iterations = st.session_state["cleaning_iterations"]

    final_anomaly_report = st.session_state["final_anomaly_report"]

    final_completeness_score = st.session_state["final_completeness_score"]
    final_validity_score = st.session_state["final_validity_score"]
    final_uniqueness_score = st.session_state["final_uniqueness_score"]
    final_anomaly_quality_score = st.session_state[
        "final_anomaly_quality_score"
    ]

    final_quality_results = st.session_state[
        "cleaned_quality_results"
    ]


    # =====================================================
    # POST-CLEANING REVIEW MENU
    # =====================================================

    section_heading(
        "🔍 Review Cleaning Results",
        "Explore the cleaned dataset, quality improvements, anomalies, and recommendations."
    )

    if "selected_cleaned_analysis" not in st.session_state:
        st.session_state["selected_cleaned_analysis"] = None

    clean_menu_col1, clean_menu_col2 = st.columns(2)

    with clean_menu_col1:
        if st.button(
            "✨ Cleaned Dataset",
            use_container_width=True
        ):
            st.session_state[
                "selected_cleaned_analysis"
            ] = "cleaned_dataset"

    with clean_menu_col2:
        if st.button(
            "📈 Before vs After",
            use_container_width=True
        ):
            st.session_state[
                "selected_cleaned_analysis"
            ] = "comparison"

    st.write("")

    clean_menu_col3, clean_menu_col4 = st.columns(2)

    with clean_menu_col3:
        if st.button(
            "🏆 Final Quality Score",
            use_container_width=True
        ):
            st.session_state[
                "selected_cleaned_analysis"
            ] = "final_score"

    with clean_menu_col4:
        if st.button(
            "🔎 Anomaly Recheck",
            use_container_width=True
        ):
            st.session_state[
                "selected_cleaned_analysis"
            ] = "recheck"

    st.write("")

    clean_menu_col5, clean_menu_col6 = st.columns(2)

    with clean_menu_col5:
        if st.button(
            "🔄 Cleaning Process",
            use_container_width=True
        ):
            st.session_state[
                "selected_cleaned_analysis"
            ] = "process"

    with clean_menu_col6:
        if st.button(
            "💡 Recommendations",
            use_container_width=True
        ):
            st.session_state[
                "selected_cleaned_analysis"
            ] = "recommendations"

    if st.button(
        "📝 Cleaning Report",
        type="primary",
        use_container_width=True
    ):
        st.session_state[
            "selected_cleaned_analysis"
        ] = "report"


    # =====================================================
    # CLEANED DATASET
    # =====================================================

    if (
        st.session_state["selected_cleaned_analysis"]
        == "cleaned_dataset"
    ):

        section_heading(
            "✨ Cleaned Dataset",
            "Preview of the final cleaned dataset."
        )

        display_dataframe_preview(cleaned_df)


    # =====================================================
    # BEFORE VS AFTER
    # =====================================================

    if (
        st.session_state["selected_cleaned_analysis"]
        == "comparison"
    ):

        section_heading(
            "📈 Before vs After",
            "Compare dataset quality before and after cleaning."
        )

        comparison_data = {
            "Metric": [
                "Completeness",
                "Validity",
                "Uniqueness",
                "Anomaly Quality",
                "Overall Quality"
            ],

            "Before Cleaning": [
                completeness_score,
                validity_score,
                uniqueness_score,
                anomaly_quality_score,
                overall_quality_score
            ],

            "After Cleaning": [
                final_completeness_score,
                final_validity_score,
                final_uniqueness_score,
                final_anomaly_quality_score,
                cleaned_quality_score
            ]
        }

        comparison_df = pd.DataFrame(comparison_data)

        comparison_df["Improvement"] = (
            comparison_df["After Cleaning"]
            - comparison_df["Before Cleaning"]
        )

        for column in [
            "Before Cleaning",
            "After Cleaning",
            "Improvement"
        ]:

            comparison_df[column] = (
                comparison_df[column].round(2)
            )

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("### 📊 Overall Improvement")

        overall_improvement = (
            cleaned_quality_score
            - overall_quality_score
        )

        metric_card(
            "Quality Improvement",
            f"+{overall_improvement:.2f}"
        )


    # =====================================================
    # FINAL QUALITY SCORE
    # =====================================================

    if (
        st.session_state["selected_cleaned_analysis"]
        == "final_score"
    ):

        section_heading(
            "🏆 Final Quality Score",
            "Final quality assessment after all cleaning iterations."
        )

        score_col1, score_col2, score_col3 = st.columns(3)

        with score_col1:

            metric_card(
                "Overall Quality",
                f"{cleaned_quality_score:.2f}"
            )

        with score_col2:

            metric_card(
                "Anomalies Remaining",
                str(after_anomaly_count)
            )

        with score_col3:

            metric_card(
                "Cleaning Iterations",
                str(cleaning_iterations)
            )

        st.markdown("### 📋 Quality Breakdown")

        final_score_df = pd.DataFrame({

            "Metric": [
                "Completeness",
                "Validity",
                "Uniqueness",
                "Anomaly Quality"
            ],

            "Score": [
                final_completeness_score,
                final_validity_score,
                final_uniqueness_score,
                final_anomaly_quality_score
            ]
        })

        final_score_df["Score"] = (
            final_score_df["Score"].round(2)
        )

        st.dataframe(
            final_score_df,
            use_container_width=True,
            hide_index=True
        )


    # =====================================================
    # ANOMALY RECHECK
    # =====================================================

    if (
        st.session_state["selected_cleaned_analysis"]
        == "recheck"
    ):

        section_heading(
            "🔎 Anomaly Recheck",
            "Verify whether anomalies remain after cleaning."
        )

        if after_anomaly_count == 0:

            st.success(
                "✅ No anomalies remain after cleaning."
            )

        else:

            st.warning(
                f"⚠️ {after_anomaly_count} anomalies remain."
            )

            if (
                final_anomaly_report is not None
                and not final_anomaly_report.empty
            ):

                display_dataframe_preview(
                    final_anomaly_report,
                    MAX_ANOMALY_DISPLAY_ROWS
                )


    # =====================================================
    # CLEANING PROCESS
    # =====================================================

    if (
        st.session_state["selected_cleaned_analysis"]
        == "process"
    ):

        section_heading(
            "🔄 Cleaning Process",
            "Summary of the iterative cleaning process."
        )

        process_col1, process_col2 = st.columns(2)

        with process_col1:

            metric_card(
                "Iterations Completed",
                str(cleaning_iterations)
            )

        with process_col2:

            metric_card(
                "Anomalies Resolved",
                str(
                    before_anomaly_count
                    - after_anomaly_count
                )
            )

        st.markdown("### 🧹 Cleaning Actions Performed")

        if (
            cleaning_report is not None
            and not cleaning_report.empty
        ):

            st.dataframe(
                cleaning_report,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "No cleaning actions were required."
            )


    # =====================================================
    # RECOMMENDATIONS
    # =====================================================

    recommendation_df = generate_recommendations(
        final_quality_results,
        final_anomaly_report,
        cleaned_quality_score
    )

    if (
        st.session_state["selected_cleaned_analysis"]
        == "recommendations"
    ):

        section_heading(
            "💡 Recommendations",
            "Suggestions for maintaining and improving dataset quality."
        )

        if (
            recommendation_df is not None
            and not recommendation_df.empty
        ):

            st.dataframe(
                recommendation_df,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.success(
                "✅ No additional recommendations. "
                "Your dataset is in good condition."
            )


    # =====================================================
    # CLEANING REPORT
    # =====================================================

    if (
        st.session_state["selected_cleaned_analysis"]
        == "report"
    ):

        section_heading(
            "📝 Cleaning Report",
            "Detailed summary of all cleaning actions performed on the dataset."
        )

        # -------------------------------------------------
        # REPORT SUMMARY
        # -------------------------------------------------

        report_col1, report_col2, report_col3, report_col4 = st.columns(4)

        with report_col1:
            metric_card(
                "Iterations",
                str(cleaning_iterations)
            )

        with report_col2:
            metric_card(
                "Anomalies Before",
                str(before_anomaly_count)
            )

        with report_col3:
            metric_card(
                "Anomalies After",
                str(after_anomaly_count)
            )

        with report_col4:
            metric_card(
                "Final Quality Score",
                f"{cleaned_quality_score:.2f}"
            )

        st.markdown("### 📋 Dataset Information")

        report_info = pd.DataFrame({
            "Property": [
                "Dataset",
                "Rows Before Cleaning",
                "Rows After Cleaning",
                "Columns",
                "Cleaning Iterations"
            ],

            "Value": [
                dataset_name,
                len(df),
                len(cleaned_df),
                len(cleaned_df.columns),
                cleaning_iterations
            ]
        })

        st.dataframe(
            report_info,
            use_container_width=True,
            hide_index=True
        )

        # -------------------------------------------------
        # DETAILED CLEANING ACTIONS
        # -------------------------------------------------

        st.markdown("### 🧹 Cleaning Actions Performed")

        if (
            cleaning_report is not None
            and not cleaning_report.empty
        ):

            st.dataframe(
                cleaning_report,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.success(
                "✅ No cleaning actions were required."
            )

        # -------------------------------------------------
        # FINAL QUALITY BREAKDOWN
        # -------------------------------------------------

        st.markdown("### 📊 Final Quality Breakdown")

        final_report_quality = pd.DataFrame({

            "Quality Dimension": [
                "Completeness",
                "Validity",
                "Uniqueness",
                "Anomaly Quality",
                "Overall Quality"
            ],

            "Score": [
                final_completeness_score,
                final_validity_score,
                final_uniqueness_score,
                final_anomaly_quality_score,
                cleaned_quality_score
            ]
        })

        final_report_quality["Score"] = (
            final_report_quality["Score"].round(2)
        )

        st.dataframe(
            final_report_quality,
            use_container_width=True,
            hide_index=True
        )

        # -------------------------------------------------
        # ANOMALY SUMMARY
        # -------------------------------------------------

        st.markdown("### 🚨 Anomaly Summary")

        anomaly_summary_report = pd.DataFrame({

            "Metric": [
                "Anomalies Before Cleaning",
                "Anomalies After Cleaning",
                "Anomalies Resolved"
            ],

            "Count": [
                before_anomaly_count,
                after_anomaly_count,
                max(
                    0,
                    before_anomaly_count
                    - after_anomaly_count
                )
            ]
        })

        st.dataframe(
            anomaly_summary_report,
            use_container_width=True,
            hide_index=True
        )

        # -------------------------------------------------
        # FINAL STATUS
        # -------------------------------------------------

        st.markdown("### 🏆 Final Dataset Status")

        if cleaned_quality_score >= 90:

            st.success(
                f"✅ Dataset quality is excellent — "
                f"{cleaned_quality_score:.2f}/100"
            )

        elif cleaned_quality_score >= 75:

            st.info(
                f"ℹ️ Dataset quality is good — "
                f"{cleaned_quality_score:.2f}/100"
            )

        elif cleaned_quality_score >= 50:

            st.warning(
                f"⚠️ Dataset still needs improvement — "
                f"{cleaned_quality_score:.2f}/100"
            )

        else:

            st.error(
                f"❌ Dataset quality remains poor — "
                f"{cleaned_quality_score:.2f}/100"
            )


    # =====================================================
    # REPORTS & DOWNLOADS
    # =====================================================

    section_heading(
        "📦 Reports & Downloads",
        "Download the cleaned dataset and generated quality reports."
    )

    cleaned_csv = cleaned_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download Cleaned Dataset",
        data=cleaned_csv,
        file_name="cleaned_dataset.csv",
        mime="text/csv",
        use_container_width=True
    )


    # -----------------------------------------------------
    # FINAL STATUS
    # -----------------------------------------------------

    if cleaned_quality_score >= 90:

        final_status = "Excellent"

    elif cleaned_quality_score >= 75:

        final_status = "Good"

    elif cleaned_quality_score >= 50:

        final_status = "Needs Improvement"

    else:

        final_status = "Poor"

    st.session_state["final_status"] = final_status


    # -----------------------------------------------------
    # GENERATE QUALITY REPORT
    # -----------------------------------------------------

    st.markdown(
        """
        <style>

        /* All normal Streamlit buttons */
        div[data-testid="stButton"] button,
        div[data-testid="stButton"] button p {

            font-size: 17px !important;
            font-weight: 600 !important;

        }

        /* Download buttons */
        div[data-testid="stDownloadButton"] button,
        div[data-testid="stDownloadButton"] button p {

            font-size: 17px !important;
            font-weight: 600 !important;

        }

        </style>
        """,
        unsafe_allow_html=True
    )


    if st.button(
        "📊 Generate Quality Report",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "Generating quality report..."
        ):

            quality_report = generate_quality_report(
                dataset_name,
                len(cleaned_df),
                len(cleaned_df.columns),
                overall_quality_score,
                cleaned_quality_score,
                before_anomaly_count,
                after_anomaly_count,
                cleaning_iterations,
                cleaning_report,
                recommendation_df,
                final_status
            )


            quality_report_html = (
                generate_quality_report_html(
                    dataset_name,
                    len(cleaned_df),
                    len(cleaned_df.columns),
                    overall_quality_score,
                    cleaned_quality_score,
                    before_anomaly_count,
                    after_anomaly_count,
                    cleaning_iterations,
                    cleaning_report,
                    recommendation_df,
                    final_status
                )
            )

            st.session_state[
                "quality_report"
            ] = quality_report

            st.session_state[
                "quality_report_html"
            ] = quality_report_html

            st.success(
                "✅ Quality report generated successfully."
            )



    # -----------------------------------------------------
    # REPORT PREVIEW & DOWNLOAD
    # -----------------------------------------------------

    if "quality_report" in st.session_state:

        st.markdown("### 📄 Quality Report Preview")

        st.text(
            st.session_state["quality_report"]
        )

        st.download_button(
            "⬇️ Download Quality Report",
            data=st.session_state["quality_report"],
            file_name="datly_quality_report.txt",
            mime="text/plain",
            use_container_width=True
        )


    if "quality_report_html" in st.session_state:

        st.download_button(
            "🌐 Download HTML Report",
            data=st.session_state[
                "quality_report_html"
            ],
            file_name="datly_quality_report.html",
            mime="text/html",
            use_container_width=True
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="datly-footer">
        <strong>Datly</strong> · Smart Data Quality & Anomaly Detection
        <br>
        Designed & Developed by Satyam Raj · 2026
        <br>
    </div>
    """,
    unsafe_allow_html=True
)
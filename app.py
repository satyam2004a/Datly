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


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f8fafc;
    }

    .hero {
        padding: 2rem 2rem 1.5rem 2rem;
        border-radius: 18px;
        background: linear-gradient(
            135deg,
            #0f172a,
            #1e293b
        );
        color: white;
        margin-bottom: 2rem;
    }

    .hero h1 {
        font-size: 3rem;
        margin-bottom: 0.3rem;
    }

    .hero .tagline {
        font-size: 1.55rem;
        color: #cbd5e1;
        margin-bottom: 0.5rem;
    }

    .hero p {
        font-size: 1.20rem;
        color: #cbd5e1;
    }

    .section-title {
        font-size: 1.7rem;
        font-weight: 700;
        margin-top: 2rem;
        margin-bottom: 0.2rem;
    }

    .section-description {
        color: #64748b;
        margin-bottom: 1rem;
    }

    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
        text-align: center;
    }

    .metric-card h3 {
        margin: 0;
        color: #64748b;
        font-size: 0.9rem;
    }

    .metric-card h2 {
        margin: 0.4rem 0 0 0;
        font-size: 1.8rem;
        color: #0f172a;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
    }

    .stDownloadButton > button {
        border-radius: 10px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def section_heading(title, description):
    st.markdown(
        f"""
        <div class="section-title">{title}</div>
        <div class="section-description">{description}</div>
        """,
        unsafe_allow_html=True
    )


def metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <h3>{label}</h3>
            <h2>{value}</h2>
        </div>
        """,
        unsafe_allow_html=True
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
# HERO SECTION
# =========================================================

st.markdown(
    """
    <div style="
        padding: 30px;
        border-radius: 18px;
        background: #0f172a;
        color: white;
    ">
        <h1>🛡️ Datly</h1>
        <h3>Smart Data Quality &amp; Anomaly Detection</h3>
        <p>Clean data. Detect anomalies. Boost your projects with cleaner, smarter data.</p>
    </div>
    """,
    unsafe_allow_html=True
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
    type=["csv"]
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

    # Clear old dataset-specific state.
    st.session_state.clear()

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
            f"Please check the file form        at. "
            f"Details: {error}"
        )

        st.stop()

    if df.shape[1] == 0:

        st.error(
            "❌ The uploaded dataset has no columns."
        )

        st.stop()

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
    "Quick preview of the uploaded dataset."
)


if df.empty:

    st.warning(
        "⚠️ The dataset contains columns but no rows."
    )

else:

    display_dataframe_preview(df)


# =========================================================
# DATASET PROFILE
# =========================================================

section_heading(
    "🔎 Dataset Profile",
    "Understand the structure and characteristics of your dataset."
)


profile_col1, profile_col2, profile_col3, profile_col4 = (
    st.columns(4)
)


with profile_col1:

    metric_card(
        "Rows",
        profile["rows"]
    )


with profile_col2:

    metric_card(
        "Columns",
        profile["columns"]
    )


with profile_col3:

    metric_card(
        "Numerical",
        len(
            profile["numerical_columns"]
        )
    )


with profile_col4:

    metric_card(
        "Categorical",
        len(
            profile["categorical_columns"]
        )
    )


# =========================================================
# COLUMN PROFILE
# =========================================================

st.subheader(
    "📋 Column Profile"
)

display_dataframe_preview(
    profile["column_profile"]
)


# =========================================================
# NUMERICAL STATISTICS
# =========================================================

if profile["numerical_statistics"]:

    st.subheader(
        "📐 Numerical Statistics"
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

section_heading(
    "🚨 Anomaly Detection",
    "Detect unusual or suspicious values in the dataset."
)


st.dataframe(
    anomaly_summary_df,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# INITIAL ANOMALY REPORT
# =========================================================

if anomaly_report.empty:

    st.success(
        "✅ No anomalies detected!"
    )

else:

    st.subheader(
        "🔎 Detected Anomalies"
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

section_heading(
    "📊 Initial Quality Score",
    "Measure the overall quality of the uploaded dataset."
)


score_col1, score_col2, score_col3, score_col4, score_col5 = (
    st.columns(5)
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
        "Anomaly Score",
        f"{anomaly_quality_score:.2f}"
    )


with score_col5:

    metric_card(
        "Overall Score",
        f"{overall_quality_score:.2f}"
    )


st.progress(
    min(
        max(
            overall_quality_score / 100,
            0.0
        ),
        1.0
    ),
    text=(
        f"Overall Quality Score: "
        f"{overall_quality_score:.2f}/100"
    )
)


st.info(
    f"📌 Dataset Status: **{quality_status}**"
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
    type="primary"
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
        text="🧹 Preparing dataset cleaning — 0%"
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
            "✅ Dataset is already clean."
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

            if (current_anomaly_report.empty
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
                    f"⚠️ Maximum of "
                    f"{max_iterations} cleaning iterations "
                    f"reached. "
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
            f"anomaly/anomalies remain."
        )


# =========================================================
# SHOW CLEANED DATASET
# =========================================================

if "cleaned_df" in st.session_state:

    cleaned_df = st.session_state[
        "cleaned_df"
    ]

    cleaning_report = st.session_state[
        "cleaning_report"
    ]

    cleaned_quality_score = st.session_state[
        "cleaned_quality_score"
    ]

    before_anomaly_count = st.session_state[
        "before_anomaly_count"
    ]

    after_anomaly_count = st.session_state[
        "after_anomaly_count"
    ]

    cleaning_iterations = st.session_state[
        "cleaning_iterations"
    ]

    final_anomaly_report = st.session_state[
        "final_anomaly_report"
    ]

    final_completeness_score = st.session_state[
        "final_completeness_score"
    ]

    final_validity_score = st.session_state[
        "final_validity_score"
    ]

    final_uniqueness_score = st.session_state[
        "final_uniqueness_score"
    ]

    final_anomaly_quality_score = st.session_state[
        "final_anomaly_quality_score"
    ]

    final_quality_results = st.session_state[
        "cleaned_quality_results"
    ]


    # =====================================================
    # CLEANED DATASET
    # =====================================================

    section_heading(
        "✨ Cleaned Dataset",
        "Dataset after Datly's automated cleaning process."
    )

    display_dataframe_preview(
        cleaned_df
    )


    # =====================================================
    # BEFORE VS AFTER
    # =====================================================

    section_heading(
        "📈 Before vs After Cleaning",
        "Comparison of dataset quality before and after cleaning."
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

    comparison_df = pd.DataFrame(
        comparison_data
    )

    comparison_df["Improvement"] = (
        comparison_df["After Cleaning"]
        - comparison_df["Before Cleaning"]
    )

    comparison_df["Before Cleaning"] = (
        comparison_df["Before Cleaning"]
        .round(2)
    )

    comparison_df["After Cleaning"] = (
        comparison_df["After Cleaning"]
        .round(2)
    )

    comparison_df["Improvement"] = (
        comparison_df["Improvement"]
        .round(2)
    )

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )


    # =====================================================
    # OVERALL IMPROVEMENT
    # =====================================================

    overall_improvement = (
        cleaned_quality_score
        - overall_quality_score
    )

    if overall_improvement > 0:

        st.success(
            f"📈 Overall dataset quality improved by "
            f"{overall_improvement:.2f} points."
        )

    elif overall_improvement == 0:

        st.info(
            "ℹ️ Overall quality score remained unchanged "
            "after cleaning."
        )

    else:

        st.warning(
            f"⚠️ Overall quality score changed by "
            f"{overall_improvement:.2f} points."
        )


    # =====================================================
    # ANOMALY RECHECK
    # =====================================================

    section_heading(
        "🔎 Anomaly Recheck",
        "Final anomaly analysis after cleaning."
    )

    anomaly_col1, anomaly_col2 = (
        st.columns(2)
    )

    with anomaly_col1:

        metric_card(
            "Anomalies Before",
            before_anomaly_count
        )

    with anomaly_col2:

        metric_card(
            "Anomalies After",
            after_anomaly_count
        )

    if final_anomaly_report.empty:

        st.success(
            "✅ No anomalies remain after cleaning."
        )

    else:

        st.warning(
            f"⚠️ {after_anomaly_count} "
            f"anomaly/anomalies still require review."
        )

        display_dataframe_preview(
            final_anomaly_report,
            MAX_ANOMALY_DISPLAY_ROWS
        )


    # =====================================================
    # CLEANING PROCESS
    # =====================================================

    section_heading(
        "🔄 Cleaning Process",
        "Summary of the iterative cleaning process."
    )

    process_col1, process_col2 = (
        st.columns(2)
    )

    with process_col1:

        metric_card(
            "Iterations",
            cleaning_iterations
        )

    with process_col2:

        if after_anomaly_count == 0:

            metric_card(
                "Final Status",
                "Clean"
            )

        else:

            metric_card(
                "Final Status",
                "Needs Attention"
            )


    # =====================================================
    # SMART RECOMMENDATIONS
    # =====================================================

    section_heading(
        "💡 Smart Recommendations",
        "Actionable recommendations based on the cleaned dataset."
    )


    # -----------------------------------------------------
    # GENERATE RECOMMENDATIONS ONLY ONCE
    # -----------------------------------------------------

    if "recommendations" not in st.session_state:

        recommendation_df = (
            generate_recommendations(
                final_quality_results,
                final_anomaly_report,
                cleaned_quality_score
            )
        )

        st.session_state[
            "recommendations"
        ] = recommendation_df

    else:

        recommendation_df = (
            st.session_state[
                "recommendations"
            ]
        )


    if recommendation_df.empty:

        st.success(
            "✅ No additional recommendations. "
            "The dataset is in good condition."
        )

    else:

        st.dataframe(
            recommendation_df,
            use_container_width=True,
            hide_index=True
        )


    # =====================================================
    # CLEANING REPORT
    # =====================================================

    section_heading(
        "📝 Cleaning Report",
        "Detailed record of actions performed during cleaning."
    )

    if cleaning_report.empty:

        st.info(
            "ℹ️ No cleaning actions were required."
        )

    else:

        display_dataframe_preview(
            cleaning_report
        )


    # =====================================================
    # DOWNLOAD CLEANED CSV
    # =====================================================

    section_heading(
        "⬇️ Download Cleaned Dataset",
        "Download the cleaned dataset for further use."
    )


    # Cache generated CSV bytes in session state.
    if "cleaned_csv_data" not in st.session_state:

        st.session_state[
            "cleaned_csv_data"
        ] = (
            cleaned_df
            .to_csv(index=False)
            .encode("utf-8")
        )

    csv_data = st.session_state[
        "cleaned_csv_data"
    ]


    st.download_button(
        label="⬇️ Download Cleaned CSV",
        data=csv_data,
        file_name="datly_cleaned_dataset.csv",
        mime="text/csv"
    )


    # =====================================================
    # PROFESSIONAL QUALITY REPORT
    # =====================================================

    section_heading(
        "📄 Professional Quality Report",
        "Generate a complete report containing dataset statistics, quality scores, anomaly analysis, cleaning actions, recommendations, and final status."
    )


    if st.button(
        "📄 Generate Quality Report",
        type="primary"
    ):

        # -------------------------------------------------
        # FINAL STATUS
        # -------------------------------------------------

        has_quality_issues = (
            recommendation_df is not None
            and not recommendation_df.empty
            and "Issue" in recommendation_df.columns
            and not (
                recommendation_df["Issue"]
                .astype(str)
                .str.strip()
                .eq("Dataset Ready")
                .all()
            )
        )

        if (
            after_anomaly_count == 0
            and not has_quality_issues
        ):

            final_status = "Dataset Ready"

        else:

            final_status = "Needs Attention"


        # -------------------------------------------------
        # GENERATE TEXT REPORT
        # -------------------------------------------------

        quality_report = generate_quality_report(

            dataset_name=dataset_name,

            rows=len(df),

            columns=len(df.columns),

            initial_quality_score=overall_quality_score,

            final_quality_score=cleaned_quality_score,

            initial_anomaly_count=before_anomaly_count,

            final_anomaly_count=after_anomaly_count,

            cleaning_iterations=cleaning_iterations,

            cleaning_report=cleaning_report,

            recommendations=recommendation_df,

            final_status=final_status
        )


        # -------------------------------------------------
        # GENERATE HTML REPORT
        # -------------------------------------------------

        quality_report_html = (
            generate_quality_report_html(

                dataset_name=dataset_name,

                rows=len(df),

                columns=len(df.columns),

                initial_quality_score=overall_quality_score,

                final_quality_score=cleaned_quality_score,

                initial_anomaly_count=before_anomaly_count,

                final_anomaly_count=after_anomaly_count,

                cleaning_iterations=cleaning_iterations,

                cleaning_report=cleaning_report,

                recommendations=recommendation_df,

                final_status=final_status
            )
        )


        # -------------------------------------------------
        # SAVE REPORTS
        # -------------------------------------------------

        st.session_state[
            "quality_report"
        ] = quality_report

        st.session_state[
            "quality_report_html"
        ] = quality_report_html

        st.session_state[
            "final_status"
        ] = final_status

        st.success(
            "✅ Professional quality reports generated successfully!"
        )


    # =====================================================
    # SHOW GENERATED REPORTS
    # =====================================================

    if (
        "quality_report" in st.session_state
        and
        "quality_report_html" in st.session_state
    ):

        quality_report = st.session_state[
            "quality_report"
        ]

        quality_report_html = st.session_state[
            "quality_report_html"
        ]


        # -------------------------------------------------
        # TEXT REPORT PREVIEW
        # -------------------------------------------------

        st.subheader(
            "📋 Quality Report Preview"
        )

        st.text_area(
            "Professional Text Report",
            quality_report,
            height=500
        )


        # -------------------------------------------------
        # DOWNLOAD TEXT REPORT
        # -------------------------------------------------

        st.download_button(
            label="⬇️ Download TXT Report",
            data=quality_report,
            file_name="datly_quality_report.txt",
            mime="text/plain"
        )


        # -------------------------------------------------
        # HTML REPORT PREVIEW
        # -------------------------------------------------

        st.subheader(
            "🌐 Professional HTML Report"
        )

        st.components.v1.html(
            quality_report_html,
            height=700,
            scrolling=True
        )


        # -------------------------------------------------
        # DOWNLOAD HTML REPORT
        # -------------------------------------------------

        st.download_button(
            label="🌐 Download Professional HTML Report",
            data=quality_report_html,
            file_name="datly_quality_report.html",
            mime="text/html"
        )

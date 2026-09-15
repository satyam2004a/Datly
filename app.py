import hashlib

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
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Datly | Smart Data Quality",
    page_icon="🛡️",
    layout="wide"
)


# =========================================================
# PROFESSIONAL UI STYLING
# =========================================================

st.markdown(
    """
    <style>

    /* Main application spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* Hero section */
    .datly-hero {
        padding: 1.6rem 1.8rem;
        margin-bottom: 1.8rem;
        border: 1px solid rgba(128, 128, 128, 0.22);
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(128, 128, 128, 0.08), rgba(128, 128, 128, 0.03));
    }

    .datly-hero-title {
        font-size: 2.5rem;
        font-weight: 750;
        margin: 0;
        line-height: 1.15;
    }

    .datly-hero-subtitle {
        font-size: 1.15rem;
        font-weight: 600;
        margin-top: 0.45rem;
        margin-bottom: 0.35rem;
        opacity: 0.9;
    }

    .datly-hero-tagline {
        font-size: 1rem;
        margin: 0;
        opacity: 0.72;
    }

    /* Section headings */
    .datly-section {
        margin-top: 2.2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.65rem;
        border-bottom: 1px solid rgba(128, 128, 128, 0.20);
    }

    .datly-section-title {
        font-size: 1.45rem;
        font-weight: 700;
        margin: 0;
    }

    .datly-section-description {
        font-size: 0.92rem;
        margin: 0.25rem 0 0;
        opacity: 0.68;
    }

    /* Metric cards */
    .datly-metric-card {
        padding: 1rem 1.05rem;
        min-height: 105px;
        border: 1px solid rgba(128, 128, 128, 0.22);
        border-radius: 14px;
        background: rgba(128, 128, 128, 0.045);
        margin-bottom: 0.7rem;
    }

    .datly-metric-label {
        font-size: 0.82rem;
        font-weight: 600;
        opacity: 0.68;
        margin-bottom: 0.35rem;
    }

    .datly-metric-value {
        font-size: 1.55rem;
        font-weight: 750;
        line-height: 1.2;
    }

    /* Status card */
    .datly-status-card {
        padding: 1rem 1.2rem;
        border: 1px solid rgba(128, 128, 128, 0.22);
        border-radius: 14px;
        background: rgba(128, 128, 128, 0.045);
        margin: 0.8rem 0 1rem;
    }

    .datly-status-label {
        font-size: 0.82rem;
        font-weight: 600;
        opacity: 0.68;
    }

    .datly-status-value {
        font-size: 1.2rem;
        font-weight: 700;
        margin-top: 0.25rem;
    }

    /* Buttons */
    div.stButton > button, div.stDownloadButton > button {
        border-radius: 10px;
        font-weight: 650;
        min-height: 2.7rem;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        border-radius: 14px;
    }

    /* Tables */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Reduce excessive vertical gaps around widgets */
    div[data-testid="stVerticalBlock"] > div:has(> div.stMarkdown) {
        margin-bottom: 0.15rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# UI HELPERS
# =========================================================

def section_heading(title, description):
    """Display a consistent professional section heading."""

    st.markdown(
        f"""
        <div class="datly-section">
            <div class="datly-section-title">{title}</div>
            <div class="datly-section-description">{description}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def metric_card(label, value):
    """Display a compact professional metric card."""

    st.markdown(
        f"""
        <div class="datly-metric-card">
            <div class="datly-metric-label">{label}</div>
            <div class="datly-metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# TITLE
# =========================================================

st.markdown(
    """
    <div class="datly-hero">
        <div class="datly-hero-title">🛡️ Datly</div>
        <div class="datly-hero-subtitle">Smart Data Quality &amp; Anomaly Detection</div>
        <div class="datly-hero-tagline">Clean data. Detect anomalies. Boost your projects with cleaner, smarter data.</div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# CSV UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)

st.caption(
    "Supported format: CSV. Datly will profile, validate, detect anomalies, clean, score, and report on your dataset."
)


if uploaded_file is not None:

    # -----------------------------------------------------
    # Detect a new uploaded file
    # -----------------------------------------------------

    file_bytes = uploaded_file.getvalue()

    file_id = hashlib.md5(
        file_bytes
    ).hexdigest()

    if st.session_state.get(
        "file_id"
    ) != file_id:

        st.session_state.clear()

        st.session_state[
            "file_id"
        ] = file_id

    # -----------------------------------------------------
    # Read CSV safely
    # -----------------------------------------------------

    try:

        df = pd.read_csv(
            uploaded_file
        )

    except Exception as error:

        st.error(
            "❌ The CSV file could not be read. "
            f"Please check the file format. "
            f"Details: {error}"
        )

        st.stop()

    if df.shape[1] == 0:

        st.error(
            "❌ The uploaded dataset has no columns."
        )

        st.stop()

    st.success(
        "CSV file uploaded successfully!"
    )


    # =====================================================
    # DATASET PREVIEW
    # =====================================================

    # =====================================================
    # DATASET PREVIEW
# =====================================================

    section_heading(
        "📋 Dataset Preview",
        "Take a quick look at the uploaded records before quality analysis begins."
    )

    if df.empty:    

        st.error(
            "❌ The uploaded dataset contains no rows."
        )

        st.warning(
            "⚠️ Please upload a CSV file containing at least "
            "one data row before running quality analysis."
        )

        st.info(
             "💡 The dataset must contain actual records. "
            "A CSV containing only column headers cannot be "
            "analyzed for data quality, anomalies, or cleaning."
        )

        st.stop()

    else:

        st.dataframe(
             df,
            use_container_width=True
        )


    # =====================================================
    # DATASET PROFILE
    # =====================================================

    profile = profile_dataset(
        df
    )

    section_heading(
        "📊 Dataset Profile",
        "Understand the structure, data types, and key characteristics of your dataset."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card(
            "Rows",
            f"{profile['rows']:,}"
        )

    with col2:
        metric_card(
            "Columns",
            f"{profile['columns']:,}"
        )

    with col3:
        metric_card(
            "Numerical Columns",
            f"{len(profile['numerical_columns']):,}"
        )

    with col4:
        metric_card(
            "Categorical Columns",
            f"{len(profile['categorical_columns']):,}"
        )


    st.subheader(
        "📋 Column Profile"
    )

    st.dataframe(
        profile["column_profile"],
        use_container_width=True,
        hide_index=True
    )


    # =====================================================
    # NUMERICAL STATISTICS
    # =====================================================

    if profile[
        "numerical_statistics"
    ]:

        st.subheader(
            "📐 Numerical Statistics"
        )

        statistics_df = (
            pd.DataFrame(
                profile[
                    "numerical_statistics"
                ]
            )
            .T
            .reset_index()
            .rename(
                columns={
                    "index": "Column"
                }
            )
        )

        st.dataframe(
            statistics_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No numerical columns are available "
            "for statistical analysis."
        )


    # =====================================================
    # DATA QUALITY RULES
    # =====================================================

    section_heading(
        "🛡️ Data Quality Rules",
        "Check completeness, validity, uniqueness, and other data-quality conditions."
    )

    quality_results = run_quality_rules(
        df
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


    # =====================================================
    # INITIAL ANOMALY DETECTION
    # =====================================================

    section_heading(
        "🚨 Anomaly Detection",
        "Identify unusual values and potential issues that may affect your analysis."
    )

    results = detect_anomalies(
        df
    )

    summary = anomaly_summary(
        results
    )

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )


    # =====================================================
    # INITIAL ANOMALY REPORT
    # =====================================================

    anomaly_report = combined_anomaly_report(
        results
    )

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

        st.dataframe(
            anomaly_report[
                display_columns
            ],
            use_container_width=True,
            hide_index=True
        )


    # =====================================================
    # ENHANCED DATASET QUALITY SCORE
    # =====================================================

    raw_anomaly_penalty = calculate_anomaly_score(
        anomaly_report,
        len(df)
    )


    completeness_score = calculate_completeness_score(
        quality_results
    )

    validity_score = calculate_validity_score(
        quality_results
    )

    uniqueness_score = calculate_uniqueness_score(
        quality_results
    )

    anomaly_quality_score = calculate_anomaly_quality_score(
        raw_anomaly_penalty
    )

    overall_quality_score = calculate_overall_quality_score(
        completeness_score,
        validity_score,
        uniqueness_score,
        anomaly_quality_score
    )

    quality_status = get_quality_status(
        overall_quality_score
    )


    # =====================================================
    # DATASET QUALITY
    # =====================================================

    section_heading(
        "📈 Dataset Quality",
        "Measure the overall health of your data using multiple quality dimensions."
    )

    score_col1, score_col2, score_col3, score_col4 = (
        st.columns(4)
    )

    with score_col1:
        metric_card(
            "Completeness",
            f"{completeness_score:.2f} / 100"
        )

    with score_col2:
        metric_card(
            "Validity",
            f"{validity_score:.2f} / 100"
        )

    with score_col3:
        metric_card(
            "Uniqueness",
            f"{uniqueness_score:.2f} / 100"
        )

    with score_col4:
        metric_card(
            "Anomaly Quality",
            f"{anomaly_quality_score:.2f} / 100"
        )

    overall_col1, overall_col2 = st.columns([2, 1])

    with overall_col1:
        metric_card(
            "Overall Quality Score",
            f"{overall_quality_score:.2f} / 100"
        )
        st.progress(
            max(0.0, min(1.0, overall_quality_score / 100)),
            text="Overall dataset quality"
        )

    with overall_col2:
        st.markdown(
            f"""
            <div class="datly-status-card">
                <div class="datly-status-label">Quality Status</div>
                <div class="datly-status-value">{quality_status}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


    # =====================================================
    # CLEAN DATASET
    # =====================================================

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

        # -------------------------------------------------
        # ITERATIVE CLEANING
        # -------------------------------------------------

        max_iterations = 5

        for iteration in range(
            max_iterations
        ):

            if current_anomaly_report.empty:

                break

            cleaned_df, cleaning_report = (
                clean_dataset(
                    current_df,
                    current_anomaly_report
                )
            )

            iterations_completed += 1

            if not cleaning_report.empty:

                if "iteration" in cleaning_report.columns:
                    cleaning_report["iteration"] = iteration + 1
                else:
                    cleaning_report.insert(
                        0,
                        "iteration",
                        iteration + 1
                    )

                all_cleaning_reports.append(
                    cleaning_report
                )

            cleaned_results = detect_anomalies(
                cleaned_df
            )

            new_anomaly_report = (
                combined_anomaly_report(
                    cleaned_results
                )
            )

            # -------------------------------------------------
            # UPDATE CLEANING STATE
            # -------------------------------------------------

            current_df = cleaned_df

            current_anomaly_report = (
                new_anomaly_report
            )

            # Stop only when no anomalies remain.
            # Otherwise, continue to the next cleaning
            # iteration until max_iterations is reached.

            if current_anomaly_report.empty:

                break


        # -------------------------------------------------
        # FINAL DATA
        # -------------------------------------------------

        final_df = current_df

        final_anomaly_report = (
            current_anomaly_report
        )


        # -------------------------------------------------
        # FINAL SCORE
        # -------------------------------------------------

        final_anomaly_penalty = (
            calculate_anomaly_score(
                final_anomaly_report,
                len(final_df)
            )
        )

        final_quality_results = (
            run_quality_rules(
                final_df
            )
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


        # -------------------------------------------------
        # COMBINE CLEANING REPORTS
        # -------------------------------------------------

        if all_cleaning_reports:

            final_cleaning_report = pd.concat(
                all_cleaning_reports,
                ignore_index=True
            )

        else:

            final_cleaning_report = (
                pd.DataFrame()
            )


        # -------------------------------------------------
        # SAVE RESULTS
        # -------------------------------------------------

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
        ] = len(
            anomaly_report
        )

        st.session_state[
            "after_anomaly_count"
        ] = len(
            final_anomaly_report
        )

        st.session_state[
            "cleaning_iterations"
        ] = iterations_completed

        st.session_state[
            "cleaned_quality_results"
        ] = final_quality_results

        st.session_state[
            "final_anomaly_report"
        ] = final_anomaly_report


        # -------------------------------------------------
        # SUCCESS MESSAGE
        # -------------------------------------------------

        if final_anomaly_report.empty:

            st.success(
                "✅ Dataset successfully cleaned! "
                "No anomalies remain."
            )

        else:

            st.warning(
                f"⚠️ Cleaning completed, but "
                f"{len(final_anomaly_report)} "
                f"anomaly/anomalies remain."
            )


    # =====================================================
    # SHOW CLEANED DATASET
    # =====================================================

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


        # =================================================
        # CLEANED DATASET
        # =================================================

        st.subheader(
            "✨ Cleaned Dataset"
        )

        st.caption(
            "The dataset after the iterative cleaning process."
        )

        st.dataframe(
            cleaned_df,
            use_container_width=True
        )


        # =================================================
        # PROFESSIONAL BEFORE VS AFTER QUALITY COMPARISON
        # =================================================

        st.subheader(
            "📊 Before vs After Cleaning"
        )

        st.write(
            "Comparison of dataset quality before and "
            "after the cleaning process."
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


        # =================================================
        # OVERALL IMPROVEMENT
        # =================================================

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


        # =================================================
        # ANOMALY RECHECK
        # =================================================

        st.subheader(
            "🔄 Anomaly Recheck"
        )

        st.caption(
            "Verify whether the cleaning process reduced the detected anomalies."
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Anomalies Before",
                before_anomaly_count
            )

        with col2:

            anomaly_change = (
                after_anomaly_count
                - before_anomaly_count
            )

            st.metric(
                "Anomalies After",
                after_anomaly_count,
                delta=anomaly_change,
                delta_color="inverse"
            )


        # =================================================
        # ITERATION INFORMATION
        # =================================================

        st.subheader(
            "🔁 Cleaning Process"
        )

        st.caption(
            "A quick summary of the iterative cleaning passes performed by Datly."
        )

        st.info(
            f"The platform performed "
            f"{cleaning_iterations} cleaning "
            f"iteration(s) and rechecked the "
            f"dataset after each pass."
        )


        # =================================================
        # FINAL STATUS
        # =================================================

        if after_anomaly_count == 0:

            st.success(
                "🎉 Final dataset quality check passed. "
                "No anomalies remain."
            )

        else:

            st.warning(
                f"⚠️ {after_anomaly_count} "
                f"anomaly/anomalies still require "
                f"review."
            )


        # =================================================
        # SMART RECOMMENDATIONS
        # =================================================

        st.subheader(
            "💡 Smart Recommendations"
        )

        st.caption(
            "Practical next steps based on the final quality analysis."
        )

        recommendation_df = generate_recommendations(
            st.session_state[
                "cleaned_quality_results"
            ],
            st.session_state[
                "final_anomaly_report"
            ],
            cleaned_quality_score
        )

        st.session_state[
            "recommendations"
        ] = recommendation_df

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


        # =================================================
        # CLEANING REPORT
        # =================================================

        st.subheader(
            "📝 Cleaning Report"
        )

        st.caption(
            "A record of the changes applied during dataset cleaning."
        )

        if cleaning_report.empty:

            st.info(
                "No cleaning actions were required."
            )

        else:

            st.dataframe(
                cleaning_report,
                use_container_width=True,
                hide_index=True
            )


        # =================================================
        # DOWNLOAD CLEANED CSV
        # =================================================

        st.subheader(
            "⬇️ Download Cleaned Dataset"
        )

        st.caption(
            "Export the cleaned dataset for use in analysis, visualization, or machine learning."
        )

        csv_data = (
            cleaned_df
            .to_csv(
                index=False
            )
            .encode("utf-8")
        )

        st.download_button(
            label="⬇️ Download Cleaned CSV",
            data=csv_data,
            file_name="cleaned_dataset.csv",
            mime="text/csv"
        )

        # =================================================
        # PROFESSIONAL QUALITY REPORT
        # =================================================

        st.subheader(
            "📄 Professional Quality Report"
        )

        st.write(
            "Generate a complete quality report containing "
            "dataset statistics, quality scores, anomaly "
            "analysis, cleaning actions, recommendations, "
            "and final status."
        )

        if st.button(
            "📄 Generate Quality Report",
            type="primary"
        ):

            # -------------------------------------------------
            # DETERMINE FINAL DATASET STATUS
            # -------------------------------------------------

            recommendations = st.session_state.get(
                "recommendations",
                pd.DataFrame()
            )

            has_quality_issues = (
                recommendations is not None
                and not recommendations.empty
                and not (
                    recommendations["Issue"]
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

                dataset_name=uploaded_file.name,

                rows=len(df),

                columns=len(df.columns),

                initial_quality_score=overall_quality_score,

                final_quality_score=cleaned_quality_score,

                initial_anomaly_count=before_anomaly_count,

                final_anomaly_count=after_anomaly_count,

                cleaning_iterations=cleaning_iterations,

                cleaning_report=cleaning_report,

                recommendations=st.session_state[
                    "recommendations"
                ],

                final_status=final_status
            )

            # -------------------------------------------------
            # GENERATE HTML REPORT
            # -------------------------------------------------

            quality_report_html = (
                generate_quality_report_html(

                    dataset_name=uploaded_file.name,

                    rows=len(df),

                    columns=len(df.columns),

                    initial_quality_score=overall_quality_score,

                    final_quality_score=cleaned_quality_score,

                    initial_anomaly_count=before_anomaly_count,

                    final_anomaly_count=after_anomaly_count,

                    cleaning_iterations=cleaning_iterations,

                    cleaning_report=cleaning_report,

                    recommendations=st.session_state[
                        "recommendations"
                    ],

                    final_status=final_status
                )
            )

            # -------------------------------------------------
            # SAVE REPORTS IN SESSION STATE
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

        # -------------------------------------------------
        # SHOW GENERATED REPORTS
        # -------------------------------------------------

        if (
            "quality_report" in st.session_state
            and "quality_report_html" in st.session_state
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

            st.caption(
                "Review the generated report before downloading it."
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

                file_name="quality_report.txt",

                mime="text/plain"
            )

            # -------------------------------------------------
            # DOWNLOAD HTML REPORT
            # -------------------------------------------------

            st.download_button(

                label="🌐 Download Professional HTML Report",

                data=quality_report_html,

                file_name="quality_report.html",

                mime="text/html"
            )

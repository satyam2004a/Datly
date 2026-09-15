import streamlit as st
import pandas as pd

from src.anomaly_detection import (
    detect_anomalies,
    anomaly_summary,
    combined_anomaly_report,
    calculate_anomaly_score
)

from src.quality_score import calculate_individual_score
from src.data_cleaning import clean_dataset


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Data Quality & Anomaly Detection Platform",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title(
    "📊 Data Quality & Anomaly Detection Platform"
)

st.write(
    "Upload a CSV file to analyze your dataset."
)


# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)


# =========================================================
# MAIN APPLICATION
# =========================================================

if uploaded_file is not None:

    # -----------------------------------------------------
    # LOAD DATASET
    # -----------------------------------------------------

    df = pd.read_csv(uploaded_file)

    st.success(
        "CSV file uploaded successfully!"
    )


    # =====================================================
    # RAW DATASET
    # =====================================================

    st.subheader("📄 Raw Dataset")

    st.dataframe(
        df,
        use_container_width=True
    )


    # =====================================================
    # DATASET OVERVIEW
    # =====================================================

    st.subheader("📊 Dataset Overview")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Rows",
            df.shape[0]
        )

    with col2:

        st.metric(
            "Columns",
            df.shape[1]
        )


    # =====================================================
    # ANOMALY DETECTION
    # =====================================================

    st.subheader("🚨 Anomaly Detection")

    try:

        # Detect anomalies
        results = detect_anomalies(df)

        # Create summary
        summary = anomaly_summary(results)

        # Create combined anomaly report
        report = combined_anomaly_report(results)


        # -------------------------------------------------
        # INDIVIDUAL ANOMALY SCORES
        # -------------------------------------------------

        if not report.empty:

            report["individual_score"] = report.apply(
                lambda row: calculate_individual_score(
                    row["z_score"],
                    row["iqr_anomaly"]
                ),
                axis=1
            )


        # -------------------------------------------------
        # OVERALL ANOMALY SCORE
        # -------------------------------------------------

        anomaly_score = calculate_anomaly_score(
            report,
            len(df)
        )

        quality_score = 100 - anomaly_score

        st.metric(
            "Overall Anomaly Score",
            f"{anomaly_score:.2f}"
       )

        st.metric(
            "🎯 Data Quality Score",
            f"{quality_score:.2f}"
        )

        # -------------------------------------------------
        # ANOMALY SUMMARY
        # -------------------------------------------------

        st.subheader(
            "📋 Anomaly Summary"
        )

        st.dataframe(
            summary,
            use_container_width=True
        )


        # -------------------------------------------------
        # DETECTED ANOMALIES
        # -------------------------------------------------

        st.subheader(
            "🔎 Detected Anomalies"
        )

        if report.empty:

            st.success(
                "No anomalies detected."
            )

        else:

            st.dataframe(
                report,
                use_container_width=True
            )


    except Exception as e:

        st.error(
            f"Error while analyzing dataset: {e}"
        )

        st.stop()


    # =====================================================
    # DATA CLEANING
    # =====================================================

    st.subheader(
        "🧹 Data Cleaning"
    )

    try:

        # -------------------------------------------------
        # CLEAN DATASET + CREATE CLEANING REPORT
        # -------------------------------------------------

        cleaned_df, cleaning_report = clean_dataset(
            df,
            report
        )


        # -------------------------------------------------
        # CLEANING STATISTICS
        # -------------------------------------------------

        original_rows = len(df)

        cleaned_rows = len(cleaned_df)

        rows_removed = (
            original_rows - cleaned_rows
        )


        col3, col4, col5 = st.columns(3)

        with col3:

            st.metric(
                "Original Rows",
                original_rows
            )

        with col4:

            st.metric(
                "Cleaned Rows",
                cleaned_rows
            )

        with col5:

            st.metric(
                "Rows Removed",
                rows_removed
            )


        # -------------------------------------------------
        # CLEANED DATASET
        # -------------------------------------------------

        st.subheader(
            "✨ Cleaned Dataset"
        )

        st.dataframe(
            cleaned_df,
            use_container_width=True
        )


        # =================================================
        # CLEANING REPORT
        # =================================================

        st.subheader(
            "📝 Cleaning Report"
        )

        if cleaning_report.empty:

            st.success(
                "No cleaning actions were required."
            )

        else:

            st.dataframe(
                cleaning_report,
                use_container_width=True
            )


        # -------------------------------------------------
        # DOWNLOAD CLEANED CSV
        # -------------------------------------------------

        csv_data = cleaned_df.to_csv(
            index=False
        ).encode("utf-8")


        st.download_button(
            label="⬇️ Download Cleaned CSV",
            data=csv_data,
            file_name="cleaned_dataset.csv",
            mime="text/csv"
        )


    except Exception as e:

        st.error(
            f"Error while cleaning dataset: {e}"
        )
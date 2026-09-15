import html
import pandas as pd


# =========================================================
# FORMATTING HELPERS
# =========================================================

def _format_value(value):
    """
    Convert a value into a clean human-readable string.
    """

    if value is None:
        return "N/A"

    try:
        if pd.isna(value):
            return "N/A"
    except Exception:
        pass

    if isinstance(value, float):
        return f"{value:.2f}"

    return str(value)


# =========================================================
# CLEANING REPORT PREPARATION
# =========================================================

def _prepare_cleaning_dataframe(cleaning_report):
    """
    Prepare the cleaning report for display.
    """

    if cleaning_report is None:
        return pd.DataFrame()

    if not isinstance(cleaning_report, pd.DataFrame):

        try:
            cleaning_report = pd.DataFrame(
                cleaning_report
            )

        except Exception:
            return pd.DataFrame()

    if cleaning_report.empty:
        return pd.DataFrame()

    df = cleaning_report.copy()

    preferred_columns = [
        "iteration",
        "row_index",
        "column",
        "original_value",
        "action",
        "reason",
        "new_value"
    ]

    available_columns = [
        column
        for column in preferred_columns
        if column in df.columns
    ]

    if available_columns:
        df = df[
            available_columns
        ].copy()

    rename_map = {
        "iteration": "Iteration",
        "row_index": "Row",
        "column": "Column",
        "original_value": "Original Value",
        "action": "Action",
        "reason": "Reason",
        "new_value": "New Value"
    }

    df = df.rename(
        columns=rename_map
    )

    for column in df.columns:

        df[column] = df[column].apply(
            _format_value
        )

    return df


# =========================================================
# RECOMMENDATION REPORT PREPARATION
# =========================================================

def _prepare_recommendations_dataframe(
    recommendations
):
    """
    Prepare recommendations for display.
    """

    if recommendations is None:
        return pd.DataFrame()

    if not isinstance(
        recommendations,
        pd.DataFrame
    ):

        try:
            recommendations = pd.DataFrame(
                recommendations
            )

        except Exception:
            return pd.DataFrame()

    if recommendations.empty:
        return pd.DataFrame()

    df = recommendations.copy()

    rename_map = {
        "Priority": "Priority",
        "Issue": "Issue",
        "Recommendation": "Recommendation"
    }

    df = df.rename(
        columns=rename_map
    )

    for column in df.columns:

        df[column] = df[column].apply(
            _format_value
        )

    return df


# =========================================================
# TEXT TABLE HELPERS
# =========================================================

def _format_cleaning_actions(
    cleaning_report
):
    """
    Convert cleaning DataFrame into
    professional text-table format.
    """

    df = _prepare_cleaning_dataframe(
        cleaning_report
    )

    if df.empty:

        return (
            "No cleaning actions were required."
        )

    return df.to_string(
        index=False
    )


def _format_recommendations(
    recommendations
):
    """
    Convert recommendation DataFrame
    into readable text.
    """

    df = _prepare_recommendations_dataframe(
        recommendations
    )

    if df.empty:

        return (
            "No additional recommendations."
        )

    return df.to_string(
        index=False
    )


# =========================================================
# HTML TABLE HELPER
# =========================================================

def _dataframe_to_html_table(
    dataframe,
    empty_message
):
    """
    Convert a DataFrame into a professional
    HTML table.
    """

    if dataframe is None:

        return (
            f"<p class='empty-message'>"
            f"{html.escape(empty_message)}"
            f"</p>"
        )

    if dataframe.empty:

        return (
            f"<p class='empty-message'>"
            f"{html.escape(empty_message)}"
            f"</p>"
        )

    safe_df = dataframe.copy()

    for column in safe_df.columns:

        safe_df[column] = safe_df[
            column
        ].apply(
            lambda value: html.escape(
                str(value)
            )
        )

    return safe_df.to_html(
        index=False,
        escape=False,
        classes="data-table",
        border=0
    )


# =========================================================
# MAIN TEXT REPORT GENERATOR
# =========================================================

def generate_quality_report(
    dataset_name,
    rows,
    columns,
    initial_quality_score,
    final_quality_score,
    initial_anomaly_count,
    final_anomaly_count,
    cleaning_iterations,
    cleaning_report,
    recommendations,
    final_status
):
    """
    Generate a professional plain-text
    data quality report.
    """

    overall_improvement = (
        final_quality_score
        - initial_quality_score
    )

    anomalies_resolved = (
        initial_anomaly_count
        - final_anomaly_count
    )

    if initial_anomaly_count > 0:

        anomaly_resolution_rate = (
            anomalies_resolved
            / initial_anomaly_count
        ) * 100

    else:

        anomaly_resolution_rate = 100.0

    if final_status == "Dataset Ready":

        status_description = (
            "The dataset appears clean and is ready "
            "for downstream analysis or machine learning."
        )

    else:

        status_description = (
            "The dataset still contains unresolved "
            "quality issues and requires further attention."
        )

    cleaning_df = _prepare_cleaning_dataframe(
        cleaning_report
    )

    cleaning_action_count = len(
        cleaning_df
    )

    report = []

    report.append(
        "DATA QUALITY & ANOMALY DETECTION PLATFORM"
    )

    report.append(
        "PROFESSIONAL DATA QUALITY REPORT"
    )

    report.append("=" * 90)

    report.append("")

    report.append(
        f"Dataset Name   : {dataset_name}"
    )

    report.append(
        f"Report Status  : {final_status}"
    )

    report.append(
        "Report Purpose : Dataset quality assessment, "
        "anomaly analysis and cleaning summary"
    )

    report.append("")

    report.append("=" * 90)

    # -----------------------------------------------------
    # EXECUTIVE SUMMARY
    # -----------------------------------------------------

    report.append("")

    report.append(
        "1. EXECUTIVE SUMMARY"
    )

    report.append("-" * 90)

    report.append(
        f"The dataset '{dataset_name}' was analyzed "
        "for data-quality issues, anomalies and "
        "potential cleaning requirements."
    )

    report.append("")

    report.append(
        f"Initial Quality Score : "
        f"{initial_quality_score:.2f} / 100"
    )

    report.append(
        f"Final Quality Score   : "
        f"{final_quality_score:.2f} / 100"
    )

    report.append(
        f"Quality Improvement   : "
        f"{overall_improvement:+.2f} points"
    )

    report.append(
        f"Anomalies Resolved    : "
        f"{anomalies_resolved}"
    )

    report.append(
        f"Cleaning Actions      : "
        f"{cleaning_action_count}"
    )

    report.append(
        f"Final Dataset Status  : "
        f"{final_status}"
    )

    report.append("")

    report.append(
        status_description
    )

    # -----------------------------------------------------
    # DATASET OVERVIEW
    # -----------------------------------------------------

    report.append("")

    report.append(
        "2. DATASET OVERVIEW"
    )

    report.append("-" * 90)

    report.append(
        f"Dataset Name : {dataset_name}"
    )

    report.append(
        f"Total Rows   : {rows}"
    )

    report.append(
        f"Total Columns: {columns}"
    )

    report.append(
        f"Dataset Size : "
        f"{rows} rows × {columns} columns"
    )

    # -----------------------------------------------------
    # QUALITY SCORE
    # -----------------------------------------------------

    report.append("")

    report.append(
        "3. QUALITY SCORE SUMMARY"
    )

    report.append("-" * 90)

    report.append(
        f"{'Metric':<32} {'Value':>20}"
    )

    report.append("-" * 55)

    report.append(
        f"{'Initial Quality Score':<32}"
        f"{initial_quality_score:.2f} / 100"
    )

    report.append(
        f"{'Final Quality Score':<32}"
        f"{final_quality_score:.2f} / 100"
    )

    report.append(
        f"{'Overall Improvement':<32}"
        f"{overall_improvement:+.2f} points"
    )

    # -----------------------------------------------------
    # ANOMALY ANALYSIS
    # -----------------------------------------------------

    report.append("")

    report.append(
        "4. ANOMALY ANALYSIS"
    )

    report.append("-" * 90)

    report.append(
        f"{'Metric':<32} {'Value':>20}"
    )

    report.append("-" * 55)

    report.append(
        f"{'Anomalies Before Cleaning':<32}"
        f"{initial_anomaly_count}"
    )

    report.append(
        f"{'Anomalies After Cleaning':<32}"
        f"{final_anomaly_count}"
    )

    report.append(
        f"{'Anomalies Resolved':<32}"
        f"{anomalies_resolved}"
    )

    report.append(
        f"{'Resolution Rate':<32}"
        f"{anomaly_resolution_rate:.2f}%"
    )

    # -----------------------------------------------------
    # CLEANING PROCESS
    # -----------------------------------------------------

    report.append("")

    report.append(
        "5. CLEANING PROCESS"
    )

    report.append("-" * 90)

    report.append(
        f"Cleaning Iterations : "
        f"{cleaning_iterations}"
    )

    report.append(
        f"Cleaning Actions    : "
        f"{cleaning_action_count}"
    )

    report.append("")

    if cleaning_action_count == 0:

        report.append(
            "No cleaning actions were required."
        )

    else:

        report.append(
            "CLEANING ACTION DETAILS"
        )

        report.append("-" * 90)

        report.append(
            _format_cleaning_actions(
                cleaning_report
            )
        )

    # -----------------------------------------------------
    # RECOMMENDATIONS
    # -----------------------------------------------------

    report.append("")

    report.append(
        "6. RECOMMENDATIONS"
    )

    report.append("-" * 90)

    report.append(
        _format_recommendations(
            recommendations
        )
    )

    # -----------------------------------------------------
    # FINAL STATUS
    # -----------------------------------------------------

    report.append("")

    report.append(
        "7. FINAL DATASET STATUS"
    )

    report.append("-" * 90)

    report.append(
        f"Status : {final_status}"
    )

    report.append("")

    report.append(
        status_description
    )

    if final_anomaly_count == 0:

        report.append("")

        report.append(
            "✓ All detected anomalies were resolved."
        )

    else:

        report.append("")

        report.append(
            f"⚠ {final_anomaly_count} "
            "anomaly/anomalies remain for "
            "further review."
        )

    # -----------------------------------------------------
    # REPORT SUMMARY
    # -----------------------------------------------------

    report.append("")

    report.append(
        "8. REPORT SUMMARY"
    )

    report.append("-" * 90)

    report.append(
        f"Dataset              : {dataset_name}"
    )

    report.append(
        f"Rows                 : {rows}"
    )

    report.append(
        f"Columns              : {columns}"
    )

    report.append(
        f"Initial Score        : "
        f"{initial_quality_score:.2f} / 100"
    )

    report.append(
        f"Final Score          : "
        f"{final_quality_score:.2f} / 100"
    )

    report.append(
        f"Score Improvement    : "
        f"{overall_improvement:+.2f} points"
    )

    report.append(
        f"Initial Anomalies    : "
        f"{initial_anomaly_count}"
    )

    report.append(
        f"Final Anomalies      : "
        f"{final_anomaly_count}"
    )

    report.append(
        f"Anomalies Resolved   : "
        f"{anomalies_resolved}"
    )

    report.append(
        f"Cleaning Iterations  : "
        f"{cleaning_iterations}"
    )

    report.append(
        f"Cleaning Actions     : "
        f"{cleaning_action_count}"
    )

    report.append(
        f"Final Status         : "
        f"{final_status}"
    )

    report.append("")

    report.append("=" * 90)

    report.append(
        "Report generated by the "
        "Data Quality & Anomaly Detection Platform"
    )

    report.append(
        "Professional Reporting Module"
    )

    report.append("=" * 90)

    return "\n".join(report)


# =========================================================
# PROFESSIONAL HTML REPORT GENERATOR
# =========================================================

def generate_quality_report_html(
    dataset_name,
    rows,
    columns,
    initial_quality_score,
    final_quality_score,
    initial_anomaly_count,
    final_anomaly_count,
    cleaning_iterations,
    cleaning_report,
    recommendations,
    final_status
):
    """
    Generate a professional HTML version
    of the quality report.

    The HTML report is:
    - Browser friendly
    - Print friendly
    - PDF conversion friendly
    - Suitable for portfolio demonstrations
    """

    overall_improvement = (
        final_quality_score
        - initial_quality_score
    )

    anomalies_resolved = (
        initial_anomaly_count
        - final_anomaly_count
    )

    if initial_anomaly_count > 0:

        anomaly_resolution_rate = (
            anomalies_resolved
            / initial_anomaly_count
        ) * 100

    else:

        anomaly_resolution_rate = 100.0

    cleaning_df = _prepare_cleaning_dataframe(
        cleaning_report
    )

    recommendations_df = (
        _prepare_recommendations_dataframe(
            recommendations
        )
    )

    cleaning_action_count = len(
        cleaning_df
    )

    if final_status == "Dataset Ready":

        status_description = (
            "The dataset appears clean and is ready "
            "for downstream analysis or machine learning."
        )

    else:

        status_description = (
            "The dataset still contains unresolved "
            "quality issues and requires further attention."
        )

    # -----------------------------------------------------
    # STATUS CLASS
    # -----------------------------------------------------

    if final_status == "Dataset Ready":

        status_class = "success"

    else:

        status_class = "warning"

    # -----------------------------------------------------
    # HTML TABLES
    # -----------------------------------------------------

    cleaning_table = _dataframe_to_html_table(
        cleaning_df,
        "No cleaning actions were required."
    )

    recommendations_table = (
        _dataframe_to_html_table(
            recommendations_df,
            "No additional recommendations."
        )
    )

    # -----------------------------------------------------
    # ESCAPED DATASET NAME
    # -----------------------------------------------------

    safe_dataset_name = html.escape(
        str(dataset_name)
    )

    safe_final_status = html.escape(
        str(final_status)
    )

    # -----------------------------------------------------
    # HTML DOCUMENT
    # -----------------------------------------------------

    report_html = f"""
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>
Data Quality Report - {safe_dataset_name}
</title>

<style>

* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    padding: 0;
    background: #f4f6f8;
    color: #1f2937;
    font-family:
        Arial,
        Helvetica,
        sans-serif;
    line-height: 1.6;
}}

.container {{
    max-width: 1200px;
    margin: 40px auto;
    padding: 0 20px;
}}

.header {{
    background: #111827;
    color: white;
    padding: 35px;
    border-radius: 14px;
    margin-bottom: 25px;
}}

.header h1 {{
    margin: 0 0 8px 0;
    font-size: 30px;
}}

.header p {{
    margin: 5px 0;
    opacity: 0.9;
}}

.section {{
    background: white;
    padding: 28px;
    margin-bottom: 25px;
    border-radius: 12px;
    box-shadow:
        0 2px 8px rgba(0, 0, 0, 0.06);
}}

.section h2 {{
    margin-top: 0;
    color: #111827;
    border-bottom: 2px solid #e5e7eb;
    padding-bottom: 10px;
}}

.meta-grid {{
    display: grid;
    grid-template-columns:
        repeat(auto-fit, minmax(220px, 1fr));
    gap: 15px;
    margin-top: 20px;
}}

.meta-card {{
    background: #f9fafb;
    padding: 18px;
    border-radius: 10px;
    border: 1px solid #e5e7eb;
}}

.meta-card strong {{
    display: block;
    margin-bottom: 5px;
    color: #6b7280;
    font-size: 13px;
    text-transform: uppercase;
}}

.score-grid {{
    display: grid;
    grid-template-columns:
        repeat(auto-fit, minmax(190px, 1fr));
    gap: 18px;
}}

.score-card {{
    padding: 22px;
    background: #f9fafb;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
}}

.score-card .label {{
    color: #6b7280;
    font-size: 14px;
    margin-bottom: 8px;
}}

.score-card .value {{
    font-size: 28px;
    font-weight: bold;
    color: #111827;
}}

.highlight {{
    border-left: 5px solid #111827;
}}

.status {{
    padding: 20px;
    border-radius: 10px;
    font-size: 18px;
    font-weight: bold;
}}

.status.success {{
    background: #ecfdf5;
    color: #065f46;
    border: 1px solid #a7f3d0;
}}

.status.warning {{
    background: #fffbeb;
    color: #92400e;
    border: 1px solid #fde68a;
}}

.summary-list {{
    display: grid;
    grid-template-columns:
        repeat(auto-fit, minmax(230px, 1fr));
    gap: 12px;
}}

.summary-item {{
    background: #f9fafb;
    padding: 15px;
    border-radius: 8px;
    border: 1px solid #e5e7eb;
}}

.summary-item strong {{
    display: block;
    color: #6b7280;
    font-size: 13px;
    margin-bottom: 4px;
}}

.data-table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 15px;
    font-size: 14px;
}}

.data-table th {{
    background: #111827;
    color: white;
    padding: 12px;
    text-align: left;
}}

.data-table td {{
    padding: 11px;
    border-bottom: 1px solid #e5e7eb;
    vertical-align: top;
}}

.data-table tr:nth-child(even) {{
    background: #f9fafb;
}}

.empty-message {{
    background: #f9fafb;
    padding: 15px;
    border-radius: 8px;
    color: #6b7280;
}}

.footer {{
    text-align: center;
    color: #6b7280;
    font-size: 13px;
    padding: 25px;
}}

@media print {{

    body {{
        background: white;
    }}

    .container {{
        max-width: none;
        margin: 0;
    }}

    .section {{
        box-shadow: none;
        border: 1px solid #ddd;
        break-inside: avoid;
    }}

    .header {{
        background: #111827 !important;
        print-color-adjust: exact;
        -webkit-print-color-adjust: exact;
    }}

    .data-table th {{
        background: #111827 !important;
        print-color-adjust: exact;
        -webkit-print-color-adjust: exact;
    }}

}}

@media (max-width: 700px) {{

    .container {{
        margin: 15px auto;
        padding: 0 10px;
    }}

    .section {{
        padding: 18px;
    }}

    .header {{
        padding: 25px;
    }}

    .header h1 {{
        font-size: 24px;
    }}

    .data-table {{
        font-size: 12px;
        display: block;
        overflow-x: auto;
    }}

}}

</style>

</head>


<body>

<div class="container">


<!-- =====================================================
     HEADER
===================================================== -->

<div class="header">

    <h1>
        📊 Data Quality & Anomaly Detection Platform
    </h1>

    <p>
        Professional Data Quality Report
    </p>

    <p>
        Dataset: <strong>
            {safe_dataset_name}
        </strong>
    </p>

</div>


<!-- =====================================================
     EXECUTIVE SUMMARY
===================================================== -->

<div class="section">

    <h2>
        Executive Summary
    </h2>

    <p>
        The dataset
        <strong>{safe_dataset_name}</strong>
        was analyzed for data-quality issues,
        anomalies and potential cleaning requirements.
    </p>

    <div class="meta-grid">

        <div class="meta-card">

            <strong>Rows</strong>

            {rows}

        </div>

        <div class="meta-card">

            <strong>Columns</strong>

            {columns}

        </div>

        <div class="meta-card">

            <strong>Cleaning Iterations</strong>

            {cleaning_iterations}

        </div>

        <div class="meta-card">

            <strong>Cleaning Actions</strong>

            {cleaning_action_count}

        </div>

    </div>

</div>


<!-- =====================================================
     QUALITY SCORE
===================================================== -->

<div class="section">

    <h2>
        Quality Score Summary
    </h2>

    <div class="score-grid">

        <div class="score-card">

            <div class="label">
                Initial Quality Score
            </div>

            <div class="value">
                {initial_quality_score:.2f}
            </div>

            <div>
                / 100
            </div>

        </div>


        <div class="score-card">

            <div class="label">
                Final Quality Score
            </div>

            <div class="value">
                {final_quality_score:.2f}
            </div>

            <div>
                / 100
            </div>

        </div>


        <div class="score-card highlight">

            <div class="label">
                Overall Improvement
            </div>

            <div class="value">
                {overall_improvement:+.2f}
            </div>

            <div>
                points
            </div>

        </div>

    </div>

</div>


<!-- =====================================================
     ANOMALY ANALYSIS
===================================================== -->

<div class="section">

    <h2>
        Anomaly Analysis
    </h2>

    <div class="summary-list">

        <div class="summary-item">

            <strong>
                Anomalies Before Cleaning
            </strong>

            {initial_anomaly_count}

        </div>


        <div class="summary-item">

            <strong>
                Anomalies After Cleaning
            </strong>

            {final_anomaly_count}

        </div>


        <div class="summary-item">

            <strong>
                Anomalies Resolved
            </strong>

            {anomalies_resolved}

        </div>


        <div class="summary-item">

            <strong>
                Resolution Rate
            </strong>

            {anomaly_resolution_rate:.2f}%

        </div>

    </div>

</div>


<!-- =====================================================
     CLEANING PROCESS
===================================================== -->

<div class="section">

    <h2>
        Cleaning Process
    </h2>

    <p>
        The platform performed
        <strong>{cleaning_iterations}</strong>
        cleaning iteration(s) and rechecked
        the dataset after each pass.
    </p>

    {cleaning_table}

</div>


<!-- =====================================================
     RECOMMENDATIONS
===================================================== -->

<div class="section">

    <h2>
        Recommendations
    </h2>

    {recommendations_table}

</div>


<!-- =====================================================
     FINAL STATUS
===================================================== -->

<div class="section">

    <h2>
        Final Dataset Status
    </h2>

    <div class="status {status_class}">

        {safe_final_status}

    </div>

    <p>
        {html.escape(status_description)}
    </p>

    <p>

        {"✅ All detected anomalies were resolved."
        if final_anomaly_count == 0
        else
        f"⚠️ {final_anomaly_count} anomaly/anomalies "
        "remain for further review."}

    </p>

</div>


<!-- =====================================================
     REPORT SUMMARY
===================================================== -->

<div class="section">

    <h2>
        Report Summary
    </h2>

    <div class="summary-list">

        <div class="summary-item">

            <strong>
                Dataset
            </strong>

            {safe_dataset_name}

        </div>


        <div class="summary-item">

            <strong>
                Initial Score
            </strong>

            {initial_quality_score:.2f} / 100

        </div>


        <div class="summary-item">

            <strong>
                Final Score
            </strong>

            {final_quality_score:.2f} / 100

        </div>


        <div class="summary-item">

            <strong>
                Score Improvement
            </strong>

            {overall_improvement:+.2f} points

        </div>


        <div class="summary-item">

            <strong>
                Initial Anomalies
            </strong>

            {initial_anomaly_count}

        </div>


        <div class="summary-item">

            <strong>
                Final Anomalies
            </strong>

            {final_anomaly_count}

        </div>


        <div class="summary-item">

            <strong>
                Anomalies Resolved
            </strong>

            {anomalies_resolved}

        </div>


        <div class="summary-item">

            <strong>
                Final Status
            </strong>

            {safe_final_status}

        </div>

    </div>

</div>


<!-- =====================================================
     FOOTER
===================================================== -->

<div class="footer">

    Report generated by the
    <strong>
        Data Quality & Anomaly Detection Platform
    </strong>

    <br>

    Professional Reporting Module

</div>


</div>

</body>

</html>
"""

    return report_html
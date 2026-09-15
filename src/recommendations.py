import pandas as pd


def generate_recommendations(
    quality_results,
    anomaly_report,
    overall_quality_score
):
    """
    Generate actionable recommendations based on
    dataset quality issues and anomaly detection results.
    """

    recommendations = []

    # =====================================================
    # SAFETY CHECK
    # =====================================================

    if quality_results is None:
        quality_results = {}

    if anomaly_report is None:
        anomaly_report = pd.DataFrame()


    # =====================================================
    # GET QUALITY RULE TABLE
    # =====================================================

    rule_table = quality_results.get(
        "rule_table",
        pd.DataFrame()
    )

    if rule_table is None:
        rule_table = pd.DataFrame()


    # =====================================================
    # ANALYZE QUALITY RULES
    # =====================================================

    if not rule_table.empty:

        for _, row in rule_table.iterrows():

            rule_name = str(
                row.get(
                    "rule",
                    ""
                )
            ).lower()

            issue_count = row.get(
                "count",
                0
            )

            severity = str(
                row.get(
                    "severity",
                    ""
                )
            ).lower()


            # ---------------------------------------------
            # MISSING VALUES
            # ---------------------------------------------

            if (
                "missing" in rule_name
                and issue_count > 0
            ):

                recommendations.append({
                    "Priority": "High",
                    "Issue": "Missing Values",
                    "Recommendation":
                        "Review missing values and consider "
                        "filling them using an appropriate "
                        "strategy such as median, mean, "
                        "mode, or domain-specific values."
                })


            # ---------------------------------------------
            # DUPLICATES
            # ---------------------------------------------

            elif (
                "duplicate" in rule_name
                and issue_count > 0
            ):

                recommendations.append({
                    "Priority": "Medium",
                    "Issue": "Duplicate Records",
                    "Recommendation":
                        "Review duplicate records and remove "
                        "them when they represent repeated "
                        "entries rather than legitimate data."
                })


            # ---------------------------------------------
            # VALIDITY
            # ---------------------------------------------

            elif (
                (
                    "valid" in rule_name
                    or "invalid" in rule_name
                )
                and issue_count > 0
            ):

                recommendations.append({
                    "Priority": "High",
                    "Issue": "Invalid Values",
                    "Recommendation":
                        "Review invalid values and correct "
                        "their format, range, or data type "
                        "before using the dataset for analysis."
                })


            # ---------------------------------------------
            # DATA TYPE ISSUES
            # ---------------------------------------------

            elif (
                "type" in rule_name
                and issue_count > 0
            ):

                recommendations.append({
                    "Priority": "Medium",
                    "Issue": "Data Type Issues",
                    "Recommendation":
                        "Review column data types and convert "
                        "values to appropriate numeric, "
                        "categorical, date, or text formats."
                })


    # =====================================================
    # ANALYZE ANOMALIES
    # =====================================================

    if not anomaly_report.empty:

        anomaly_count = len(
            anomaly_report
        )

        recommendations.append({
            "Priority": "High",
            "Issue": "Anomalies Detected",
            "Recommendation":
                f"{anomaly_count} anomaly record(s) were "
                "detected. Review extreme or unusual values "
                "and verify whether they are genuine observations "
                "or data-quality problems."
        })


    # =====================================================
    # QUALITY SCORE RECOMMENDATION
    # =====================================================

    if overall_quality_score < 60:

        recommendations.append({
            "Priority": "Critical",
            "Issue": "Low Dataset Quality",
            "Recommendation":
                "The dataset has significant quality issues. "
                "Clean missing, invalid, duplicate, and anomalous "
                "records before using it for downstream analysis."
        })

    elif overall_quality_score < 80:

        recommendations.append({
            "Priority": "High",
            "Issue": "Moderate Dataset Quality",
            "Recommendation":
                "The dataset has several quality issues. "
                "Review the detected problems and clean the "
                "dataset before relying on analytical results."
        })

    elif overall_quality_score < 95:

        recommendations.append({
            "Priority": "Medium",
            "Issue": "Minor Quality Issues",
            "Recommendation":
                "The dataset is generally usable, but some "
                "quality issues should be reviewed to improve "
                "reliability."
        })


    # =====================================================
    # CLEAN DATASET MESSAGE
    # =====================================================

    if (
        overall_quality_score >= 95
        and anomaly_report.empty
        and not recommendations
    ):

        recommendations.append({
            "Priority": "Low",
            "Issue": "Dataset Ready",
            "Recommendation":
                "The dataset appears clean and is ready "
                "for downstream analysis or machine learning."
        })


    # =====================================================
    # REMOVE DUPLICATE RECOMMENDATIONS
    # =====================================================

    unique_recommendations = []

    seen = set()

    for recommendation in recommendations:

        key = (
            recommendation["Issue"],
            recommendation["Recommendation"]
        )

        if key not in seen:

            seen.add(key)

            unique_recommendations.append(
                recommendation
            )


    # =====================================================
    # CREATE DATAFRAME
    # =====================================================

    if unique_recommendations:

        recommendation_df = pd.DataFrame(
            unique_recommendations
        )

    else:

        recommendation_df = pd.DataFrame(
            columns=[
                "Priority",
                "Issue",
                "Recommendation"
            ]
        )


    return recommendation_df
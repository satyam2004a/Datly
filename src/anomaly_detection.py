import pandas as pd


# =========================================================
# CONFIGURATION
# =========================================================

ID_COLUMN_SUFFIX = "_id"

Z_ANOMALY_THRESHOLD = 3

CRITICAL_Z_THRESHOLD = 4
HIGH_Z_THRESHOLD = 3
MEDIUM_Z_THRESHOLD = 2

IQR_MULTIPLIER = 1.5

Z_SCORE_WEIGHT = 0.7
IQR_WEIGHT = 0.3


# =========================================================
# SEVERITY
# =========================================================

def get_severity(z_score, iqr_anomaly):

    if pd.isna(z_score):
        return "Unknown"

    abs_z = abs(z_score)

    if abs_z > CRITICAL_Z_THRESHOLD:
        return "Critical"

    elif abs_z > HIGH_Z_THRESHOLD:
        return "High"

    elif abs_z >= MEDIUM_Z_THRESHOLD or iqr_anomaly:
        return "Medium"

    else:
        return "Normal"


# =========================================================
# INDIVIDUAL ANOMALY SCORE
# =========================================================

def calculate_individual_score(z_score, iqr_anomaly):

    if pd.isna(z_score):
        return 0

    abs_z = abs(z_score)

    # Z-score contribution: 70%
    z_score_component = min(
        (abs_z / 5) * 100,
        100
    )

    # IQR contribution: 30%
    iqr_component = (
        100
        if iqr_anomaly
        else 0
    )

    # Combined score
    score = (
        z_score_component * Z_SCORE_WEIGHT
        + iqr_component * IQR_WEIGHT
    )

    return round(score, 2)


# =========================================================
# DETECT ANOMALIES
# =========================================================

def detect_anomalies(df):

    # Find numerical columns
    numerical_columns = df.select_dtypes(
        include="number"
    ).columns

    # Remove ID-like columns
    numerical_columns = [
        column
        for column in numerical_columns
        if not column.lower().endswith(
            ID_COLUMN_SUFFIX
        )
    ]

    results = {}

    for column in numerical_columns:

        # Count missing values
        missing_count = df[column].isna().sum()

        # Mean and standard deviation
        mean = df[column].mean()
        std = df[column].std()

        # IQR calculation
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)

        iqr = q3 - q1

        lower_limit = (
            q1 - IQR_MULTIPLIER * iqr
        )

        upper_limit = (
            q3 + IQR_MULTIPLIER * iqr
        )

        # Handle constant columns
        if std == 0 or pd.isna(std):

            results[column] = {
                "status": "skipped",
                "reason": "Column has no variation",
                "missing_values": int(
                    missing_count
                )
            }

            continue

        # Z-score
        z_scores = (
            df[column] - mean
        ) / std

        z_anomaly = (
            z_scores.abs()
            > Z_ANOMALY_THRESHOLD
        )

        # IQR anomaly
        iqr_anomaly = (
            (df[column] < lower_limit)
            |
            (df[column] > upper_limit)
        )

        # Final anomaly
        is_anomaly = (
            z_anomaly
            |
            iqr_anomaly
        )

        # -------------------------------------------------
        # Calculate method, severity and score together
        # -------------------------------------------------

        anomaly_method = []
        severity = []
        individual_scores = []

        for z, z_flag, iqr_flag in zip(
            z_scores,
            z_anomaly,
            iqr_anomaly
        ):

            # Detection method
            if z_flag and iqr_flag:
                method = "Z-score + IQR"

            elif z_flag:
                method = "Z-score"

            elif iqr_flag:
                method = "IQR"

            else:
                method = "None"

            anomaly_method.append(
                method
            )

            # Severity
            severity.append(
                get_severity(
                    z,
                    iqr_flag
                )
            )

            # Individual score
            individual_scores.append(
                calculate_individual_score(
                    z,
                    iqr_flag
                )
            )

        # Store results
        results[column] = {

            "status": "analyzed",

            "missing_values": int(
                missing_count
            ),

            "anomaly_count": int(
                is_anomaly.sum()
            ),

            "data": pd.DataFrame({

                "row_index": df.index,

                "value": df[column],

                "z_score": z_scores,

                "iqr_anomaly": iqr_anomaly,

                "anomaly_method": anomaly_method,

                "severity": severity,

                "individual_score": individual_scores,

                "is_anomaly": is_anomaly
            })
        }

    return results


# =========================================================
# ANOMALY SUMMARY
# =========================================================

def anomaly_summary(results):

    summary = []

    for column, result in results.items():

        summary.append({

            "column": column,

            "status": result["status"],

            "missing_values":
                result["missing_values"],

            "anomaly_count":
                result.get(
                    "anomaly_count",
                    0
                )
        })

    return pd.DataFrame(summary)


# =========================================================
# COMBINED ANOMALY REPORT
# =========================================================

def combined_anomaly_report(results):

    all_anomalies = []

    for column, result in results.items():

        if result["status"] != "analyzed":
            continue

        data = result["data"]

        anomalies = data[
            data["is_anomaly"]
        ].copy()

        if not anomalies.empty:

            anomalies["column"] = column

            all_anomalies.append(
                anomalies
            )

    if not all_anomalies:

        return pd.DataFrame(
            columns=[
                "row_index",
                "value",
                "z_score",
                "iqr_anomaly",
                "anomaly_method",
                "severity",
                "individual_score",
                "is_anomaly",
                "column"
            ]
        )

    return pd.concat(
        all_anomalies,
        ignore_index=True
    )


# =========================================================
# ANOMALY SCORE
# =========================================================

def calculate_anomaly_score(
    report,
    total_rows
):

    # No anomalies = perfect anomaly score
    if report.empty:
        return 0.0

    if total_rows == 0:
        return 0.0

    severity_scores = {

        "Normal": 0,

        "Medium": 50,

        "High": 75,

        "Critical": 100

    }

    scores = report[
        "severity"
    ].map(
        severity_scores
    ).fillna(0)

    # Total anomaly severity
    total_penalty = scores.sum()

    # Normalize against dataset size
    anomaly_score = (
        total_penalty / total_rows
    )

    # Keep score between 0 and 100
    anomaly_score = min(
        max(anomaly_score, 0),
        100
    )

    return round(
        anomaly_score,
        2
    )


# =========================================================
# QUALITY SCORE
# =========================================================

def calculate_quality_score(
    anomaly_score
):

    quality_score = (
        100 - anomaly_score
    )

    return max(
        0,
        min(
            100,
            quality_score
        )
    )
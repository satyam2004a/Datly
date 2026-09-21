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

    z_score_component = min(
        (abs_z / 5) * 100,
        100
    )

    iqr_component = (
        100
        if iqr_anomaly
        else 0
    )

    score = (
        z_score_component * Z_SCORE_WEIGHT
        + iqr_component * IQR_WEIGHT
    )

    return round(score, 2)


# =========================================================
# DETECT ANOMALIES
# =========================================================

def detect_anomalies(df):
    """
    Detect numerical anomalies using Z-score and IQR.

    Performance optimized:
    - Uses vectorized Pandas operations.
    - Avoids row-by-row Python loops.
    - Stores detailed data only for anomalous rows.
    - Preserves the existing result structure expected by
      the rest of Datly.
    """

    # ---------------------------------------------------------
    # FIND NUMERICAL COLUMNS
    # ---------------------------------------------------------

    numerical_columns = df.select_dtypes(
        include="number"
    ).columns

    numerical_columns = [
        column
        for column in numerical_columns
        if not str(column).lower().endswith(
            ID_COLUMN_SUFFIX
        )
    ]

    results = {}

    # ---------------------------------------------------------
    # ANALYZE EACH NUMERICAL COLUMN
    # ---------------------------------------------------------

    for column in numerical_columns:

        series = df[column]

        # -----------------------------------------------------
        # MISSING VALUES
        # -----------------------------------------------------

        missing_count = int(
            series.isna().sum()
        )

        # -----------------------------------------------------
        # BASIC STATISTICS
        # -----------------------------------------------------

        valid_series = series.dropna()

        # Statistical anomaly detection is unreliable
        # when there are too few valid observations.
        if len(valid_series) < 5:

            results[column] = {
                "status": "skipped",
                "reason": (
                    "Insufficient valid values "
                    "for statistical anomaly detection"
                ),
                "missing_values": missing_count
            }

            continue

        mean = valid_series.mean()
        std = valid_series.std()

        q1 = valid_series.quantile(0.25)
        q3 = valid_series.quantile(0.75)
        iqr = q3 - q1

        lower_limit = (
            q1
            - (IQR_MULTIPLIER * iqr)
        )

        upper_limit = (
            q3
            + (IQR_MULTIPLIER * iqr)
        )

        # -----------------------------------------------------
        # CONSTANT / INVALID STATISTICAL COLUMN
        # -----------------------------------------------------

        if std == 0 or pd.isna(std):

            results[column] = {
                "status": "skipped",
                "reason": "Column has no variation",
                "missing_values": missing_count
            }

            continue

        # -----------------------------------------------------
        # Z-SCORE
        # -----------------------------------------------------

        z_scores = (
            series - mean
        ) / std

        z_anomaly = (
            z_scores.abs()
            > Z_ANOMALY_THRESHOLD
        )

        # -----------------------------------------------------
        # IQR ANOMALY
        # -----------------------------------------------------

        iqr_anomaly = (
            (series < lower_limit)
            |
            (series > upper_limit)
        )

        # -----------------------------------------------------
        # FINAL ANOMALY MASK
        # -----------------------------------------------------

        is_anomaly = (
            z_anomaly
            |
            iqr_anomaly
        )

        print(
            f"\nDEBUG — {column}"
        )

        print(
            pd.DataFrame({
                "value": series,
                "z_score": z_scores,
                "iqr_anomaly": iqr_anomaly,
                "z_anomaly": z_anomaly,
                "is_anomaly": is_anomaly
            })
        )

        anomaly_count = int(
            is_anomaly.sum()
        )

        # -----------------------------------------------------
        # STORE ONLY ANOMALOUS ROWS
        # -----------------------------------------------------

        if anomaly_count > 0:

            anomaly_z_scores = (
                z_scores[is_anomaly]
            )

            anomaly_iqr = (
                iqr_anomaly[is_anomaly]
            )

            anomaly_z_flag = (
                z_anomaly[is_anomaly]
            )

            # -------------------------------------------------
            # VECTORIZED METHOD
            # -------------------------------------------------

            anomaly_method = pd.Series(
                "IQR",
                index=anomaly_z_scores.index,
                dtype="object"
            )

            both_mask = (
                anomaly_z_flag
                & anomaly_iqr
            )

            z_only_mask = (
                anomaly_z_flag
                & ~anomaly_iqr
            )

            anomaly_method.loc[
                both_mask
            ] = "Z-score + IQR"

            anomaly_method.loc[
                z_only_mask
            ] = "Z-score"

            # -------------------------------------------------
            # VECTORIZED SEVERITY
            # -------------------------------------------------

            abs_z = (
                anomaly_z_scores.abs()
            )

            severity = pd.Series(
                "Medium",
                index=anomaly_z_scores.index,
                dtype="object"
            )

            severity.loc[
                abs_z > CRITICAL_Z_THRESHOLD
            ] = "Critical"

            severity.loc[
                (abs_z > HIGH_Z_THRESHOLD)
                & (abs_z <= CRITICAL_Z_THRESHOLD)
            ] = "High"

            # -------------------------------------------------
            # VECTORIZED INDIVIDUAL SCORE
            # -------------------------------------------------

            z_score_component = (
                (abs_z / 5) * 100
            ).clip(
                upper=100
            )

            iqr_component = (
                anomaly_iqr.astype(float) * 100
            )

            individual_scores = (
                z_score_component * Z_SCORE_WEIGHT
                + iqr_component * IQR_WEIGHT
            ).round(2)

            # -------------------------------------------------
            # CREATE COMPACT ANOMALY DATA
            # -------------------------------------------------

            anomaly_data = pd.DataFrame({

                "row_index":
                    anomaly_z_scores.index,

                "value":
                    series.loc[is_anomaly],

                "z_score":
                    anomaly_z_scores,

                "iqr_anomaly":
                    anomaly_iqr,

                "anomaly_method":
                    anomaly_method,

                "severity":
                    severity,

                "individual_score":
                    individual_scores,

                "is_anomaly":
                    True
            })

        else:

            # Keep the same DataFrame schema while avoiding
            # creation of a full-size DataFrame.

            anomaly_data = pd.DataFrame(
                columns=[
                    "row_index",
                    "value",
                    "z_score",
                    "iqr_anomaly",
                    "anomaly_method",
                    "severity",
                    "individual_score",
                    "is_anomaly"
                ]
            )

        # -----------------------------------------------------
        # STORE RESULTS
        # -----------------------------------------------------

        results[column] = {

            "status": "analyzed",

            "missing_values":
                missing_count,

            "anomaly_count":
                anomaly_count,

            "data":
                anomaly_data
        }

    return results


# =========================================================
# ANOMALY SUMMARY
# =========================================================

def anomaly_summary(results):

    summary = []

    for column, result in results.items():

        summary.append({

            "column":
                column,

            "status":
                result["status"],

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

        if data.empty:
            continue

        anomalies = data.copy()

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

    total_penalty = scores.sum()

    anomaly_score = (
        total_penalty / total_rows
    )

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
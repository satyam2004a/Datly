# =========================================================
# 1. QUALITY STATUS
# =========================================================

def get_quality_status(score):
    """
    Convert numerical quality score into a readable status.
    """

    if score >= 90:
        return "Excellent"

    elif score >= 75:
        return "Good"

    elif score >= 50:
        return "Average"

    else:
        return "Poor"


# =========================================================
# 2. COMPLETENESS SCORE
# =========================================================

def calculate_completeness_score(quality_results):
    """
    Calculate completeness score from quality rules.

    Completeness is based primarily on missing values
    and columns with a high percentage of missing data.
    """

    rules = quality_results.get("rules", [])

    missing_rate = 0.0

    for rule in rules:

        if rule["rule"] == "Missing Values":
            missing_rate = rule["rate"]
            break

    score = 100 - missing_rate

    return round(max(score, 0), 2)


# =========================================================
# 3. VALIDITY SCORE
# =========================================================

def calculate_validity_score(quality_results):
    """
    Calculate validity score using validity-related
    quality rules.
    """

    rules = quality_results.get("rules", [])

    penalties = []

    for rule in rules:

        if rule["category"] == "Validity":

            if rule["severity"] == "High":
                penalties.append(20)

            elif rule["severity"] == "Medium":
                penalties.append(10)

            elif rule["severity"] == "Low":
                penalties.append(5)

    total_penalty = sum(penalties)

    score = 100 - total_penalty

    return round(max(score, 0), 2)


# =========================================================
# 4. UNIQUENESS SCORE
# =========================================================

def calculate_uniqueness_score(quality_results):
    """
    Calculate uniqueness score based on duplicate rows.
    """

    rules = quality_results.get("rules", [])

    duplicate_rate = 0.0

    for rule in rules:

        if rule["rule"] == "Duplicate Rows":
            duplicate_rate = rule["rate"]
            break

    score = 100 - duplicate_rate

    return round(max(score, 0), 2)


# =========================================================
# 5. ANOMALY QUALITY SCORE
# =========================================================

def calculate_anomaly_quality_score(anomaly_penalty):
    """
    Convert anomaly penalty into anomaly quality score.

    0 penalty   -> 100 quality
    100 penalty -> 0 quality
    """

    if anomaly_penalty is None:
        return 100.0

    try:
        penalty = float(anomaly_penalty)
    except (TypeError, ValueError):
        return 0.0

    penalty = max(
        0.0,
        min(penalty, 100.0)
    )

    quality_score = 100.0 - penalty

    return round(
        quality_score,
        2
    )


# =========================================================
# 6. OVERALL QUALITY SCORE
# =========================================================

def calculate_overall_quality_score(
    completeness_score,
    validity_score,
    uniqueness_score,
    anomaly_quality_score
):
    """
    Calculate the final dataset quality score.

    All four dimensions have equal weight.
    """

    overall_score = (
        completeness_score
        + validity_score
        + uniqueness_score
        + anomaly_quality_score
    ) / 4

    return round(
        max(
            0.0,
            min(
                overall_score,
                100.0
            )
        ),
        2
    )

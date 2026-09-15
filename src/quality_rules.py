import pandas as pd


# =========================================================
# DATA QUALITY RULES ENGINE
# =========================================================

def check_missing_values(df):
    """Detect missing cells and calculate the missing rate."""

    total_cells = df.shape[0] * df.shape[1]
    missing_cells = int(df.isna().sum().sum())

    missing_rate = (
        missing_cells / total_cells * 100
        if total_cells > 0
        else 0.0
    )

    affected_columns = [
        column
        for column in df.columns
        if df[column].isna().any()
    ]

    return {
        "rule": "Missing Values",
        "category": "Completeness",
        "count": missing_cells,
        "rate": round(missing_rate, 2),
        "severity": (
            "High" if missing_rate > 20
            else "Medium" if missing_cells > 0
            else "Pass"
        ),
        "details": (
            f"{len(affected_columns)} column(s) contain "
            "missing values."
            if affected_columns
            else "No missing values found."
        )
    }


def check_duplicate_rows(df):
    """Detect exact duplicate rows."""

    duplicate_count = int(df.duplicated().sum())

    duplicate_rate = (
        duplicate_count / len(df) * 100
        if len(df) > 0
        else 0.0
    )

    return {
        "rule": "Duplicate Rows",
        "category": "Uniqueness",
        "count": duplicate_count,
        "rate": round(duplicate_rate, 2),
        "severity": (
            "High" if duplicate_rate > 10
            else "Medium" if duplicate_count > 0
            else "Pass"
        ),
        "details": (
            f"{duplicate_count} exact duplicate row(s) found."
            if duplicate_count
            else "No exact duplicate rows found."
        )
    }


def check_empty_columns(df):
    """Detect columns containing no usable values."""

    empty_columns = [
        column
        for column in df.columns
        if df[column].isna().all()
    ]

    return {
        "rule": "Empty Columns",
        "category": "Validity",
        "count": len(empty_columns),
        "rate": (
            round(
                len(empty_columns) / len(df.columns) * 100,
                2
            )
            if len(df.columns) > 0
            else 0.0
        ),
        "severity": (
            "High" if empty_columns
            else "Pass"
        ),
        "details": (
            ", ".join(map(str, empty_columns))
            if empty_columns
            else "No completely empty columns found."
        )
    }


def check_duplicate_column_names(df):
    """Detect repeated column names."""

    duplicated_mask = df.columns.duplicated(keep=False)

    duplicate_names = df.columns[
        duplicated_mask
    ].tolist()

    unique_duplicate_names = list(
        dict.fromkeys(duplicate_names)
    )

    return {
        "rule": "Duplicate Column Names",
        "category": "Validity",
        "count": len(unique_duplicate_names),
        "rate": (
            round(
                len(unique_duplicate_names)
                / len(df.columns) * 100,
                2
            )
            if len(df.columns) > 0
            else 0.0
        ),
        "severity": (
            "High" if unique_duplicate_names
            else "Pass"
        ),
        "details": (
            ", ".join(
                map(str, unique_duplicate_names)
            )
            if unique_duplicate_names
            else "All column names are unique."
        )
    }


def check_constant_columns(df):
    """Detect columns with only one distinct non-missing value."""

    constant_columns = []

    for column in df.columns:

        non_missing = df[column].dropna()

        if (
            len(non_missing) > 0
            and non_missing.nunique() <= 1
        ):
            constant_columns.append(column)

    return {
        "rule": "Constant Columns",
        "category": "Validity",
        "count": len(constant_columns),
        "rate": (
            round(
                len(constant_columns)
                / len(df.columns) * 100,
                2
            )
            if len(df.columns) > 0
            else 0.0
        ),
        "severity": (
            "Medium" if constant_columns
            else "Pass"
        ),
        "details": (
            ", ".join(
                map(str, constant_columns)
            )
            if constant_columns
            else "No constant columns found."
        )
    }


def check_negative_numeric_values(df):
    """
    Detect negative numeric values.

    Negative values are reported as potential quality issues rather
    than automatically declared invalid because some real-world
    datasets legitimately contain negative numbers.
    """

    negative_counts = {}

    for column in df.select_dtypes(
        include="number"
    ).columns:

        count = int(
            (df[column] < 0).sum()
        )

        if count > 0:
            negative_counts[column] = count

    total_negative = sum(
        negative_counts.values()
    )

    numeric_columns = len(
        df.select_dtypes(
            include="number"
        ).columns
    )

    return {
        "rule": "Negative Numeric Values",
        "category": "Validity",
        "count": total_negative,
        "rate": (
            round(
                total_negative
                / max(
                    numeric_columns * len(df),
                    1
                )
                * 100,
                2
            )
            if len(df) > 0
            else 0.0
        ),
        "severity": (
            "Medium" if total_negative > 0
            else "Pass"
        ),
        "details": (
            ", ".join(
                f"{column}: {count}"
                for column, count
                in negative_counts.items()
            )
            if negative_counts
            else "No negative numeric values found."
        )
    }


def check_high_missing_columns(
    df,
    threshold=50
):
    """Detect columns whose missing percentage exceeds a threshold."""

    high_missing = []

    if len(df) == 0:

        return {
            "rule": "High Missing Columns",
            "category": "Completeness",
            "count": 0,
            "rate": 0.0,
            "severity": "Pass",
            "details": "Dataset has no rows."
        }

    for column in df.columns:

        missing_percentage = (
            df[column].isna().mean() * 100
        )

        if missing_percentage > threshold:
            high_missing.append(column)

    return {
        "rule": "High Missing Columns",
        "category": "Completeness",
        "count": len(high_missing),
        "rate": (
            round(
                len(high_missing)
                / len(df.columns)
                * 100,
                2
            )
            if len(df.columns) > 0
            else 0.0
        ),
        "severity": (
            "High" if high_missing
            else "Pass"
        ),
        "details": (
            ", ".join(
                map(str, high_missing)
            )
            if high_missing
            else (
                f"No columns exceed "
                f"{threshold}% missing values."
            )
        )
    }


def run_quality_rules(df):
    """
    Run all general-purpose data quality rules.

    Returns both machine-friendly rule results and a DataFrame
    suitable for the Streamlit dashboard.
    """

    rules = [
        check_missing_values(df),
        check_duplicate_rows(df),
        check_empty_columns(df),
        check_duplicate_column_names(df),
        check_constant_columns(df),
        check_negative_numeric_values(df),
        check_high_missing_columns(df)
    ]

    results = {
        "rules": rules,
        "rule_table": pd.DataFrame(rules)
    }

    return results
import pandas as pd


# =========================================================
# DATA QUALITY RULES ENGINE
# =========================================================

def check_missing_values(df, missing_counts=None):
    """Detect missing cells and calculate the missing rate."""

    total_rows, total_columns = df.shape
    total_cells = total_rows * total_columns

    if missing_counts is None:
        missing_counts = df.isna().sum()

    missing_cells = int(missing_counts.sum())

    missing_rate = (
        missing_cells / total_cells * 100
        if total_cells > 0
        else 0.0
    )

    affected_columns = missing_counts[
        missing_counts > 0
    ].index.tolist()

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


def check_duplicate_rows(df, duplicate_count=None):
    """Detect exact duplicate rows."""

    if duplicate_count is None:
        duplicate_count = int(
            df.duplicated().sum()
        )

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


def check_empty_columns(df, missing_counts=None):
    """Detect columns containing no usable values."""

    if missing_counts is None:
        missing_counts = df.isna().sum()

    empty_columns = missing_counts[
        missing_counts == len(df)
    ].index.tolist()

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

    duplicated_mask = df.columns.duplicated(
        keep=False
    )

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

    # nunique() is performed once per column.
    # This preserves the original behavior while
    # avoiding creation of temporary dropna() Series.
    unique_counts = df.nunique(
        dropna=True
    )

    non_missing_counts = df.notna().sum()

    constant_mask = (
        (non_missing_counts > 0)
        & (unique_counts <= 1)
    )

    constant_columns = (
        unique_counts.index[constant_mask]
        .tolist()
    )

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


def check_negative_numeric_values(
    df,
    numeric_df=None
):
    """
    Detect negative numeric values.

    Negative values are reported as potential quality issues rather
    than automatically declared invalid because some real-world
    datasets legitimately contain negative numbers.
    """

    if numeric_df is None:
        numeric_df = df.select_dtypes(
            include="number"
        )

    if numeric_df.empty:

        return {
            "rule": "Negative Numeric Values",
            "category": "Validity",
            "count": 0,
            "rate": 0.0,
            "severity": "Pass",
            "details": "No numeric columns found."
        }

    # One vectorized operation for all numeric columns.
    negative_mask = numeric_df < 0

    negative_counts_series = (
        negative_mask.sum()
    )

    negative_counts = {
        column: int(count)
        for column, count
        in negative_counts_series.items()
        if count > 0
    }

    total_negative = sum(
        negative_counts.values()
    )

    numeric_columns = numeric_df.shape[1]

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
    threshold=50,
    missing_counts=None
):
    """Detect columns whose missing percentage exceeds a threshold."""

    if len(df) == 0:

        return {
            "rule": "High Missing Columns",
            "category": "Completeness",
            "count": 0,
            "rate": 0.0,
            "severity": "Pass",
            "details": "Dataset has no rows."
        }

    if missing_counts is None:
        missing_counts = df.isna().sum()

    # Vectorized missing percentage calculation.
    missing_percentages = (
        missing_counts / len(df) * 100
    )

    high_missing = (
        missing_percentages[
            missing_percentages > threshold
        ]
        .index
        .tolist()
    )

    return {
        "rule": "High Missing Columns",
        "category": "Completeness",
        "count": len(high_missing),
        "rate": (
            round(
                len(high_missing)
                / len(df.columns) * 100,
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


# =========================================================
# MAIN QUALITY RULE ENGINE
# =========================================================

def run_quality_rules(df):
    """
    Run all general-purpose data quality rules.

    Performance optimized:
    - Missing-value counts calculated once.
    - Numeric columns selected once.
    - Duplicate rows calculated once.
    - Missing percentages reused.
    - Constant-column analysis is vectorized.
    - Existing rule names and output structure preserved.
    """

    # ---------------------------------------------------------
    # SHARED CALCULATIONS
    # ---------------------------------------------------------

    missing_counts = df.isna().sum()

    duplicate_count = int(
        df.duplicated().sum()
    )

    numeric_df = df.select_dtypes(
        include="number"
    )

    # ---------------------------------------------------------
    # RUN RULES
    # ---------------------------------------------------------

    rules = [
        check_missing_values(
            df,
            missing_counts
        ),

        check_duplicate_rows(
            df,
            duplicate_count
        ),

        check_empty_columns(
            df,
            missing_counts
        ),

        check_duplicate_column_names(
            df
        ),

        check_constant_columns(
            df
        ),

        check_negative_numeric_values(
            df,
            numeric_df
        ),

        check_high_missing_columns(
            df,
            threshold=50,
            missing_counts=missing_counts
        )
    ]

    # ---------------------------------------------------------
    # FINAL RESULT
    # ---------------------------------------------------------

    return {
        "rules": rules,
        "rule_table": pd.DataFrame(rules)
    }

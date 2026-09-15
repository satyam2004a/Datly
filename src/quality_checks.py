import pandas as pd


def check_missing_values(df):
    """Check the number of missing values in the dataset."""

    missing_values = int(
        df.isna().sum().sum()
    )

    return missing_values


def check_duplicate_rows(df):
    """Check the number of duplicate rows."""

    duplicate_rows = int(
        df.duplicated().sum()
    )

    return duplicate_rows


def check_data_types(df):
    """Return the data types of all columns."""

    return df.dtypes.astype(str).to_dict()


def run_quality_checks(df):
    """
    Backward-compatible quality check function.

    This keeps the older interface available while the new
    quality_rules.py module handles the professional rules engine.
    """

    return {
        "missing_values": check_missing_values(df),
        "duplicate_rows": check_duplicate_rows(df),
        "data_types": check_data_types(df)
    }


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("data/test_employees.csv")


# ============================================================
# 2. BASIC COLUMN DETECTION
# ============================================================

def get_numerical_columns(df):
    return df.select_dtypes(include="number").columns.tolist()


def get_categorical_columns(df):
    return df.select_dtypes(exclude="number").columns.tolist()


# ============================================================
# 3. ANALYSIS COLUMNS
#    Exclude ID columns from statistical analysis
# ============================================================

def get_analysis_columns(df):

    numerical_columns = get_numerical_columns(df)

    analysis_columns = [
        column
        for column in numerical_columns
        if "id" not in column.lower()
    ]

    return analysis_columns


# ============================================================
# 4. MISSING VALUE CHECK
# ============================================================

def check_missing_values(df):

    return df.isnull().sum()


# ============================================================
# 5. DUPLICATE ROW CHECK
# ============================================================

def check_duplicate_rows(df):

    return int(df.duplicated().sum())


# ============================================================
# 6. NEGATIVE VALUE CHECK
# ============================================================

def check_negative_values(df):

    numerical_columns = get_numerical_columns(df)

    negative_values = {}

    for column in numerical_columns:

        count = (df[column] < 0).sum()

        if count > 0:
            negative_values[column] = int(count)

    return negative_values


# ============================================================
# 7. NUMERICAL SUMMARY
# ============================================================

def get_numerical_summary(df):

    numerical_columns = get_numerical_columns(df)

    summary = {}

    for column in numerical_columns:

        summary[column] = {
            "min": df[column].min(),
            "max": df[column].max(),
            "mean": df[column].mean(),
            "median": df[column].median()
        }

    return summary


# ============================================================
# 8. OUTLIER DETECTION USING IQR
# ============================================================

def detect_outliers(df):

    outliers = {}

    analysis_columns = get_analysis_columns(df)

    for column in analysis_columns:

        # Remove missing values before calculating statistics
        values = df[column].dropna()

        # If there are not enough values, skip the column
        if len(values) < 2:
            continue

        # Calculate quartiles
        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)

        # Calculate IQR
        IQR = Q3 - Q1

        # Calculate boundaries
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Identify outliers
        outlier_mask = (
            (df[column] < lower_bound) |
            (df[column] > upper_bound)
        )

        # Count outliers
        outlier_count = outlier_mask.sum()

        # Get row indexes
        outlier_rows = df[outlier_mask].index.tolist()

        # Get actual values
        outlier_values = df.loc[
            outlier_mask,
            column
        ].tolist()

        # Store results
        outliers[column] = {
            "outlier_count": int(outlier_count),
            "outlier_rows": outlier_rows,
            "outlier_values": outlier_values,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound
        }

    return outliers


# ============================================================
# 9. COLUMN-LEVEL QUALITY METRICS
# ============================================================

def column_quality_metrics(df):

    metrics = {}

    for column in df.columns:

        # Missing values
        missing_count = df[column].isnull().sum()

        # Missing percentage
        missing_percentage = (
            missing_count / len(df)
        ) * 100

        metrics[column] = {
            "missing_count": int(missing_count),
            "missing_percentage": round(
                missing_percentage,
                2
            )
        }

    return metrics


# ============================================================
# 10. MAIN QUALITY CHECK FUNCTION
# ============================================================

def run_quality_checks(df):

    # Column types
    numerical_columns = get_numerical_columns(df)
    categorical_columns = get_categorical_columns(df)

    # Missing values
    missing_values = check_missing_values(df)

    total_missing_values = int(
        missing_values.sum()
    )

    # Duplicate rows
    duplicate_rows = check_duplicate_rows(df)

    # Negative values
    negative_values = check_negative_values(df)

    # Numerical summary
    numerical_summary = get_numerical_summary(df)

    # Statistical outliers
    outlier_results = detect_outliers(df)

    # Column-level metrics
    quality_metrics = column_quality_metrics(df)

    # Final generalized results
    final_results = {

        "numerical_columns": numerical_columns,

        "categorical_columns": categorical_columns,

        "missing_values": missing_values.to_dict(),

        "total_missing_values": total_missing_values,

        "duplicate_rows": duplicate_rows,

        "negative_values": negative_values,

        "numerical_summary": numerical_summary,

        "outliers": outlier_results,

        "column_quality_metrics": quality_metrics
    }

    return final_results


# ============================================================
# 11. DATASET OVERVIEW
# ============================================================

print("\n========== DATASET OVERVIEW ==========")

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nData Types:")
print(df.dtypes)


# ============================================================
# 12. RUN OUTLIER DETECTION
# ============================================================

outlier_results = detect_outliers(df)

print("\n========== OUTLIER DETECTION ==========")
print(outlier_results)


# ============================================================
# 13. COLUMN INFORMATION
# ============================================================

numerical_columns = get_numerical_columns(df)
categorical_columns = get_categorical_columns(df)

print("\nNumerical Columns:")
print(numerical_columns)

print("\nCategorical Columns:")
print(categorical_columns)


# ============================================================
# 14. MISSING VALUES
# ============================================================

missing_values = check_missing_values(df)

print("\nMissing Values:")
print(missing_values)

total_missing_values = int(
    missing_values.sum()
)

print("\nTotal Missing Values:")
print(total_missing_values)


# ============================================================
# 15. DUPLICATES
# ============================================================

duplicate_rows = check_duplicate_rows(df)

print("\nDuplicate Rows:")
print(duplicate_rows)


# ============================================================
# 16. NEGATIVE VALUES
# ============================================================

negative_values = check_negative_values(df)

print("\nNegative Values:")
print(negative_values)


# ============================================================
# 17. NUMERICAL SUMMARY
# ============================================================

numerical_summary = get_numerical_summary(df)

print("\n========== NUMERICAL SUMMARY ==========")

for column, values in numerical_summary.items():

    print(f"\n{column}:")

    print("Minimum :", values["min"])
    print("Maximum :", values["max"])
    print("Mean    :", values["mean"])
    print("Median  :", values["median"])


# ============================================================
# 18. COLUMN QUALITY METRICS
# ============================================================

quality_metrics = column_quality_metrics(df)

print("\n========== COLUMN QUALITY METRICS ==========")
print(quality_metrics)


# ============================================================
# 19. FINAL GENERALIZED RESULTS
# ============================================================

results = run_quality_checks(df)

print("\n========== FINAL GENERALIZED RESULTS ==========")
print(results)

print("================================================")
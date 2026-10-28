import pandas as pd
import numpy as np


# ============================================================
# DATLY DATA CLEANING
# ============================================================

# Columns containing these terms are treated as
# identifier / code-like fields.
# They should NOT receive median imputation.
CODE_LIKE_KEYWORDS = {
    "postal",
    "postcode",
    "zip",
    "zipcode",
    "pin",
    "pincode",
}


# If more than this percentage of a column is missing,
# Datly will avoid automatic imputation.
HIGH_MISSINGNESS_THRESHOLD = 0.50


def _is_code_like_column(column):
    """
    Detect columns that look like postal/ZIP/PIN/code fields.
    """

    column_name = str(column).lower().replace("_", " ")

    return any(
        keyword in column_name
        for keyword in CODE_LIKE_KEYWORDS
    )


def clean_dataset(df, anomaly_report):
    """
    Clean a dataset using detected quality issues
    and anomalies.

    Performance optimized:
    - Processes anomalies column-by-column.
    - Calculates replacement values once per column.
    - Avoids iterrows() over every anomaly.
    - Avoids scanning the full column for every anomaly.
    - Protects identifier/code-like columns.
    - Protects columns with very high missingness.
    """

    cleaned_df = df.copy()
    cleaning_actions = []

    # ========================================================
    # PROTECTED COLUMNS
    # ========================================================

    protected_columns = {
        column
        for column in cleaned_df.columns
        if str(column).lower().endswith("_id")
    }

    if "id" in cleaned_df.columns:
        protected_columns.add("id")

    # ========================================================
    # DUPLICATE ROWS
    # ========================================================

    duplicate_rows = cleaned_df.duplicated()

    if duplicate_rows.any():

        duplicate_indices = (
            cleaned_df.index[duplicate_rows].tolist()
        )

        cleaned_df = cleaned_df.drop_duplicates()

        cleaning_actions.extend(
            {
                "row_index": row_index,
                "column": "N/A",
                "original_value": "Duplicate Row",
                "action": "Removed Duplicate",
                "reason": "Exact duplicate record",
                "new_value": "Row Removed"
            }
            for row_index in duplicate_indices
        )

    # ========================================================
    # MISSING VALUES
    # ========================================================

    missing_counts = cleaned_df.isna().sum()

    total_rows = len(cleaned_df)

    for column in cleaned_df.columns:

        missing_count = int(missing_counts[column])

        if missing_count == 0:
            continue

        missing_mask = cleaned_df[column].isna()
        missing_indices = cleaned_df.index[missing_mask]

        # ----------------------------------------------------
        # PROTECTED ID / CODE COLUMNS
        # ----------------------------------------------------

        if column in protected_columns:

            cleaning_actions.append(
                {
                    "row_index": "N/A",
                    "column": column,
                    "original_value": "Missing",
                    "action": "Skipped Missing Value",
                    "reason": "Identifier column protected",
                    "new_value": "Remains Missing"
                }
            )

            continue

        # ----------------------------------------------------
        # POSTAL / ZIP / PIN / CODE-LIKE COLUMNS
        # ----------------------------------------------------

        if _is_code_like_column(column):

            # Use mode only when automatic filling is safe.
            # Otherwise retain missing values.

            missing_rate = (
                missing_count / total_rows
                if total_rows > 0
                else 0
            )

            if missing_rate > HIGH_MISSINGNESS_THRESHOLD:

                cleaning_actions.append(
                    {
                        "row_index": "N/A",
                        "column": column,
                        "original_value": "Missing",
                        "action": "Skipped Missing Value",
                        "reason": (
                            f"High missingness "
                            f"({missing_rate * 100:.2f}%) "
                            f"in code-like column"
                        ),
                        "new_value": "Remains Missing"
                    }
                )

                continue

            mode_values = cleaned_df[column].mode()

            if mode_values.empty:

                cleaning_actions.append(
                    {
                        "row_index": "N/A",
                        "column": column,
                        "original_value": "Missing",
                        "action": "Skipped Missing Value",
                        "reason": (
                            "No reliable value available "
                            "for code-like column"
                        ),
                        "new_value": "Remains Missing"
                    }
                )

                continue

            replacement_value = mode_values.iloc[0]

            cleaned_df.loc[
                missing_mask,
                column
            ] = replacement_value

            cleaning_actions.extend(
                {
                    "row_index": row_index,
                    "column": column,
                    "original_value": None,
                    "action": "Filled Missing Value",
                    "reason": "Mode used for code-like column",
                    "new_value": replacement_value
                }
                for row_index in missing_indices
            )

            continue

        # ----------------------------------------------------
        # HIGH MISSINGNESS PROTECTION
        # ----------------------------------------------------

        missing_rate = (
            missing_count / total_rows
            if total_rows > 0
            else 0
        )

        if missing_rate > HIGH_MISSINGNESS_THRESHOLD:

            cleaning_actions.append(
                {
                    "row_index": "N/A",
                    "column": column,
                    "original_value": "Missing",
                    "action": "Skipped Missing Value",
                    "reason": (
                        f"High missingness "
                        f"({missing_rate * 100:.2f}%)"
                    ),
                    "new_value": "Remains Missing"
                }
            )

            continue


        # ====================================================
        # NUMERIC COLUMNS
        # ====================================================

        if pd.api.types.is_numeric_dtype(
            cleaned_df[column]
        ):

            # Use available numeric values.
            valid_values = cleaned_df[column].dropna()

            # Exclude anomalies detected by Datly.
            if (
                anomaly_report is not None
                and not anomaly_report.empty
                and "column" in anomaly_report.columns
                and "row_index" in anomaly_report.columns
            ):
                detected_indices = anomaly_report.loc[
                    anomaly_report["column"] == column,
                    "row_index"
                ]

                valid_values = valid_values.loc[
                    ~valid_values.index.isin(detected_indices)
                ]

            # Exclude invalid ages while calculating the median.
            if "age" in str(column).lower():
                valid_values = valid_values[
                    valid_values.between(0, 120)
                ]

            if valid_values.empty:
                cleaning_actions.append({
                    "row_index": "N/A",
                    "column": column,
                    "original_value": "Missing",
                    "action": "Skipped Missing Value",
                    "reason": "No reliable numeric value for imputation",
                    "new_value": "Remains Missing"
                })
                continue

            replacement_value = valid_values.median()

            # Keep countable values as whole numbers
            column_name = str(column).lower()

            count_keywords = [
                "age", "count", "number", "num_", "quantity",
                "qty", "total", "children", "orders"
            ]

            if any(keyword in column_name for keyword in count_keywords):
                replacement_value = int(round(replacement_value))


        # ====================================================
        # CATEGORICAL / TEXT COLUMNS
        # ====================================================

        else:

            # Do not guess missing email addresses.
            if "email" in str(column).lower():
                cleaning_actions.append({
                    "row_index": "N/A",
                    "column": column,
                    "original_value": "Missing",
                    "action": "Flagged for Review",
                    "reason": "Missing email cannot be safely inferred",
                    "new_value": "Remains Missing"
            })
            continue

            mode_values = (
                cleaned_df[column].mode()
            )

            if not mode_values.empty:

                replacement_value = (
                    mode_values.iloc[0]
                )

            else:

                replacement_value = "Unknown"

        # ====================================================
        # APPLY MISSING VALUE REPLACEMENT
        # ====================================================

        cleaned_df.loc[
            missing_mask,
            column
        ] = replacement_value

        cleaning_actions.extend(
            {
                "row_index": row_index,
                "column": column,
                "original_value": None,
                "action": "Filled Missing Value",
                "reason": "Missing value",
                "new_value": replacement_value
            }
            for row_index in missing_indices
        )

    # ========================================================
    # ANOMALY CLEANING
    # ========================================================

    if (
        anomaly_report is not None
        and not anomaly_report.empty
        and "column" in anomaly_report.columns
        and "row_index" in anomaly_report.columns
    ):

        for column in cleaned_df.columns:

            # Skip identifier columns.
            if (
                str(column).lower() == "id"
                or str(column).lower().endswith("_id")
            ):
                continue

            # Skip code-like columns.
            if _is_code_like_column(column):
                continue

            column_anomalies = anomaly_report[
                anomaly_report["column"] == column
            ]

            if column_anomalies.empty:
                continue


            # Handle numeric and invalid-numeric anomalies.
            for row_index in column_anomalies["row_index"]:

                if row_index not in cleaned_df.index:
                    continue

                original_value = cleaned_df.at[
                    row_index, column
                ]

                if pd.isna(original_value):
                    continue

                # Preserve valid ages and scores.
                column_name = str(column).lower()

                numeric_original = pd.to_numeric(
                    original_value,
                    errors="coerce"
                )

                if pd.notna(numeric_original):

                    if (
                        "age" in column_name
                        and 0 <= numeric_original <= 120
                    ):
                        continue

                    if (
                        any(word in column_name for word in [
                            "score", "mark", "percentage"
                        ])
                        and 0 <= numeric_original <= 100
                    ):
                        continue

                # Convert values to numeric for calculating
                # the replacement. Invalid text becomes NaN.
                numeric_values = pd.to_numeric(
                    cleaned_df[column],
                    errors="coerce"
                )

                valid_values = numeric_values.dropna()

                anomaly_indices = column_anomalies["row_index"]

                valid_values = valid_values.loc[
                    ~valid_values.index.isin(anomaly_indices)
                ]

                # Keep realistic ages only.
                if "age" in str(column).lower():
                    valid_values = valid_values[
                        valid_values.between(0, 120)
                    ]

                # Marks should be within the 0–100 range.
                if "mark" in str(column).lower():
                    valid_values = valid_values[
                        valid_values.between(0, 100)
                    ]

                if valid_values.empty:
                    cleaning_actions.append({
                        "row_index": row_index,
                        "column": column,
                        "original_value": original_value,
                        "action": "Flagged for Review",
                        "reason": "No reliable numeric values for replacement",
                        "new_value": original_value
                    })
                    continue

                replacement_value = valid_values.median()

                # Keep countable values as whole numbers.
                column_name = str(column).lower()

                count_keywords = [
                    "age", "count", "number", "num_",
                    "quantity", "qty", "children", "orders"
                ]

                if any(
                    keyword in column_name
                    for keyword in count_keywords
                ):
                    replacement_value = int(
                        round(replacement_value)
                    )

                # Convert the column to numeric before assigning the replacement.
                cleaned_df[column] = pd.to_numeric(
                    cleaned_df[column],
                    errors="coerce"
                )

                # Allow decimal replacements in integer columns when necessary.
                if not float(replacement_value).is_integer():
                    cleaned_df[column] = cleaned_df[column].astype("float64")

                # Assign the numeric replacement.
                cleaned_df.at[row_index, column] = replacement_value

                cleaning_actions.append({
                    "row_index": row_index,
                    "column": column,
                    "original_value": original_value,
                    "action": "Replaced Anomaly",
                    "reason": "Replaced using median of valid numeric values",
                    "new_value": replacement_value
                })

            # Restore numeric dtype when the column is now numeric.
            cleaned_df[column] = pd.to_numeric(
                cleaned_df[column],
                errors="coerce"
            )




    # ========================================================
    # CLEANING REPORT
    # ========================================================

    if cleaning_actions:

        cleaning_report = pd.DataFrame(
            cleaning_actions
        )

    else:

        cleaning_report = pd.DataFrame(
            columns=[
                "row_index",
                "column",
                "original_value",
                "action",
                "reason",
                "new_value"
            ]
        )

    return cleaned_df, cleaning_report
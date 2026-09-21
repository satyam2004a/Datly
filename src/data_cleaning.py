import pandas as pd


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

            replacement_value = (
                cleaned_df[column].median()
            )

        # ====================================================
        # CATEGORICAL / TEXT COLUMNS
        # ====================================================

        else:

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
    ):

        for column, column_anomalies in (
            anomaly_report.groupby("column")
        ):

            if column not in cleaned_df.columns:
                continue

            # Never alter identifier columns.
            if column in protected_columns:
                continue

            # Never automatically replace anomalies
            # in code-like columns.
            if _is_code_like_column(column):
                continue

            anomaly_indices = (
                column_anomalies["row_index"]
            )

            valid_indices = anomaly_indices[
                anomaly_indices.isin(
                    cleaned_df.index
                )
            ]

            if valid_indices.empty:
                continue

            # ------------------------------------------------
            # NUMERIC ANOMALIES
            # ------------------------------------------------

            if pd.api.types.is_numeric_dtype(
                cleaned_df[column]
            ):

                replacement_value = (
                    cleaned_df[column].median()
                )

                if pd.isna(replacement_value):
                    continue

                original_values = (
                    cleaned_df.loc[
                        valid_indices,
                        column
                    ].copy()
                )

                cleaned_df.loc[
                    valid_indices,
                    column
                ] = replacement_value

                cleaning_actions.extend(
                    {
                        "row_index": row_index,
                        "column": column,
                        "original_value": original_value,
                        "action": "Replaced Anomaly",
                        "reason": "Statistical anomaly",
                        "new_value": replacement_value
                    }
                    for (
                        row_index,
                        original_value
                    ) in original_values.items()
                )

            # ------------------------------------------------
            # CATEGORICAL ANOMALIES
            # ------------------------------------------------

            else:

                mode_values = (
                    cleaned_df[column].mode()
                )

                if mode_values.empty:
                    continue

                # Do not guess when multiple values
                # have the same frequency.
                if len(mode_values) > 1:
                    continue

                replacement_value = (
                    mode_values.iloc[0]
                )

                original_values = (
                    cleaned_df.loc[
                        valid_indices,
                        column
                    ].copy()
                )

                cleaned_df.loc[
                    valid_indices,
                    column
                ] = replacement_value

                cleaning_actions.extend(
                    {
                        "row_index": row_index,
                        "column": column,
                        "original_value": original_value,
                        "action": "Replaced Anomaly",
                        "reason": "Categorical anomaly",
                        "new_value": replacement_value
                    }
                    for (
                        row_index,
                        original_value
                    ) in original_values.items()
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
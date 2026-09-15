import pandas as pd


# =========================================================
# DATA CLEANING
# =========================================================

def clean_dataset(df, anomaly_report):
    """
    Clean a dataset using detected quality issues
    and anomalies.

    The function returns:
    - cleaned DataFrame
    - cleaning report

    Identifier columns ending with '_id' are protected
    from anomaly-based value replacement.
    """

    cleaned_df = df.copy()

    cleaning_actions = []

    # ---------------------------------------------------------
    # PROTECTED IDENTIFIER COLUMNS
    # ---------------------------------------------------------

    protected_columns = {
        column
        for column in cleaned_df.columns
        if str(column).lower().endswith("_id")
    }

    # Keep the generic 'id' column protected as well.
    if "id" in cleaned_df.columns:
        protected_columns.add("id")

    # ---------------------------------------------------------
    # REMOVE EXACT DUPLICATE ROWS
    # ---------------------------------------------------------

    duplicate_rows = cleaned_df.duplicated()

    duplicate_indices = cleaned_df.index[
        duplicate_rows
    ].tolist()

    if duplicate_indices:

        cleaned_df = cleaned_df.drop_duplicates()

        for row_index in duplicate_indices:

            cleaning_actions.append({
                "row_index": row_index,
                "column": "N/A",
                "original_value": "Duplicate Row",
                "action": "Removed Duplicate",
                "reason": "Exact duplicate record",
                "new_value": "Row Removed"
            })

    # ---------------------------------------------------------
    # HANDLE MISSING VALUES
    # ---------------------------------------------------------

    for column in cleaned_df.columns:

        missing_indices = cleaned_df.index[
            cleaned_df[column].isna()
        ].tolist()

        if not missing_indices:
            continue

        if pd.api.types.is_numeric_dtype(
            cleaned_df[column]
        ):

            replacement_value = (
                cleaned_df[column].median()
            )

        else:

            mode_values = cleaned_df[column].mode()

            if not mode_values.empty:
                replacement_value = mode_values.iloc[0]
            else:
                replacement_value = "Unknown"

        for row_index in missing_indices:

            cleaning_actions.append({
                "row_index": row_index,
                "column": column,
                "original_value": None,
                "action": "Filled Missing Value",
                "reason": "Missing value",
                "new_value": replacement_value
            })

        cleaned_df.loc[
            missing_indices,
            column
        ] = replacement_value

    # ---------------------------------------------------------
    # HANDLE DETECTED ANOMALIES
    # ---------------------------------------------------------

    if (
        anomaly_report is not None
        and not anomaly_report.empty
    ):

        for _, anomaly in anomaly_report.iterrows():

            row_index = anomaly.get(
                "row_index"
            )

            column = anomaly.get(
                "column"
            )

            if row_index not in cleaned_df.index:
                continue

            if column not in cleaned_df.columns:
                continue

            # -------------------------------------------------
            # PROTECT IDENTIFIER COLUMNS
            # -------------------------------------------------

            if column in protected_columns:
                continue

            original_value = cleaned_df.at[
                row_index,
                column
            ]

            # -------------------------------------------------
            # NUMERICAL ANOMALIES
            # -------------------------------------------------

            if pd.api.types.is_numeric_dtype(
                cleaned_df[column]
            ):

                values_without_current_row = (
                    cleaned_df.loc[
                        cleaned_df.index != row_index,
                        column
                    ]
                )

                replacement_value = (
                    values_without_current_row.median()
                )

                if pd.isna(replacement_value):
                    continue

                cleaned_df.at[
                    row_index,
                    column
                ] = replacement_value

                cleaning_actions.append({
                    "row_index": row_index,
                    "column": column,
                    "original_value": original_value,
                    "action": "Replaced Anomaly",
                    "reason": "Statistical anomaly",
                    "new_value": replacement_value
                })

            # -------------------------------------------------
            # CATEGORICAL ANOMALIES
            # -------------------------------------------------

            else:

                mode_values = cleaned_df[column].mode()

                if mode_values.empty:
                    continue

                replacement_value = mode_values.iloc[0]

                cleaned_df.at[
                    row_index,
                    column
                ] = replacement_value

                cleaning_actions.append({
                    "row_index": row_index,
                    "column": column,
                    "original_value": original_value,
                    "action": "Replaced Anomaly",
                    "reason": "Categorical anomaly",
                    "new_value": replacement_value
                })

    # ---------------------------------------------------------
    # CREATE CLEANING REPORT
    # ---------------------------------------------------------

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
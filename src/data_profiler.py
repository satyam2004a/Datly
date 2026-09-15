import pandas as pd


def profile_dataset(df):
    """
    Build a professional, reusable profile of a DataFrame.

    The function only analyzes the data. It does not modify the
    original DataFrame.
    """

    profile = {}

    # ---------------------------------------------------------
    # DATASET OVERVIEW
    # ---------------------------------------------------------

    profile["rows"] = int(df.shape[0])
    profile["columns"] = int(df.shape[1])
    profile["column_names"] = list(df.columns)

    # ---------------------------------------------------------
    # COLUMN TYPE DETECTION
    # ---------------------------------------------------------

    profile["numerical_columns"] = list(
        df.select_dtypes(include="number").columns
    )

    profile["categorical_columns"] = list(
        df.select_dtypes(
            include=["object", "string", "category"]
        ).columns
    )

    # ---------------------------------------------------------
    # BASIC COLUMN METRICS
    # ---------------------------------------------------------

    missing_values = df.isnull().sum()
    unique_values = df.nunique(dropna=True)

    if profile["rows"] > 0:
        missing_percentages = (
            missing_values / profile["rows"] * 100
        ).round(2)
    else:
        missing_percentages = pd.Series(
            0.0,
            index=df.columns
        )

    profile["missing_values"] = (
        missing_values.astype(int).to_dict()
    )

    profile["missing_percentages"] = (
        missing_percentages.to_dict()
    )

    profile["unique_values"] = (
        unique_values.astype(int).to_dict()
    )

    # ---------------------------------------------------------
    # NUMERICAL STATISTICS
    # ---------------------------------------------------------

    if profile["numerical_columns"]:
        profile["numerical_statistics"] = df[
            profile["numerical_columns"]
        ].describe().to_dict()
    else:
        profile["numerical_statistics"] = {}

    # ---------------------------------------------------------
    # COLUMN PROFILE
    # ---------------------------------------------------------

    column_profile = []

    for column in df.columns:
        column_profile.append({
            "Column": column,
            "Data Type": str(df[column].dtype),
            "Missing Values": int(missing_values[column]),
            "Missing %": float(
                missing_percentages[column]
            ),
            "Unique Values": int(unique_values[column])
        })

    profile["column_profile"] = pd.DataFrame(
        column_profile
    )

    return profile


if __name__ == "__main__":

    df = pd.read_csv("data/students.csv")

    result = profile_dataset(df)

    print("\n========== DATASET PROFILE ==========")
    print("Rows:", result["rows"])
    print("Columns:", result["columns"])
    print("Numerical:", result["numerical_columns"])
    print("Categorical:", result["categorical_columns"])

    print("\nColumn Profile:")
    print(result["column_profile"])

    print("\nNumerical Statistics:")
    print(result["numerical_statistics"])
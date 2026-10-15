"""Data loading and quality inspection for the Heart Disease project."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "heart_disease.csv"


def load_raw_data(path=RAW_DATA_PATH):
    """
    Load the original downloaded dataset without modifying its values.

    Parameters
    ----------
    path : str or pathlib.Path
        Location of the original CSV file.

    Returns
    -------
    pandas.DataFrame
        Original dataset.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            "Raw dataset not found. Run scripts/download_data.py first."
        )

    return pd.read_csv(path)


def inspect_data(df):
    """
    Generate a structured data-quality report.

    Checks:
    - Dataset dimensions.
    - Column names and data types.
    - Missing values.
    - Question-mark placeholders.
    - Duplicate rows.
    - Unique values in each column.
    - Original target distribution.
    """

    report = {
        "dataset_shape": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
        },
        "column_names": df.columns.tolist(),
        "data_types": {
            column: str(dtype)
            for column, dtype in df.dtypes.items()
        },
        "missing_values": {
            column: int(count)
            for column, count in df.isna().sum().items()
        },
        "question_mark_values": {
            column: int(count)
            for column, count in df.astype(str)
            .eq("?")
            .sum()
            .items()
        },
        "duplicate_rows": int(df.duplicated().sum()),
        "unique_values": {
            column: int(df[column].nunique(dropna=False))
            for column in df.columns
        },
    }

    if "num" in df.columns:
        target_counts = df["num"].value_counts(dropna=False).sort_index()

        report["original_target_distribution"] = {
            str(label): int(count)
            for label, count in target_counts.items()
        }

    return report


def print_data_quality_report(report):
    """Print a readable summary of the data-quality report."""

    print("=" * 65)
    print("HEART DISEASE DATA QUALITY REPORT")
    print("=" * 65)

    shape = report["dataset_shape"]

    print("\n1. Dataset dimensions")
    print(f"Rows: {shape['rows']}")
    print(f"Columns: {shape['columns']}")

    print("\n2. Column names")
    print(report["column_names"])

    print("\n3. Data types")
    for column, dtype in report["data_types"].items():
        print(f"{column}: {dtype}")

    print("\n4. Missing values")
    for column, count in report["missing_values"].items():
        print(f"{column}: {count}")

    print("\n5. Question-mark placeholders")
    for column, count in report["question_mark_values"].items():
        print(f"{column}: {count}")

    print("\n6. Duplicate rows")
    print(report["duplicate_rows"])

    print("\n7. Number of unique values per column")
    for column, count in report["unique_values"].items():
        print(f"{column}: {count}")

    if "original_target_distribution" in report:
        print("\n8. Original target distribution")
        for label, count in report[
            "original_target_distribution"
        ].items():
            print(f"num = {label}: {count}")

    print("\n" + "=" * 65)



def clean_data(df):
    """
    Validate and prepare the Heart Disease dataset.

    Operations:
    1. Check the expected columns.
    2. Convert question-mark missing-value markers to NaN.
    3. Validate the original target.
    4. Convert the original target into a binary target.
    5. Remove the original target column to prevent leakage.

    Missing feature values are intentionally preserved for
    imputation inside the model preprocessing pipeline.
    """

    import numpy as np

    expected_columns = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "num",
    ]

    # Work on a copy so the original DataFrame remains unchanged.
    df = df.copy()

    # Validate the dataset schema.
    missing_columns = set(expected_columns) - set(df.columns)
    unexpected_columns = set(df.columns) - set(expected_columns)

    if missing_columns or unexpected_columns:
        raise ValueError(
            "Unexpected dataset schema. "
            f"Missing columns: {sorted(missing_columns)}. "
            f"Unexpected columns: {sorted(unexpected_columns)}."
        )

    # Use a consistent representation for missing-value markers.
    df = df.replace("?", np.nan)

    # Convert all expected columns to numeric values.
    for column in expected_columns:
        converted = pd.to_numeric(df[column], errors="coerce")

        # Do not silently accept invalid, non-missing strings.
        invalid_values = df[column].notna() & converted.isna()

        if invalid_values.any():
            examples = df.loc[invalid_values, column].unique()[:5]

            raise ValueError(
                f"Invalid numeric values in {column}: {examples}"
            )

        df[column] = converted

    # The target must be present for supervised learning.
    if df["num"].isna().any():
        raise ValueError("The original target column contains missing values.")

    # Verify that target values are the expected integer severity levels.
    if not df["num"].isin([0, 1, 2, 3, 4]).all():
        raise ValueError(
            "Unexpected target values. Expected values from 0 to 4."
        )

    # Convert disease severity into a binary classification target.
    df["target"] = (df["num"] > 0).astype("int64")

    # Remove the original severity target to prevent accidental leakage.
    df = df.drop(columns=["num"])

    # Check that the binary target contains both classes.
    if set(df["target"].unique()) != {0, 1}:
        raise ValueError(
            "The prepared target must contain both classes: 0 and 1."
        )

    return df
"""Prepare the Heart Disease dataset for exploratory analysis and modelling."""

import json

from heart_mlops.data import (
    PROJECT_ROOT,
    clean_data,
    load_raw_data,
)


def main():
    """Load, validate, prepare and save the dataset."""

    print("Loading the original dataset...")

    raw_df = load_raw_data()

    print("Original shape:", raw_df.shape)

    print("\nPreparing the dataset...")

    clean_df = clean_data(raw_df)

    # Save the prepared dataset.
    output_dir = PROJECT_ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "heart_disease_clean.csv"
    clean_df.to_csv(output_path, index=False)

    # Record a summary of the preparation results.
    target_counts = clean_df["target"].value_counts().sort_index()

    report = {
        "original_rows": int(raw_df.shape[0]),
        "prepared_rows": int(clean_df.shape[0]),
        "prepared_columns": clean_df.columns.tolist(),
        "missing_values_after_preparation": {
            column: int(count)
            for column, count in clean_df.isna().sum().items()
        },
        "target_mapping": {
            "0": "No heart disease (original num = 0)",
            "1": "Heart disease present (original num > 0)",
        },
        "target_distribution": {
            str(label): int(count)
            for label, count in target_counts.items()
        },
    }

    report_path = PROJECT_ROOT / "reports" / "preparation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\nPreparation completed successfully.")

    print("\nPrepared shape:", clean_df.shape)

    print("\nMissing values in prepared dataset:")
    print(clean_df.isna().sum())

    print("\nBinary target distribution:")
    print(clean_df["target"].value_counts().sort_index())

    print("\nPrepared dataset saved to:")
    print(output_path)

    print("\nPreparation report saved to:")
    print(report_path)


if __name__ == "__main__":
    main()
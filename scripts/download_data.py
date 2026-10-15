"""
Download the Heart Disease dataset from the UCI Machine Learning Repository.

Dataset ID: 45
Source: https://archive.ics.uci.edu/dataset/45/heart+disease
"""

import json
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


# Find the project root independently of the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def download_heart_disease():
    """Download the dataset and save the raw data and metadata."""

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading Heart Disease dataset from UCI...")

    # Fetch dataset ID 45 from the UCI repository.
    heart_disease = fetch_ucirepo(id=45)

    # Retrieve features and target separately.
    features = heart_disease.data.features
    targets = heart_disease.data.targets

    # Combine features and target without modifying their original values.
    dataset = pd.concat([features, targets], axis=1)

    # Save the downloaded data.
    data_path = RAW_DATA_DIR / "heart_disease.csv"
    dataset.to_csv(data_path, index=False)

    # Save dataset variable descriptions.
    variables_path = RAW_DATA_DIR / "variables.csv"
    heart_disease.variables.to_csv(variables_path, index=False)

    # Save selected metadata for documentation and reproducibility.
    metadata = {
        "dataset_id": heart_disease.metadata.uci_id,
        "dataset_name": heart_disease.metadata.name,
        "source": heart_disease.metadata.repository_url,
        "num_instances": heart_disease.metadata.num_instances,
        "num_features": heart_disease.metadata.num_features,
        "target_column": heart_disease.metadata.target_col,
        "license": "CC BY 4.0",
        "citation": (
            "Janosi, A., Steinbrunn, W., Pfisterer, H., "
            "& Detrano, R. (1989). Heart Disease. "
            "UCI Machine Learning Repository. "
            "https://doi.org/10.24432/C52P4X"
        ),
    }

    metadata_path = RAW_DATA_DIR / "metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    # Print a summary so we can verify the download.
    print("\nDownload completed successfully.")
    print("Dataset saved to:", data_path)
    print("Variables saved to:", variables_path)
    print("Metadata saved to:", metadata_path)

    print("\nDataset shape:", dataset.shape)
    print("\nColumn names:")
    print(dataset.columns.tolist())

    print("\nFirst five rows:")
    print(dataset.head())

    print("\nData types:")
    print(dataset.dtypes)

    print("\nMissing-value representations:")
    print(dataset.isin(["?", ""]).sum())

    print("\nMissing values recognised by pandas:")
    print(dataset.isna().sum())


if __name__ == "__main__":
    download_heart_disease()
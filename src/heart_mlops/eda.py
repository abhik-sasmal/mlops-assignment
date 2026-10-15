"""Exploratory data analysis for the UCI Heart Disease dataset."""

import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from heart_mlops.data import PROJECT_ROOT


DATA_PATH = PROJECT_ROOT / "data" / "processed" / "heart_disease_clean.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
REPORT_PATH = PROJECT_ROOT / "reports" / "eda_report.json"


# Features whose integer values represent categories, not continuous quantities.
CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]

NUMERICAL_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
]


def load_prepared_data():
    """Load the prepared dataset and validate its essential structure."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found: {DATA_PATH}. "
            "Run scripts/prepare_data.py first."
        )

    df = pd.read_csv(DATA_PATH)

    if "target" not in df.columns:
        raise ValueError("The prepared dataset must contain a target column.")

    if set(df["target"].unique()) != {0, 1}:
        raise ValueError("The target must contain both classes: 0 and 1.")

    return df


def generate_eda_report(df):
    """Calculate descriptive statistics and data-quality information."""
    report = {
        "dataset_shape": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
        },
        "target_distribution": {
            str(label): int(count)
            for label, count in df["target"].value_counts().sort_index().items()
        },
        "target_percentages": {
            str(label): round(float(percentage), 2)
            for label, percentage in (
                df["target"].value_counts(normalize=True).sort_index() * 100
            ).items()
        },
        "missing_values": {
            column: int(count)
            for column, count in df.isna().sum().items()
        },
        "numerical_summary": (
            df[NUMERICAL_FEATURES]
            .describe()
            .round(2)
            .to_dict()
        ),
        "categorical_unique_values": {
            column: sorted(
                int(value)
                for value in df[column].dropna().unique()
            )
            for column in CATEGORICAL_FEATURES
        },
        "numerical_means_by_target": (
            df.groupby("target")[NUMERICAL_FEATURES]
            .mean()
            .round(2)
            .to_dict()
        ),
    }

    return report


def create_plots(df):
    """Generate plots used in the analysis and assignment report."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    # 1. Target distribution
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(data=df, x="target")
    ax.set_title("Heart Disease Target Distribution")
    ax.set_xlabel("Target (0 = absent, 1 = present)")
    ax.set_ylabel("Number of observations")

    for container in ax.containers:
        ax.bar_label(container)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "target_distribution.png", dpi=150)
    plt.close()

    # 2. Missing values
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    plt.figure(figsize=(7, 5))
    if missing.empty:
        plt.text(0.5, 0.5, "No missing values", ha="center", va="center")
        plt.axis("off")
    else:
        ax = sns.barplot(x=missing.index, y=missing.values)
        ax.set_title("Missing Values by Feature")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Number of missing observations")
        for container in ax.containers:
            ax.bar_label(container)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "missing_values.png", dpi=150)
    plt.close()

    # 3. Numerical feature distributions
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes = axes.flatten()

    for index, feature in enumerate(NUMERICAL_FEATURES):
        sns.histplot(
            data=df,
            x=feature,
            hue="target",
            kde=True,
            element="step",
            stat="density",
            common_norm=False,
            ax=axes[index],
        )
        axes[index].set_title(f"Distribution of {feature}")

    axes[-1].axis("off")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "numerical_distributions.png", dpi=150)
    plt.close(fig)

    # 4. Numerical feature correlation
    plt.figure(figsize=(9, 7))
    correlation = df[NUMERICAL_FEATURES + ["target"]].corr()
    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
    )
    plt.title("Correlation Among Numerical Features and Target")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()

    # 5. Categorical feature counts by target
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    axes = axes.flatten()

    for index, feature in enumerate(CATEGORICAL_FEATURES):
        sns.countplot(
            data=df,
            x=feature,
            hue="target",
            ax=axes[index],
        )
        axes[index].set_title(f"{feature} by target")
        axes[index].set_xlabel(feature)
        axes[index].set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "categorical_distributions.png", dpi=150)
    plt.close(fig)


def main():
    print("Loading prepared dataset...")
    df = load_prepared_data()

    print("\nDataset shape:", df.shape)
    print("\nTarget distribution:")
    print(df["target"].value_counts().sort_index())

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nNumerical feature summary:")
    print(df[NUMERICAL_FEATURES].describe().round(2))

    report = generate_eda_report(df)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    create_plots(df)

    print("\nEDA completed successfully.")
    print("Report:", REPORT_PATH)
    print("Figures directory:", FIGURES_DIR)


if __name__ == "__main__":
    main()
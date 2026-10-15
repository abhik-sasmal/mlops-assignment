"""Run data-quality checks and save the results."""

import json

from heart_mlops.data import (
    PROJECT_ROOT,
    inspect_data,
    load_raw_data,
    print_data_quality_report,
)


def main():
    """Execute the data-quality inspection workflow."""

    print("Loading the original Heart Disease dataset...")

    df = load_raw_data()

    report = inspect_data(df)

    print_data_quality_report(report)

    # Save the machine-readable report.
    report_dir = PROJECT_ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    report_path = report_dir / "data_quality_report.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\nData-quality report saved to:")
    print(report_path)


if __name__ == "__main__":
    main()
"""Tests for Heart Disease data preparation."""

import pandas as pd
import pytest

from heart_mlops.data import clean_data


@pytest.fixture
def sample_raw_data():
    """Create a small sample resembling the original UCI dataset."""

    return pd.DataFrame(
        {
            "age": [50, 60, 45, 55],
            "sex": [1, 0, 1, 0],
            "cp": [1, 4, 2, 3],
            "trestbps": [120, 140, 130, 125],
            "chol": [200, 250, 220, 230],
            "fbs": [0, 1, 0, 0],
            "restecg": [0, 1, 0, 2],
            "thalach": [160, 130, 170, 145],
            "exang": [0, 1, 0, 1],
            "oldpeak": [0.0, 2.0, 0.5, 1.0],
            "slope": [1, 2, 1, 3],
            "ca": [0.0, None, 1.0, 0.0],
            "thal": [3.0, 7.0, None, 6.0],
            "num": [0, 2, 0, 1],
        }
    )


def test_target_is_converted_to_binary(sample_raw_data):
    """All original positive disease levels must become class 1."""

    prepared = clean_data(sample_raw_data)

    assert prepared["target"].tolist() == [0, 1, 0, 1]
    assert "num" not in prepared.columns


def test_missing_feature_values_are_preserved(sample_raw_data):
    """Missing values remain available for pipeline imputation."""

    prepared = clean_data(sample_raw_data)

    assert prepared["ca"].isna().sum() == 1
    assert prepared["thal"].isna().sum() == 1
    assert prepared["target"].isna().sum() == 0


def test_unexpected_column_is_rejected(sample_raw_data):
    """Unexpected columns must not silently enter the model."""

    sample_raw_data["unexpected_feature"] = 123

    with pytest.raises(ValueError):
        clean_data(sample_raw_data)


def test_invalid_target_is_rejected(sample_raw_data):
    """Targets outside the original UCI target range must be rejected."""

    sample_raw_data.loc[0, "num"] = 5

    with pytest.raises(ValueError):
        clean_data(sample_raw_data)
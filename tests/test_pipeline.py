"""Tests for preprocessing and classification pipelines."""

import numpy as np
import pandas as pd
import pytest

from heart_mlops.pipeline import (
    EXPECTED_FEATURES,
    build_logistic_regression,
    build_random_forest,
)


@pytest.fixture
def sample_features():
    """Create a small dataset with categorical and numerical features."""

    rows = []

    for i in range(40):
        rows.append(
            {
                "age": 35 + (i % 35),
                "trestbps": 110 + (i % 50),
                "chol": 180 + (i % 100),
                "thalach": 100 + (i % 90),
                "oldpeak": float(i % 5),
                "sex": i % 2,
                "cp": 1 + (i % 4),
                "fbs": i % 2,
                "restecg": i % 3,
                "exang": i % 2,
                "slope": 1 + (i % 3),
                "ca": np.nan if i == 0 else i % 4,
                "thal": np.nan if i == 1 else [3, 6, 7][i % 3],
            }
        )

    return pd.DataFrame(rows, columns=EXPECTED_FEATURES)


@pytest.fixture
def sample_target():
    """Create a balanced binary target."""

    return pd.Series([0, 1] * 20, name="target")


@pytest.mark.parametrize(
    "model_builder",
    [
        build_logistic_regression,
        build_random_forest,
    ],
)
def test_pipeline_predicts_valid_classes(
    sample_features,
    sample_target,
    model_builder,
):
    """Each pipeline must train and produce valid binary predictions."""

    model = model_builder()
    model.fit(sample_features, sample_target)

    predictions = model.predict(sample_features)

    assert len(predictions) == len(sample_target)
    assert set(predictions).issubset({0, 1})


@pytest.mark.parametrize(
    "model_builder",
    [
        build_logistic_regression,
        build_random_forest,
    ],
)
def test_pipeline_returns_valid_probabilities(
    sample_features,
    sample_target,
    model_builder,
):
    """Predicted probabilities must be valid and sum to one."""

    model = model_builder()
    model.fit(sample_features, sample_target)

    probabilities = model.predict_proba(sample_features)

    assert probabilities.shape == (40, 2)
    assert np.isfinite(probabilities).all()
    assert (probabilities >= 0).all()
    assert (probabilities <= 1).all()

    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        np.ones(40),
    )
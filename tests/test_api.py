"""Tests for the FastAPI prediction service."""

from fastapi.testclient import TestClient

from heart_mlops import api


client = TestClient(api.app)


def test_root_endpoint():
    """The root endpoint should provide basic API information."""
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()
    assert data["message"] == "Heart Disease Prediction API"
    assert data["docs"] == "/docs"
    assert data["health"] == "/health"


def test_health_endpoint():
    """The health endpoint should report the model status."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
    }


def test_predict_endpoint():
    """A valid patient record should produce a prediction."""
    payload = {
        "age": 63,
        "trestbps": 145,
        "chol": 233,
        "thalach": 150,
        "oldpeak": 2.3,
        "sex": 1,
        "cp": 4,
        "fbs": 1,
        "restecg": 0,
        "exang": 1,
        "slope": 2,
        "ca": 0,
        "thal": 7,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()
    assert data["prediction"] in [0, 1]
    assert data["prediction_label"] in [
        "no_disease_recorded",
        "disease_recorded",
    ]
    assert data["model"] == "logistic_regression"
    assert data["disclaimer"] == (
        "Educational prediction only; not a clinical diagnosis."
    )

    probabilities = data["probabilities"]
    assert 0.0 <= probabilities["no_disease_recorded"] <= 1.0
    assert 0.0 <= probabilities["disease_recorded"] <= 1.0

    probability_sum = sum(probabilities.values())
    assert abs(probability_sum - 1.0) < 1e-6


def test_predict_rejects_missing_required_field():
    """A request missing a required feature should be rejected."""
    payload = {
        "age": 63,
        "trestbps": 145,
        "chol": 233,
        "thalach": 150,
        "oldpeak": 2.3,
        "sex": 1,
        "cp": 4,
        "fbs": 1,
        "restecg": 0,
        "exang": 1,
        "slope": 2,
        "ca": 0,
        # The required "thal" field is intentionally omitted.
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
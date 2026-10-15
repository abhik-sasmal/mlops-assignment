"""FastAPI application for heart-disease prediction."""

from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# Locate the project root and the saved Logistic Regression pipeline.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "logistic_regression.joblib"

app = FastAPI(
    title="Heart Disease Prediction API",
    description=(
        "Educational ML API that predicts the binary target "
        "defined for the UCI Heart Disease dataset."
    ),
    version="1.0.0",
)

# Load the saved pipeline once when the API module is imported.
model = None

try:
    model = joblib.load(MODEL_PATH)
except (FileNotFoundError, OSError, ValueError) as exc:
    print(f"Could not load model from {MODEL_PATH}: {exc}")


class PatientFeatures(BaseModel):
    """Input features expected by the trained model."""

    age: Optional[float] = Field(..., description="Age in years")
    trestbps: Optional[float] = Field(
        ..., description="Resting blood pressure (mm Hg)"
    )
    chol: Optional[float] = Field(
        ..., description="Serum cholesterol (mg/dl)"
    )
    thalach: Optional[float] = Field(
        ..., description="Maximum heart rate achieved"
    )
    oldpeak: Optional[float] = Field(
        ..., description="ST depression induced by exercise"
    )

    sex: Optional[int] = Field(..., description="Dataset code for sex")
    cp: Optional[int] = Field(..., description="Chest pain type code")
    fbs: Optional[int] = Field(
        ..., description="Fasting blood sugar category code"
    )
    restecg: Optional[int] = Field(
        ..., description="Resting ECG category code"
    )
    exang: Optional[int] = Field(
        ..., description="Exercise-induced angina category code"
    )
    slope: Optional[int] = Field(
        ..., description="ST segment slope category code"
    )
    ca: Optional[float] = Field(
        ..., description="Number of major vessels; may be missing"
    )
    thal: Optional[int] = Field(
        ..., description="Thalassemia category code; may be missing"
    )


@app.get("/")
def root():
    """Return basic API information."""
    return {
        "message": "Heart Disease Prediction API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    """Report whether the model was loaded successfully."""
    return {
        "status": "ok" if model is not None else "unavailable",
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict(patient: PatientFeatures):
    """Predict the binary target for one patient."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="The prediction model is not available.",
        )

    # Preserve the exact feature names and order used during training.
    patient_data = pd.DataFrame(
        [patient.model_dump()],
        columns=[
            "age",
            "trestbps",
            "chol",
            "thalach",
            "oldpeak",
            "sex",
            "cp",
            "fbs",
            "restecg",
            "exang",
            "slope",
            "ca",
            "thal",
        ],
    )

    try:
        prediction = int(model.predict(patient_data)[0])
        probabilities = model.predict_proba(patient_data)[0]
        class_labels = [int(label) for label in model.classes_]
        probability_by_class = {
            label: float(probability)
            for label, probability in zip(class_labels, probabilities)
        }
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Could not process the supplied features: {exc}",
        ) from exc

    return {
        "prediction": prediction,
        "prediction_label": (
            "disease_recorded" if prediction == 1 else "no_disease_recorded"
        ),
        "probabilities": {
            "no_disease_recorded": probability_by_class.get(0),
            "disease_recorded": probability_by_class.get(1),
        },
        "model": "logistic_regression",
        "disclaimer": (
            "Educational prediction only; not a clinical diagnosis."
        ),
    }

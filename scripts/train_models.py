"""Train and evaluate baseline heart disease classifiers."""

import json

import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)

from heart_mlops.data import PROJECT_ROOT
from heart_mlops.pipeline import (
    EXPECTED_FEATURES,
    build_logistic_regression,
    build_random_forest,
)


RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "heart_disease_clean.csv"
)

METRICS_PATH = PROJECT_ROOT / "reports" / "model_metrics.json"
MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"


def load_dataset():
    """Load the prepared dataset and separate features from the target."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. "
            "Run scripts/prepare_data.py first."
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = set(EXPECTED_FEATURES + ["target"])
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )

    X = df[EXPECTED_FEATURES].copy()
    y = df["target"].astype(int).copy()

    if X.empty:
        raise ValueError("The dataset contains no observations.")

    if y.isna().any():
        raise ValueError("The target contains missing values.")

    if set(y.unique()) != {0, 1}:
        raise ValueError("The target must contain exactly classes 0 and 1.")

    return X, y


def calculate_metrics(y_true, y_pred, y_probability):
    """Calculate classification metrics for binary classification."""

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_probability)),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).tolist(),
    }


def main():
    print("Loading dataset...")
    X, y = load_dataset()

    print("Dataset shape:", X.shape)
    print("\nTarget distribution:")
    print(y.value_counts().sort_index())

    # Keep the final test set separate from cross-validation.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nTraining observations:", len(X_train))
    print("Test observations:", len(X_test))

    models = {
        "logistic_regression": build_logistic_regression(),
        "random_forest": build_random_forest(),
    }

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    cv_scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    results = {
        "dataset": {
            "observations": int(len(X)),
            "features": int(X.shape[1]),
            "training_observations": int(len(X_train)),
            "test_observations": int(len(X_test)),
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "cv_folds": CV_FOLDS,
        },
        "models": {},
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for model_name, model in models.items():
        print("\n" + "=" * 60)
        print("Model:", model_name)
        print("=" * 60)

        # Cross-validation uses only the training partition.
        cv_results = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=cv_scoring,
            return_train_score=False,
            n_jobs=1,
            error_score="raise",
        )

        cv_summary = {}

        for metric_name in cv_scoring:
            scores = cv_results[f"test_{metric_name}"]

            cv_summary[metric_name] = {
                "mean": float(scores.mean()),
                "std": float(scores.std()),
                "fold_scores": [float(score) for score in scores],
            }

            print(
                f"CV {metric_name:>9}: "
                f"{scores.mean():.4f} "
                f"+/- {scores.std():.4f}"
            )

        # Fit a fresh copy on the complete training partition.
        fitted_model = clone(model)
        fitted_model.fit(X_train, y_train)

        # Evaluate exactly once on the held-out test partition.
        y_pred = fitted_model.predict(X_test)
        y_probability = fitted_model.predict_proba(X_test)[:, 1]

        test_metrics = calculate_metrics(
            y_test,
            y_pred,
            y_probability,
        )

        print("\nHeld-out test metrics:")

        for metric_name, metric_value in test_metrics.items():
            print(f"{metric_name}: {metric_value}")

        results["models"][model_name] = {
            "cross_validation": cv_summary,
            "test_metrics": test_metrics,
        }

        # Save the fitted end-to-end pipeline.
        import joblib

        model_path = MODELS_DIR / f"{model_name}.joblib"
        joblib.dump(fitted_model, model_path)

        print("Saved model:", model_path)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("\nTraining and evaluation completed.")
    print("Metrics report:", METRICS_PATH)


if __name__ == "__main__":
    main()
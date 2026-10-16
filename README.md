# Heart Disease Classification — MLOps Assignment

An end-to-end ML workflow using the UCI Cleveland Heart Disease dataset: data acquisition, cleaning, EDA, Logistic Regression and Random Forest, cross-validation, MLflow tracking, tests, FastAPI, Docker, and local Kubernetes with Kind.


## Recorded results

| Model | CV accuracy | CV F1 | CV ROC-AUC | Test accuracy | Test precision | Test recall | Test F1 | Test ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.847 | 0.825 | 0.903 | 0.885 | 0.839 | 0.929 | 0.881 | 0.966 |
| Random Forest | 0.814 | 0.794 | 0.896 | 0.869 | 0.813 | 0.929 | 0.867 | 0.949 |

Logistic Regression was selected for the saved API model based on the recorded evaluation.

## Repository structure

.github/workflows/tests.yml
artifacts/models/                 # saved fitted model pipelines
data/raw/                         # downloaded data and metadata
data/processed/                   # cleaned data
deployment/kind-config.yaml
deployment/kubernetes.yaml
reports/figures/                  # EDA figures
reports/*_report.json
reports/model_metrics.json
scripts/                           # download, inspect, prepare, EDA, train
src/heart_mlops/
  api.py                           # FastAPI service
  data.py                          # data loading and cleaning
  eda.py                           # plots and EDA
  pipeline.py                      # preprocessing and classifiers
tests/                             # data, pipeline, API tests
Dockerfile
.dockerignore
requirements.txt
README.md


## 1. Setup

Requires Python 3.9, Git, Docker, kubectl, and Kind for the Kubernetes section.

git clone https://github.com/abhik-sasmal/mlops-assignment.git
cd mlops-assignment
conda create -n mlops python=3.9 -y
conda activate mlops
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

If the `mlops` environment already exists, activate it instead of recreating it.

## 2. Data, preparation, EDA, and training

Run from the repository root:

python scripts/download_data.py
python scripts/inspect_data.py
python scripts/prepare_data.py
python scripts/run_eda.py
python scripts/train_models.py


The target is binarized as `num == 0 → 0` and `num > 0 → 1`. The pipeline imputes numerical features with medians and categorical features with the most frequent value; it standardizes numerical features and one-hot encodes categorical features. Preprocessing is fitted within each model pipeline.

Training uses a stratified 80/20 split (242 train, 61 test) with seed 42 and 5-fold stratified cross-validation on the training set. It saves model artifacts under `artifacts/models/`, metrics in `reports/model_metrics.json`, and runs in MLflow.

## 3. MLflow
Run : 

mlflow ui --host 127.0.0.1 --port 5000


Open http://127.0.0.1:5000. Run the command from the repository root so the UI can find the local `mlruns/` store. Stop with Ctrl+C.

## 4. Tests and GitHub Actions
PYTHONPATH=src python -m pytest tests/ -v

The last recorded local run passed 12 tests. The workflow is `.github/workflows/tests.yml`. Verify the latest GitHub Actions run is green before submission. Describe only the steps actually present in the workflow; do not claim linting or model training are in CI unless they have been added and verified.

## 5. Run the API locally
PYTHONPATH=src uvicorn heart_mlops.api:app --host 127.0.0.1 --port 8000


In another terminal:

curl http://127.0.0.1:8000/health


Expected: `{"status":"ok","model_loaded":true}`. Interactive docs: http://127.0.0.1:8000/docs.

Example request:

curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "age": 63, "trestbps": 145, "chol": 233, "thalach": 150,
    "oldpeak": 2.3, "sex": 1, "cp": 4, "fbs": 1, "restecg": 0,
    "exang": 1, "slope": 2, "ca": 0, "thal": 7
  }'

The response includes a binary prediction, label, class probabilities, model name, and educational disclaimer.

## 6. Docker
docker build -t heart-disease-api:1.0 .
docker run --rm --name heart-disease-api -p 8000:8000 heart-disease-api:1.0

Then test `/health`, `/docs`, and `/predict` at `http://127.0.0.1:8000`. If a container with that name already exists, remove it first with `docker rm -f heart-disease-api`. Inspect logs with `docker logs heart-disease-api`.

## 7. Local Kubernetes (Kind)

Start Docker first, then from the repository root:

kind create cluster --name heart-mlops --config deployment/kind-config.yaml
kind load docker-image heart-disease-api:1.0 --name heart-mlops
kubectl apply -f deployment/kubernetes.yaml
kubectl rollout status deployment/heart-disease-api
kubectl get deployments,pods,services

Keep this command running in one terminal:

kubectl port-forward service/heart-disease-api 8001:8000

In a second terminal, verify the service at `http://127.0.0.1:8001/health` and `http://127.0.0.1:8001/docs`. Send the same sample JSON to `http://127.0.0.1:8001/predict`.

Inspect deployment and logs:

kubectl describe deployment heart-disease-api
kubectl logs deployment/heart-disease-api --tail=50

The manifest has one replica, readiness/liveness probes, resource requests/limits, and a NodePort Service. Port-forwarding is the reliable host access method for this Kind configuration. This is local Kubernetes, not a public cloud deployment.

Clean up after collecting evidence:


kind delete cluster --name heart-mlops


## 8. Monitoring, evidence, limitations

Basic operational visibility comes from Uvicorn HTTP access logs and Kubernetes health probes. This is **not** a Prometheus/Grafana dashboard; do not claim a metrics dashboard unless one is implemented.

Capture screenshots of: (1) EDA target distribution and heatmap, (2) MLflow runs/metrics, (3) successful GitHub Actions run, (4) Docker container and health endpoint, (5) Docker prediction response, (6) Kubernetes pod/service status, (7) Kubernetes prediction response via port-forward, and (8) Kubernetes logs. Store them under `screenshots/` and add the important ones to the report.

## Architecture

UCI dataset → download/clean/EDA → preprocessing + two classifiers
            → cross-validation/test metrics + MLflow → saved model
            → FastAPI → Docker image → Kind Kubernetes Deployment/Service
            → health probes and HTTP access logs

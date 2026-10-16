FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY artifacts/models/logistic_regression.joblib ./artifacts/models/logistic_regression.joblib

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "heart_mlops.api:app", "--host", "0.0.0.0", "--port", "8000"]
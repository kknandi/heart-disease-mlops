# syntax=docker/dockerfile:1
# Heart disease risk API - production image (FastAPI + uvicorn, non-root).
FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_PATH=/app/models/model.joblib \
    LOG_LEVEL=INFO \
    PORT=8000

WORKDIR /app

# 1) dependencies first -> cached layer unless requirements change
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

# 2) application package
COPY pyproject.toml .
COPY src ./src
RUN pip install --no-deps . && rm -rf src pyproject.toml

# 3) trained model artifact (produced by `python -m heart_disease.train`)
COPY models/model.joblib models/metadata.json ./models/

RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"

CMD ["sh", "-c", "uvicorn heart_disease.api.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --no-access-log"]

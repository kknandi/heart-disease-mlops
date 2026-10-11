"""Heart disease prediction API.

    uvicorn heart_disease.api.main:app --host 0.0.0.0 --port 8000

Endpoints: POST /predict, GET /health, GET /model-info, GET /metrics, GET /docs
"""

from __future__ import annotations

import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from heart_disease import __version__, config
from heart_disease.api import monitoring as mon
from heart_disease.api.schemas import HealthResponse, PatientFeatures, PredictionResponse
from heart_disease.predict import load_metadata, load_model, predict_records

mon.configure_logging()
log = logging.getLogger("heart_disease.api")

_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9\-_.]{1,64}$")
_QUIET_PATHS = {"/metrics", "/health"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path(os.getenv("MODEL_PATH", config.MODEL_FILE))
    meta_path = Path(os.getenv("MODEL_METADATA_PATH", model_path.with_name("metadata.json")))
    app.state.model, app.state.metadata = None, {}
    try:
        app.state.model = load_model(model_path)
        app.state.metadata = load_metadata(meta_path)
        mon.MODEL_LOADED.set(1)
        mon.MODEL_INFO.info(
            {
                "name": str(app.state.metadata.get("model_name", "unknown")),
                "version": str(app.state.metadata.get("model_version", "unknown")),
            }
        )
        log.info("model loaded", extra={"model_path": str(model_path)})
    except Exception:
        mon.MODEL_LOADED.set(0)
        log.exception("model failed to load", extra={"model_path": str(model_path)})
    yield


app = FastAPI(
    title="Heart Disease Risk API",
    version=__version__,
    description="Predicts the presence of heart disease from 13 clinical attributes.",
    lifespan=lifespan,
)


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    incoming = request.headers.get("x-request-id", "")
    request_id = incoming if _REQUEST_ID_RE.match(incoming) else uuid.uuid4().hex
    request.state.request_id = request_id
    start = time.perf_counter()
    status_code = 500
    mon.REQUESTS_IN_PROGRESS.inc()
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        elapsed = time.perf_counter() - start
        mon.REQUESTS_IN_PROGRESS.dec()
        route = request.scope.get("route")
        endpoint = getattr(route, "path", "unmatched")  # template path -> bounded cardinality
        mon.REQUESTS_TOTAL.labels(request.method, endpoint, str(status_code)).inc()
        mon.REQUEST_LATENCY.labels(request.method, endpoint).observe(elapsed)
        log.log(
            logging.DEBUG if endpoint in _QUIET_PATHS else logging.INFO,
            "request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": status_code,
                "latency_ms": round(elapsed * 1000, 2),
                "client": request.client.host if request.client else None,
            },
        )


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {
        "service": "heart-disease-api",
        "version": __version__,
        "endpoints": ["/predict", "/health", "/model-info", "/metrics", "/docs"],
    }


@app.get("/health", response_model=HealthResponse)
def health(request: Request, response: Response) -> HealthResponse:
    loaded = request.app.state.model is not None
    meta = request.app.state.metadata
    if not loaded:
        response.status_code = 503
    return HealthResponse(
        status="ok" if loaded else "unavailable",
        model_loaded=loaded,
        model_name=meta.get("model_name"),
        model_version=meta.get("model_version"),
    )


@app.get("/model-info")
def model_info(request: Request) -> dict:
    meta = request.app.state.metadata
    if not meta:
        raise HTTPException(status_code=503, detail="Model metadata not available")
    keys = (
        "model_name",
        "model_version",
        "trained_at",
        "best_params",
        "cv_metrics",
        "test_metrics",
        "features",
        "derived_features",
        "decision_threshold",
    )
    return {k: meta.get(k) for k in keys}


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientFeatures, request: Request) -> PredictionResponse:
    model = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    meta = request.app.state.metadata
    request_id = request.state.request_id
    try:
        result = predict_records(
            model, [patient.model_dump()], threshold=meta.get("decision_threshold", 0.5)
        )[0]
    except Exception as exc:
        mon.PREDICTION_ERRORS.inc()
        log.exception("prediction failed", extra={"request_id": request_id})
        raise HTTPException(status_code=500, detail="Prediction failed") from exc

    mon.PREDICTIONS_TOTAL.labels(result["label"]).inc()
    mon.PREDICTION_PROBABILITY.observe(result["probability_disease"])
    # Only model outputs are logged - patient features are health data (PII) and stay out of logs.
    log.info(
        "prediction",
        extra={
            "request_id": request_id,
            "label": result["label"],
            "probability_disease": result["probability_disease"],
            "risk_band": result["risk_band"],
        },
    )
    return PredictionResponse(
        **result,
        model_name=str(meta.get("model_name", "unknown")),
        model_version=str(meta.get("model_version", "unknown")),
        request_id=request_id,
    )


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

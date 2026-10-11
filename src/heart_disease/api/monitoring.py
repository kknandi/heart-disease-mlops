"""Prometheus metrics and structured (JSON) logging for the API."""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime

from prometheus_client import Counter, Gauge, Histogram, Info

REQUESTS_TOTAL = Counter(
    "api_requests_total", "HTTP requests processed", ["method", "endpoint", "status_code"]
)
REQUEST_LATENCY = Histogram(
    "api_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)
REQUESTS_IN_PROGRESS = Gauge("api_requests_in_progress", "Requests currently being served")
PREDICTIONS_TOTAL = Counter("model_predictions_total", "Predictions served", ["label"])
PREDICTION_PROBABILITY = Histogram(
    "model_prediction_probability",
    "Predicted probability of heart disease (watch for drift)",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
PREDICTION_ERRORS = Counter("model_prediction_errors_total", "Failed predictions")
MODEL_LOADED = Gauge("model_loaded", "1 if a model is loaded and ready")
MODEL_INFO = Info("model", "Metadata of the served model")

_STD_ATTRS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message"}


class JsonFormatter(logging.Formatter):
    """One JSON object per line - easy to ship to Loki / ELK / CloudWatch."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload |= {k: v for k, v in record.__dict__.items() if k not in _STD_ATTRS}
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())

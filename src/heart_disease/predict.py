"""Inference helpers shared by the API, notebooks and CLI.

python -m heart_disease.predict --input sample_request.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from heart_disease import config

LABELS = {0: "no_heart_disease", 1: "heart_disease"}


def load_model(path: Path | str = config.MODEL_FILE) -> Pipeline:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}; run `python -m heart_disease.train`")
    return joblib.load(path)


def load_metadata(path: Path | str = config.MODEL_METADATA_FILE) -> dict[str, Any]:
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else {}


def risk_band(probability: float) -> str:
    if probability < 0.3:
        return "low"
    if probability < 0.7:
        return "moderate"
    return "high"


def predict_records(
    model: Pipeline, records: Iterable[Mapping[str, Any]], threshold: float = 0.5
) -> list[dict[str, Any]]:
    """Predict for raw patient dicts (missing keys / None are imputed by the pipeline)."""
    frame = pd.DataFrame(list(records)).reindex(columns=config.FEATURES).astype(float)
    proba = model.predict_proba(frame)[:, 1]
    results = []
    for p in proba:
        pred = int(p >= threshold)
        results.append(
            {
                "prediction": pred,
                "label": LABELS[pred],
                "probability_disease": round(float(p), 4),
                "confidence": round(float(p if pred else 1 - p), 4),
                "risk_band": risk_band(float(p)),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Score patients from a JSON file")
    parser.add_argument("--input", type=Path, required=True, help="JSON object or list")
    parser.add_argument("--model", type=Path, default=config.MODEL_FILE)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    records = payload if isinstance(payload, list) else [payload]
    print(json.dumps(predict_records(load_model(args.model), records), indent=2))


if __name__ == "__main__":
    main()

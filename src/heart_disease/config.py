"""Central configuration: paths, column definitions and validation rules."""

from __future__ import annotations

import os
from pathlib import Path

# Project root = two levels above src/heart_disease/ (overridable for containers / CI).
PROJECT_ROOT = Path(os.getenv("HD_PROJECT_ROOT", Path(__file__).resolve().parents[2]))

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_FILE = RAW_DATA_DIR / "processed.cleveland.data"
CLEAN_DATA_FILE = PROCESSED_DATA_DIR / "heart_disease_clean.csv"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_FILE = MODELS_DIR / "model.joblib"
MODEL_METADATA_FILE = MODELS_DIR / "metadata.json"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
)
MLFLOW_ARTIFACT_ROOT = (PROJECT_ROOT / "mlruns").as_uri()
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "heart-disease-classification")
REGISTERED_MODEL_NAME = "heart-disease-classifier"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# --- Dataset schema (UCI "processed.cleveland.data", 14 attributes) ---
RAW_COLUMNS = [
    "age",  # years
    "sex",  # 1 = male, 0 = female
    "cp",  # chest pain type: 1 typical angina, 2 atypical, 3 non-anginal, 4 asymptomatic
    "trestbps",  # resting blood pressure (mm Hg)
    "chol",  # serum cholesterol (mg/dl)
    "fbs",  # fasting blood sugar > 120 mg/dl
    "restecg",  # resting ECG: 0 normal, 1 ST-T abnormality, 2 LV hypertrophy
    "thalach",  # maximum heart rate achieved
    "exang",  # exercise induced angina
    "oldpeak",  # ST depression induced by exercise relative to rest
    "slope",  # slope of peak exercise ST segment: 1 up, 2 flat, 3 down
    "ca",  # number of major vessels coloured by fluoroscopy (0-3)
    "thal",  # 3 normal, 6 fixed defect, 7 reversible defect
    "num",  # diagnosis 0 (<50% narrowing) .. 4
]
TARGET = "target"

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
BINARY_FEATURES = ["sex", "fbs", "exang"]
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "thal"]
FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

# Physiologically plausible ranges; values outside are treated as missing.
VALID_RANGES: dict[str, tuple[float, float]] = {
    "age": (1, 120),
    "trestbps": (60, 250),
    "chol": (80, 700),
    "thalach": (50, 250),
    "oldpeak": (-3, 10),
    "ca": (0, 3),
}
ALLOWED_CATEGORIES: dict[str, set[int]] = {
    "sex": {0, 1},
    "fbs": {0, 1},
    "exang": {0, 1},
    "cp": {1, 2, 3, 4},
    "restecg": {0, 1, 2},
    "slope": {1, 2, 3},
    "thal": {3, 6, 7},
}

"""Request / response schemas with input validation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

EXAMPLE_PATIENT = {
    "age": 58,
    "sex": 1,
    "cp": 4,
    "trestbps": 140,
    "chol": 289,
    "fbs": 0,
    "restecg": 2,
    "thalach": 120,
    "exang": 1,
    "oldpeak": 1.8,
    "slope": 2,
    "ca": 2,
    "thal": 7,
}


class PatientFeatures(BaseModel):
    """Clinical measurements of one patient (UCI Cleveland attribute coding)."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE_PATIENT]})

    age: int = Field(..., ge=1, le=120, description="Age in years")
    sex: Literal[0, 1] = Field(..., description="1 = male, 0 = female")
    cp: Literal[1, 2, 3, 4] = Field(
        ..., description="Chest pain: 1 typical, 2 atypical, 3 non-anginal, 4 asymptomatic"
    )
    trestbps: float = Field(..., ge=60, le=250, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=80, le=700, description="Serum cholesterol (mg/dl)")
    fbs: Literal[0, 1] = Field(..., description="Fasting blood sugar > 120 mg/dl")
    restecg: Literal[0, 1, 2] = Field(..., description="Resting ECG result")
    thalach: float = Field(..., ge=50, le=250, description="Maximum heart rate achieved")
    exang: Literal[0, 1] = Field(..., description="Exercise induced angina")
    oldpeak: float = Field(..., ge=-3, le=10, description="ST depression induced by exercise")
    slope: Literal[1, 2, 3] = Field(..., description="Slope of peak exercise ST segment")
    ca: int | None = Field(
        None, ge=0, le=3, description="Major vessels coloured by fluoroscopy (optional)"
    )
    thal: Literal[3, 6, 7] | None = Field(
        None, description="3 normal, 6 fixed defect, 7 reversible defect (optional)"
    )


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="1 = heart disease predicted, 0 = not")
    label: str
    probability_disease: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1, description="Probability of the predicted class")
    risk_band: Literal["low", "moderate", "high"]
    model_name: str
    model_version: str
    request_id: str


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    model_loaded: bool
    model_name: str | None = None
    model_version: str | None = None

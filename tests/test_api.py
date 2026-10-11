import pytest
from fastapi.testclient import TestClient

from heart_disease.api.main import app


@pytest.fixture
def client(trained_model_dir, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(trained_model_dir / "model.joblib"))
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client_without_model(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    with TestClient(app) as c:
        yield c


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {
        "status": "ok",
        "model_loaded": True,
        "model_name": "test_logreg",
        "model_version": "test",
    }


def test_predict_returns_prediction_and_confidence(client, sample_patient):
    r = client.post("/predict", json=sample_patient)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert body["label"] in ("heart_disease", "no_heart_disease")
    assert 0.5 <= body["confidence"] <= 1
    assert body["model_name"] == "test_logreg"
    assert r.headers["X-Request-ID"] == body["request_id"]


def test_predict_accepts_missing_optional_fields(client, sample_patient):
    sample_patient.pop("ca")
    sample_patient.pop("thal")
    assert client.post("/predict", json=sample_patient).status_code == 200


def test_predict_propagates_valid_request_id(client, sample_patient):
    r = client.post("/predict", json=sample_patient, headers={"X-Request-ID": "abc-123"})
    assert r.json()["request_id"] == "abc-123"
    r = client.post("/predict", json=sample_patient, headers={"X-Request-ID": "bad id\n<x>"})
    assert r.json()["request_id"] != "bad id\n<x>"


@pytest.mark.parametrize(
    "patch",
    [{"age": -1}, {"cp": 9}, {"chol": "high"}, {"thal": 4}, {"unexpected": 1}],
)
def test_predict_rejects_invalid_input(client, sample_patient, patch):
    r = client.post("/predict", json=sample_patient | patch)
    assert r.status_code == 422


def test_predict_rejects_missing_required_field(client, sample_patient):
    sample_patient.pop("age")
    assert client.post("/predict", json=sample_patient).status_code == 422


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    assert r.json()["model_name"] == "test_logreg"


def test_metrics_exposes_request_and_prediction_counters(client, sample_patient):
    client.post("/predict", json=sample_patient)
    text = client.get("/metrics").text
    assert 'api_requests_total{endpoint="/predict",method="POST",status_code="200"}' in text
    assert "model_predictions_total" in text
    assert "api_request_duration_seconds_bucket" in text
    assert "model_loaded 1.0" in text


def test_service_unavailable_without_model(client_without_model, sample_patient):
    assert client_without_model.get("/health").status_code == 503
    assert client_without_model.post("/predict", json=sample_patient).status_code == 503

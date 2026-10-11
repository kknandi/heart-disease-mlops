import json

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from heart_disease import config
from heart_disease.evaluate import compute_metrics, feature_importance
from heart_disease.features import build_pipeline
from heart_disease.predict import load_model, predict_records, risk_band
from heart_disease.train import model_specs, run_training

from .conftest import make_raw_frame


def test_compute_metrics_perfect_and_threshold():
    y = np.array([0, 0, 1, 1])
    m = compute_metrics(y, np.array([0.1, 0.4, 0.6, 0.9]))
    assert m["accuracy"] == m["precision"] == m["recall"] == m["roc_auc"] == 1.0
    m = compute_metrics(y, np.array([0.1, 0.4, 0.6, 0.9]), threshold=0.95)
    assert m["recall"] == 0.0 and m["precision"] == 0.0


@pytest.mark.parametrize(
    "estimator", [LogisticRegression(max_iter=2000), RandomForestClassifier(n_estimators=50)]
)
def test_models_learn_signal(Xy, estimator):
    X, y = Xy
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    pipe = build_pipeline(estimator).fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()
    assert compute_metrics(y_te, proba)["roc_auc"] > 0.7


def test_feature_importance_for_linear_and_tree(Xy):
    X, y = Xy
    for est in (LogisticRegression(max_iter=2000), RandomForestClassifier(n_estimators=20)):
        imp = feature_importance(build_pipeline(est).fit(X, y))
        assert imp is not None and len(imp) == 24 and (imp >= 0).all()


@pytest.mark.parametrize("quick", [True, False])
def test_model_specs_grids_target_classifier_step(quick):
    specs = model_specs(quick)
    assert {"logistic_regression", "random_forest"} <= set(specs)
    for estimator, grid in specs.values():
        assert all(k.startswith("clf__") for k in grid)
        assert set(k.removeprefix("clf__") for k in grid) <= set(estimator.get_params())


def test_predict_records_output_contract(trained_model_dir, sample_patient):
    model = load_model(trained_model_dir / "model.joblib")
    no_optional = {k: v for k, v in sample_patient.items() if k not in ("ca", "thal")}
    results = predict_records(model, [sample_patient, no_optional])
    assert len(results) == 2
    for r in results:
        assert set(r) == {"prediction", "label", "probability_disease", "confidence", "risk_band"}
        assert r["prediction"] in (0, 1)
        assert 0.5 <= r["confidence"] <= 1.0
        expected = r["probability_disease"] if r["prediction"] else 1 - r["probability_disease"]
        assert r["confidence"] == pytest.approx(expected, abs=1e-4)


def test_load_model_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "missing.joblib")


@pytest.mark.parametrize(
    "p,band", [(0.05, "low"), (0.3, "moderate"), (0.69, "moderate"), (0.7, "high")]
)
def test_risk_band(p, band):
    assert risk_band(p) == band


def test_run_training_end_to_end(tmp_path):
    """Quick grid through the real training entry point incl. MLflow logging."""
    from heart_disease.data import clean_data

    data = tmp_path / "clean.csv"
    clean_data(make_raw_frame(200, seed=3)).to_csv(data, index=False)
    meta = run_training(
        data_path=data,
        model_dir=tmp_path / "models",
        figures_dir=tmp_path / "figures",
        tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
        experiment="pytest-experiment",
        artifact_root=(tmp_path / "mlruns").as_uri(),
        quick=True,
        register=False,
    )
    assert meta["model_name"] in {"logistic_regression", "random_forest"}
    assert (tmp_path / "models" / "model.joblib").exists()
    saved = json.loads((tmp_path / "models" / "metadata.json").read_text())
    assert saved["features"] == config.FEATURES
    assert 0.5 < saved["test_metrics"]["test_roc_auc"] <= 1.0
    assert (tmp_path / "figures" / "model_comparison.png").exists()

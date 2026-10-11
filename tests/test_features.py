import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from heart_disease import config
from heart_disease.features import (
    DERIVED_FEATURES,
    ClinicalFeatureAdder,
    build_pipeline,
    build_preprocessor,
    transformed_feature_names,
)


def _one_patient(**overrides) -> pd.DataFrame:
    base = dict.fromkeys(config.FEATURES, 0.0) | {
        "age": 60.0,
        "thalach": 120.0,
        "chol": 250.0,
        "trestbps": 150.0,
        "fbs": 1.0,
        "exang": 0.0,
        "cp": 4.0,
        "restecg": 0.0,
        "slope": 2.0,
        "thal": 7.0,
    }
    return pd.DataFrame([base | overrides])


def test_clinical_features_values():
    out = ClinicalFeatureAdder().fit_transform(_one_patient())
    assert out.loc[0, "hr_achievement"] == pytest.approx(120 / 160)
    # fbs=1, chol>240, trestbps>=140 -> 3 flags (exang=0)
    assert out.loc[0, "risk_flag_count"] == 3


def test_clinical_features_tolerate_missing_values():
    out = ClinicalFeatureAdder().transform(_one_patient(thalach=np.nan, fbs=np.nan))
    assert np.isnan(out.loc[0, "hr_achievement"])
    assert out.loc[0, "risk_flag_count"] == 2


def test_clinical_features_names():
    names = ClinicalFeatureAdder().get_feature_names_out()
    assert list(names) == config.FEATURES + DERIVED_FEATURES


def test_preprocessor_outputs_dense_without_nans(Xy):
    X, _ = Xy
    X = X.copy()
    X.iloc[:5, X.columns.get_loc("ca")] = np.nan
    X.iloc[:5, X.columns.get_loc("thal")] = np.nan
    Xt = build_preprocessor().fit_transform(ClinicalFeatureAdder().transform(X))
    assert not np.isnan(Xt).any()
    # 8 numeric (6 + 2 derived) + 3 binary + one-hot(cp 4 + restecg 3 + slope 3 + thal 3)
    assert Xt.shape == (len(X), 8 + 3 + 13)


def test_numeric_features_are_standardised(Xy):
    X, _ = Xy
    Xt = build_preprocessor().fit_transform(ClinicalFeatureAdder().transform(X))
    numeric = Xt[:, :8]
    assert np.allclose(numeric.mean(axis=0), 0, atol=1e-7)
    assert np.allclose(numeric.std(axis=0), 1, atol=1e-6)


def test_pipeline_handles_unseen_category(Xy):
    X, y = Xy
    pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    unseen = X.head(1).copy()
    unseen["thal"] = 5.0  # not seen in training
    proba = pipe.predict_proba(unseen)
    assert proba.shape == (1, 2)


def test_transformed_feature_names(Xy):
    X, y = Xy
    pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    names = transformed_feature_names(pipe)
    assert "hr_achievement" in names and "cp_4.0" in names
    assert len(names) == 24

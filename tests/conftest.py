"""Shared pytest fixtures. Tests are self-contained: they use a synthetic dataset so they
run in CI without network access or previously trained artifacts."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from heart_disease import config
from heart_disease.data import clean_data, split_features_target
from heart_disease.features import build_pipeline


def make_raw_frame(n: int = 160, seed: int = 7) -> pd.DataFrame:
    """Synthetic raw frame in UCI layout with a learnable signal."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "age": rng.integers(29, 78, n),
            "sex": rng.integers(0, 2, n),
            "cp": rng.integers(1, 5, n),
            "trestbps": rng.integers(94, 200, n),
            "chol": rng.integers(126, 564, n),
            "fbs": rng.integers(0, 2, n),
            "restecg": rng.integers(0, 3, n),
            "thalach": rng.integers(71, 202, n),
            "exang": rng.integers(0, 2, n),
            "oldpeak": rng.uniform(0, 6.2, n).round(1),
            "slope": rng.integers(1, 4, n),
            "ca": rng.integers(0, 4, n),
            "thal": rng.choice([3, 6, 7], n),
        }
    )
    score = (
        1.2 * (df["cp"] == 4)
        + 1.0 * df["exang"]
        + 0.5 * df["ca"]
        + 0.4 * df["oldpeak"]
        - 0.02 * (df["thalach"] - 150)
        + rng.normal(0, 0.6, n)
    )
    df["num"] = np.where(score > score.median(), rng.integers(1, 5, n), 0)
    return df


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    return make_raw_frame()


@pytest.fixture
def clean_frame(raw_frame) -> pd.DataFrame:
    return clean_data(raw_frame)


@pytest.fixture
def Xy(clean_frame):  # noqa: N802 - sklearn-style (X, y) naming
    return split_features_target(clean_frame)


@pytest.fixture(scope="session")
def trained_model_dir(tmp_path_factory):
    """A small fitted pipeline + metadata saved like the real training step does."""
    out = tmp_path_factory.mktemp("model")
    X, y = split_features_target(clean_data(make_raw_frame(240, seed=11)))
    pipe = build_pipeline(LogisticRegression(max_iter=2000)).fit(X, y)
    joblib.dump(pipe, out / "model.joblib")
    (out / "metadata.json").write_text(
        json.dumps(
            {
                "model_name": "test_logreg",
                "model_version": "test",
                "decision_threshold": 0.5,
                "features": config.FEATURES,
            }
        )
    )
    return out


@pytest.fixture
def sample_patient() -> dict:
    return {
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

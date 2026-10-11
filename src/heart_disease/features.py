"""Feature engineering and the reusable preprocessing pipeline.

The full sklearn ``Pipeline`` (feature engineering -> imputation -> scaling/encoding ->
classifier) is persisted as one artifact, so training and serving share identical logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heart_disease import config

DERIVED_FEATURES = ["hr_achievement", "risk_flag_count"]


class ClinicalFeatureAdder(BaseEstimator, TransformerMixin):
    """Adds two domain-motivated features.

    * ``hr_achievement``: max heart rate achieved / age-predicted max (220 - age).
      Low values (chronotropic incompetence) are associated with coronary disease.
    * ``risk_flag_count``: number of classic risk flags present - high fasting blood
      sugar, exercise-induced angina, cholesterol > 240 mg/dl, resting BP >= 140 mm Hg.
    """

    def fit(self, X: pd.DataFrame, y=None):  # noqa: ARG002 - stateless transformer
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X, columns=config.FEATURES).astype(float)
        predicted_max_hr = (220.0 - X["age"]).where(lambda s: s > 0)
        X["hr_achievement"] = X["thalach"] / predicted_max_hr
        X["risk_flag_count"] = (
            X["fbs"].fillna(0).eq(1).astype(int)
            + X["exang"].fillna(0).eq(1).astype(int)
            + X["chol"].gt(240).astype(int)
            + X["trestbps"].ge(140).astype(int)
        ).astype(float)
        return X.replace([np.inf, -np.inf], np.nan)

    def get_feature_names_out(self, input_features=None):  # noqa: ARG002
        return np.array(config.FEATURES + DERIVED_FEATURES, dtype=object)


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    binary = Pipeline([("impute", SimpleImputer(strategy="most_frequent"))])
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, config.NUMERIC_FEATURES + DERIVED_FEATURES),
            ("bin", binary, config.BINARY_FEATURES),
            ("cat", categorical, config.CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(estimator: BaseEstimator) -> Pipeline:
    return Pipeline(
        [
            ("features", ClinicalFeatureAdder()),
            ("preprocess", build_preprocessor()),
            ("clf", estimator),
        ]
    )


def transformed_feature_names(pipeline: Pipeline) -> list[str]:
    return list(pipeline.named_steps["preprocess"].get_feature_names_out())

"""Model training, hyper-parameter tuning and MLflow experiment tracking.

    python -m heart_disease.train            # full grid search (all models)
    python -m heart_disease.train --quick    # tiny grids, for smoke tests

For every model family a *parent* MLflow run is created holding the best configuration
(params, CV + test metrics, plots, model artifact) and one *nested* run per grid
candidate (params + CV metrics). A final "model-selection" run compares all families.
The champion (highest mean CV ROC-AUC - never selected on the test set) is saved to
models/model.joblib and registered in the MLflow model registry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import joblib
import matplotlib
import mlflow
import numpy as np
import pandas as pd
import sklearn
from mlflow.models import infer_signature
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, make_scorer, precision_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from heart_disease import config
from heart_disease.data import load_clean, split_features_target
from heart_disease.evaluate import (
    compute_metrics,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_model_comparison,
    plot_roc_overlay,
    plot_roc_pr,
)
from heart_disease.features import DERIVED_FEATURES, build_pipeline

log = logging.getLogger(__name__)

SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]
# Heavily regularised candidates may predict a single class; score precision as 0 then.
SCORERS = {m: m for m in SCORING} | {"precision": make_scorer(precision_score, zero_division=0)}


def model_specs(quick: bool = False) -> dict[str, tuple[BaseEstimator, dict]]:
    """Candidate estimators and their hyper-parameter grids."""
    rs = config.RANDOM_STATE
    if quick:
        return {
            "logistic_regression": (
                LogisticRegression(solver="saga", max_iter=5000, random_state=rs),
                {"clf__C": [0.1, 1.0]},
            ),
            "random_forest": (
                RandomForestClassifier(n_estimators=50, random_state=rs),
                {"clf__max_depth": [4, None]},
            ),
        }
    return {
        "logistic_regression": (
            LogisticRegression(solver="saga", max_iter=5000, random_state=rs),
            {
                "clf__C": [0.01, 0.1, 1.0, 10.0],
                "clf__l1_ratio": [0.0, 0.5, 1.0],  # 0 = L2 (ridge), 1 = L1 (lasso)
                "clf__class_weight": [None, "balanced"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(random_state=rs),
            {
                "clf__n_estimators": [200, 500],
                "clf__max_depth": [None, 4, 8],
                "clf__min_samples_leaf": [1, 4],
                "clf__max_features": ["sqrt", 0.5],
            },
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=rs),
            {
                "clf__n_estimators": [100, 300],
                "clf__learning_rate": [0.03, 0.1],
                "clf__max_depth": [2, 3],
                "clf__subsample": [0.8, 1.0],
            },
        ),
    }


@dataclass
class ModelResult:
    name: str
    search: GridSearchCV
    run_id: str
    model_uri: str
    cv_metrics: dict[str, float]
    test_metrics: dict[str, float]
    test_proba: np.ndarray = field(repr=False)


def _strip(params: dict) -> dict:
    return {k.removeprefix("clf__"): v for k, v in params.items()}


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _setup_mlflow(tracking_uri: str, experiment: str, artifact_root: str | None) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=artifact_root)
    mlflow.set_experiment(experiment)


def _tune_and_log(
    name: str,
    estimator: BaseEstimator,
    grid: dict,
    data: dict[str, pd.DataFrame | pd.Series],
    cv: StratifiedKFold,
    data_path: Path,
) -> ModelResult:
    X_train, X_test, y_train, y_test = (data[k] for k in ("X_train", "X_test", "y_train", "y_test"))
    search = GridSearchCV(
        build_pipeline(estimator),
        grid,
        scoring=SCORERS,
        refit="roc_auc",
        cv=cv,
        n_jobs=-1,
        return_train_score=False,
    )

    with mlflow.start_run(run_name=name) as parent:
        mlflow.set_tags(
            {
                "model_family": name,
                "run_type": "tuning",
                "git_sha": os.getenv("GITHUB_SHA", "local"),
                "estimator": type(estimator).__name__,
            }
        )
        mlflow.log_input(
            mlflow.data.from_pandas(
                pd.concat([X_train, y_train], axis=1).astype(float),
                source=data_path.resolve().as_uri(),
                name="heart_disease_clean-train",
                targets=config.TARGET,
            ),
            context="training",
        )
        log.info("[%s] grid search over %d candidates", name, _n(grid))
        search.fit(X_train, y_train)
        cv_res = pd.DataFrame(search.cv_results_)

        # --- one nested run per hyper-parameter candidate ---
        for i, row in cv_res.iterrows():
            with mlflow.start_run(run_name=f"{name}-cand-{i:02d}", nested=True):
                mlflow.set_tags({"model_family": name, "run_type": "candidate"})
                mlflow.log_params(_strip(row["params"]))
                mlflow.log_metrics(
                    {f"cv_{m}_mean": row[f"mean_test_{m}"] for m in SCORING}
                    | {f"cv_{m}_std": row[f"std_test_{m}"] for m in SCORING}
                    | {"mean_fit_time_s": row["mean_fit_time"]}
                )
                mlflow.set_tag("rank_by_roc_auc", int(row["rank_test_roc_auc"]))

        # --- best candidate: params, CV metrics, test metrics, plots, model ---
        best = search.best_estimator_
        bi = search.best_index_
        cv_metrics = {f"cv_{m}_mean": float(cv_res.loc[bi, f"mean_test_{m}"]) for m in SCORING}
        cv_metrics |= {f"cv_{m}_std": float(cv_res.loc[bi, f"std_test_{m}"]) for m in SCORING}
        test_proba = best.predict_proba(X_test)[:, 1]
        test_metrics = {f"test_{k}": v for k, v in compute_metrics(y_test, test_proba).items()}

        mlflow.log_params(
            _strip(search.best_params_)
            | {
                "cv_folds": cv.get_n_splits(),
                "test_size": config.TEST_SIZE,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "derived_features": ",".join(DERIVED_FEATURES),
                "numeric_imputer": "median",
                "categorical_encoding": "onehot(most_frequent impute)",
                "n_candidates": len(cv_res),
            }
        )
        mlflow.log_metrics(cv_metrics | test_metrics)

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            y_pred = (test_proba >= 0.5).astype(int)
            plot_confusion_matrix(y_test, y_pred, name, tmp / "confusion_matrix.png")
            plot_roc_pr(y_test, test_proba, name, tmp / "roc_pr_curves.png")
            plot_feature_importance(best, name, tmp / "feature_importance.png")
            cv_res.drop(columns="params").to_csv(tmp / "cv_results.csv", index=False)
            (tmp / "classification_report.txt").write_text(
                classification_report(y_test, y_pred, target_names=["no_disease", "disease"])
            )
            mlflow.log_artifacts(str(tmp), artifact_path="evaluation")

        example = X_train.dropna().head(3)
        model_info = mlflow.sklearn.log_model(
            best,
            name="model",
            serialization_format="cloudpickle",
            signature=infer_signature(example, best.predict_proba(example)),
            input_example=example,
            code_paths=[str(Path(__file__).resolve().parent)],
            pip_requirements=str(config.PROJECT_ROOT / "requirements-serve.txt"),
        )
        log.info(
            "[%s] best=%s cv_roc_auc=%.4f test_roc_auc=%.4f",
            name,
            _strip(search.best_params_),
            cv_metrics["cv_roc_auc_mean"],
            test_metrics["test_roc_auc"],
        )
        return ModelResult(
            name,
            search,
            parent.info.run_id,
            model_info.model_uri,
            cv_metrics,
            test_metrics,
            test_proba,
        )


def _n(grid: dict) -> int:
    return int(np.prod([len(v) for v in grid.values()]))


def run_training(
    data_path: Path = config.CLEAN_DATA_FILE,
    model_dir: Path = config.MODELS_DIR,
    figures_dir: Path = config.FIGURES_DIR,
    tracking_uri: str = config.MLFLOW_TRACKING_URI,
    experiment: str = config.MLFLOW_EXPERIMENT,
    artifact_root: str | None = config.MLFLOW_ARTIFACT_ROOT,
    quick: bool = False,
    register: bool = True,
) -> dict:
    matplotlib.use("Agg")
    data_path, model_dir, figures_dir = Path(data_path), Path(model_dir), Path(figures_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    _setup_mlflow(tracking_uri, experiment, artifact_root)

    df = load_clean(data_path)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )
    data = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    log.info(
        "Train=%d Test=%d (positives in train: %.1f%%)",
        len(X_train),
        len(X_test),
        100 * y_train.mean(),
    )

    results = [
        _tune_and_log(name, est, grid, data, cv, data_path)
        for name, (est, grid) in model_specs(quick).items()
    ]

    summary = pd.DataFrame(
        {r.name: r.cv_metrics | r.test_metrics | {"run_id": r.run_id} for r in results}
    ).T.sort_values("cv_roc_auc_mean", ascending=False)
    champion = next(r for r in results if r.name == summary.index[0])
    log.info("Model comparison:\n%s", summary.drop(columns="run_id").astype(float).round(4))

    # --- model-selection summary run ---
    with mlflow.start_run(run_name="model-selection"):
        mlflow.set_tags({"run_type": "selection", "champion": champion.name})
        mlflow.log_param("selection_metric", "cv_roc_auc_mean")
        mlflow.log_param("champion", champion.name)
        mlflow.log_metrics({f"champion_{k}": v for k, v in champion.test_metrics.items()})
        comparison_png = plot_model_comparison(summary, figures_dir / "model_comparison.png")
        roc_png = plot_roc_overlay(
            {r.name: r.test_proba for r in results}, y_test, figures_dir / "model_roc_overlay.png"
        )
        y_pred = (champion.test_proba >= 0.5).astype(int)
        cm_png = plot_confusion_matrix(
            y_test, y_pred, f"Champion: {champion.name}", figures_dir / "model_confusion_matrix.png"
        )
        fi_png = plot_feature_importance(
            champion.search.best_estimator_,
            champion.name,
            figures_dir / "model_feature_importance.png",
        )
        summary_csv = model_dir / "model_comparison.csv"
        summary.to_csv(summary_csv)
        for p in (comparison_png, roc_png, cm_png, fi_png, summary_csv):
            if p:
                mlflow.log_artifact(str(p), artifact_path="selection")

    # --- persist champion for serving ---
    best_pipeline = champion.search.best_estimator_
    joblib.dump(best_pipeline, model_dir / "model.joblib")
    metadata = {
        "model_name": champion.name,
        "model_version": datetime.now(UTC).strftime("%Y%m%d%H%M%S"),
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "best_params": {k: str(v) for k, v in _strip(champion.search.best_params_).items()},
        "cv_metrics": champion.cv_metrics,
        "test_metrics": champion.test_metrics,
        "features": config.FEATURES,
        "derived_features": DERIVED_FEATURES,
        "decision_threshold": 0.5,
        "mlflow_run_id": champion.run_id,
        "mlflow_model_uri": champion.model_uri,
        "data_sha256": _file_sha256(data_path),
        "n_train": len(X_train),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
    }
    if register:
        mv = mlflow.register_model(champion.model_uri, config.REGISTERED_MODEL_NAME)
        mlflow.MlflowClient().set_registered_model_alias(
            config.REGISTERED_MODEL_NAME, "champion", mv.version
        )
        metadata["registry_version"] = str(mv.version)
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    log.info("Champion '%s' saved to %s", champion.name, model_dir / "model.joblib")
    return metadata


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="Train heart disease classifiers")
    parser.add_argument("--quick", action="store_true", help="small grids for smoke testing")
    parser.add_argument("--no-register", action="store_true", help="skip MLflow model registry")
    args = parser.parse_args()
    meta = run_training(quick=args.quick, register=not args.no_register)
    print(json.dumps({k: meta[k] for k in ("model_name", "cv_metrics", "test_metrics")}, indent=2))


if __name__ == "__main__":
    main()

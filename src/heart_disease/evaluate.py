"""Evaluation metrics and diagnostic plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from heart_disease.features import transformed_feature_names


def compute_metrics(y_true, y_proba, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "avg_precision": average_precision_score(y_true, y_proba),
    }


def plot_confusion_matrix(y_true, y_pred, title: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["No disease", "Disease"], cmap="Blues", ax=ax
    )
    ax.set_title(title)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_roc_pr(y_true, y_proba, title: str, path: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax1, plot_chance_level=True)
    ax1.set_title(f"ROC - {title}")
    PrecisionRecallDisplay.from_predictions(y_true, y_proba, ax=ax2, plot_chance_level=True)
    ax2.set_title(f"Precision-Recall - {title}")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def feature_importance(pipeline: Pipeline) -> pd.Series | None:
    """Coefficient magnitude (linear) or impurity importance (trees), by feature name."""
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        values = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        values = np.abs(clf.coef_).ravel()
    else:
        return None
    return pd.Series(values, index=transformed_feature_names(pipeline)).sort_values()


def plot_feature_importance(pipeline: Pipeline, title: str, path: Path) -> Path | None:
    imp = feature_importance(pipeline)
    if imp is None:
        return None
    fig, ax = plt.subplots(figsize=(7, 6))
    imp.tail(15).plot.barh(ax=ax, color="#457b9d")
    ax.set_title(f"Top features - {title}")
    ax.set_xlabel(
        "|coefficient|" if hasattr(pipeline.named_steps["clf"], "coef_") else "importance"
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_model_comparison(summary: pd.DataFrame, path: Path) -> Path:
    """Bar chart of CV mean +/- std for each model and metric."""
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    x = np.arange(len(metrics))
    width = 0.8 / len(summary)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, (name, row) in enumerate(summary.iterrows()):
        ax.bar(
            x + i * width,
            [row[f"cv_{m}_mean"] for m in metrics],
            width,
            yerr=[row[f"cv_{m}_std"] for m in metrics],
            capsize=3,
            label=name,
        )
    ax.set_xticks(x + width * (len(summary) - 1) / 2, metrics)
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("5-fold CV score")
    ax.set_title("Model comparison (cross-validation on training split)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=len(summary), frameon=False)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_roc_overlay(curves: dict[str, np.ndarray], y_true, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, y_proba in curves.items():
        RocCurveDisplay.from_predictions(y_true, y_proba, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_title("ROC curves on held-out test set")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

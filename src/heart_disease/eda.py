"""Exploratory data analysis plots (used by notebooks/01_eda.ipynb and the report).

python -m heart_disease.eda      # writes PNGs to reports/figures/
"""

from __future__ import annotations

import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from heart_disease import config
from heart_disease.data import load_clean

log = logging.getLogger(__name__)

PALETTE = {0: "#2a9d8f", 1: "#e76f51"}
TARGET_LABELS = {0: "No disease", 1: "Disease"}
CATEGORY_LABELS = {
    "sex": {0: "Female", 1: "Male"},
    "cp": {1: "Typical", 2: "Atypical", 3: "Non-anginal", 4: "Asymptomatic"},
    "fbs": {0: "<=120", 1: ">120"},
    "restecg": {0: "Normal", 1: "ST-T abn.", 2: "LV hypertr."},
    "exang": {0: "No", 1: "Yes"},
    "slope": {1: "Up", 2: "Flat", 3: "Down"},
    "thal": {3: "Normal", 6: "Fixed", 7: "Reversible"},
}
NICE_NAMES = {
    "age": "Age (years)",
    "trestbps": "Resting BP (mm Hg)",
    "chol": "Cholesterol (mg/dl)",
    "thalach": "Max heart rate",
    "oldpeak": "ST depression (oldpeak)",
    "ca": "Major vessels (ca)",
}

sns.set_theme(style="whitegrid", context="notebook")


def _save(fig: plt.Figure, name: str) -> plt.Figure:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / name, dpi=150, bbox_inches="tight")
    return fig


def plot_class_balance(df: pd.DataFrame) -> plt.Figure:
    counts = df[config.TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(
        [TARGET_LABELS[i] for i in counts.index],
        counts.values,
        color=[PALETTE[i] for i in counts.index],
    )
    for bar, n in zip(bars, counts.values, strict=True):
        ax.annotate(
            f"{n}\n({n / counts.sum():.1%})",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
        )
    ax.set_ylim(0, counts.max() * 1.25)
    ax.set_ylabel("Patients")
    ax.set_title("Class balance of the target")
    return _save(fig, "eda_class_balance.png")


def plot_numeric_distributions(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.flat, config.NUMERIC_FEATURES, strict=True):
        sns.histplot(
            data=df,
            x=col,
            hue=config.TARGET,
            palette=PALETTE,
            kde=col != "ca",
            discrete=col == "ca",
            element="step",
            stat="density",
            common_norm=False,
            ax=ax,
        )
        ax.set_xlabel(NICE_NAMES[col])
        legend = ax.get_legend()
        if legend:
            legend.set_title("")
            for text in legend.get_texts():
                text.set_text(TARGET_LABELS[int(text.get_text())])
    fig.suptitle("Numeric feature distributions by diagnosis", fontsize=14, y=1.0)
    fig.tight_layout()
    return _save(fig, "eda_numeric_histograms.png")


def plot_categorical_disease_rate(df: pd.DataFrame) -> plt.Figure:
    cols = config.BINARY_FEATURES + config.CATEGORICAL_FEATURES
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, col in zip(axes.flat, cols, strict=False):
        stats = (
            df.dropna(subset=[col])
            .groupby(col)[config.TARGET]
            .agg(["mean", "size"])
            .rename(index=CATEGORY_LABELS[col])
        )
        bars = ax.bar(stats.index.astype(str), stats["mean"], color="#457b9d")
        for bar, n in zip(bars, stats["size"], strict=True):
            ax.annotate(
                f"n={n}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.axhline(df[config.TARGET].mean(), ls="--", c="grey", lw=1)
        ax.set_ylim(0, 1)
        ax.set_title(col)
        ax.set_ylabel("Disease rate")
        ax.tick_params(axis="x", rotation=20)
    axes.flat[-1].axis("off")
    fig.suptitle("Disease rate per category (dashed line = overall prevalence)", fontsize=14, y=1.0)
    fig.tight_layout()
    return _save(fig, "eda_categorical_disease_rate.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    corr = df[config.FEATURES + [config.TARGET]].astype(float).corr(method="spearman")
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        cbar_kws={"shrink": 0.7, "label": "Spearman rho"},
        ax=ax,
    )
    ax.set_title("Spearman correlation matrix (features + target)")
    return _save(fig, "eda_correlation_heatmap.png")


def plot_target_correlation(df: pd.DataFrame) -> plt.Figure:
    corr = (
        df[config.FEATURES + [config.TARGET]]
        .astype(float)
        .corr(method="spearman")[config.TARGET]
        .drop(config.TARGET)
        .sort_values()
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(corr.index, corr.values, color=[PALETTE[int(v > 0)] for v in corr.values])
    ax.axvline(0, c="black", lw=0.8)
    ax.set_xlabel("Spearman correlation with target")
    ax.set_title("Univariate association with heart disease")
    return _save(fig, "eda_target_correlation.png")


def plot_age_hr_scatter(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, grp in df.groupby(config.TARGET):
        ax.scatter(
            grp["age"],
            grp["thalach"],
            s=28,
            alpha=0.7,
            c=PALETTE[label],
            label=TARGET_LABELS[label],
        )
    ages = np.linspace(df["age"].min(), df["age"].max(), 50)
    ax.plot(ages, 220 - ages, "k--", lw=1, label="Predicted max HR (220 - age)")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Max heart rate achieved")
    ax.set_title("Heart-rate achievement vs age (motivates `hr_achievement`)")
    ax.legend()
    return _save(fig, "eda_age_vs_max_hr.png")


def run_all(df: pd.DataFrame | None = None) -> None:
    df = load_clean() if df is None else df
    for fn in (
        plot_class_balance,
        plot_numeric_distributions,
        plot_categorical_disease_rate,
        plot_correlation_heatmap,
        plot_target_correlation,
        plot_age_hr_scatter,
    ):
        fn(df)
        plt.close("all")
        log.info("Created %s", fn.__name__)


if __name__ == "__main__":
    matplotlib.use("Agg")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_all()

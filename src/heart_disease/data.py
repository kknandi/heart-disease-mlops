"""Data loading, validation and cleaning.

Run as a module to turn the raw UCI file into the cleaned CSV:
    python -m heart_disease.data
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from heart_disease import config

log = logging.getLogger(__name__)

DISCRETE_COLUMNS = ["ca", *config.BINARY_FEATURES, *config.CATEGORICAL_FEATURES]


def load_raw(path: Path | str = config.RAW_DATA_FILE) -> pd.DataFrame:
    """Read the header-less UCI file; '?' marks missing values."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `python scripts/download_data.py` first")
    return pd.read_csv(path, header=None, names=config.RAW_COLUMNS, na_values="?")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean a raw frame.

    * coerce every column to numeric (non-numeric -> NaN)
    * values outside plausible ranges / unknown category codes -> NaN
    * binarise the target: num > 0 -> 1 (disease present)
    * drop rows without a target and exact duplicates

    Remaining NaNs are intentionally *kept*: they are imputed inside the sklearn
    pipeline (fit on training folds only) so the same logic is applied at inference.
    """
    missing_cols = set(config.RAW_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {sorted(missing_cols)}")

    out = df[config.RAW_COLUMNS].apply(pd.to_numeric, errors="coerce").astype(float)

    for col, (low, high) in config.VALID_RANGES.items():
        out.loc[~out[col].between(low, high), col] = np.nan
    for col, allowed in config.ALLOWED_CATEGORIES.items():
        out.loc[~out[col].isin(allowed), col] = np.nan

    out = out.dropna(subset=["num"])
    out[config.TARGET] = (out["num"] > 0).astype(int)
    out = out.drop(columns="num").drop_duplicates().reset_index(drop=True)

    # Nullable ints keep the CSV human readable (1 instead of 1.0, blanks for missing).
    out[DISCRETE_COLUMNS] = out[DISCRETE_COLUMNS].astype("Int64")
    return out


def load_clean(path: Path | str = config.CLEAN_DATA_FILE) -> pd.DataFrame:
    return pd.read_csv(path)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[config.FEATURES].astype(float)
    y = df[config.TARGET].astype(int)
    return X, y


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.isna().sum()
    return (
        pd.DataFrame({"missing": counts, "pct": (counts / len(df) * 100).round(2)})
        .query("missing > 0")
        .sort_values("missing", ascending=False)
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raw = load_raw()
    log.info("Raw data: %d rows x %d cols", *raw.shape)
    clean = clean_data(raw)
    log.info("Clean data: %d rows x %d cols", *clean.shape)
    log.info("Missing values (imputed later in pipeline):\n%s", missing_value_report(clean))
    log.info("Class balance:\n%s", clean[config.TARGET].value_counts().to_string())
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    clean.to_csv(config.CLEAN_DATA_FILE, index=False)
    log.info("Saved %s", config.CLEAN_DATA_FILE)


if __name__ == "__main__":
    main()

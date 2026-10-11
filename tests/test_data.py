import numpy as np
import pandas as pd
import pytest

from heart_disease import config
from heart_disease.data import (
    clean_data,
    load_clean,
    load_raw,
    missing_value_report,
    split_features_target,
)


def test_load_raw_parses_question_marks_as_nan(tmp_path):
    raw = tmp_path / "raw.data"
    raw.write_text(
        "63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,0.0,6.0,0\n"
        "67.0,1.0,4.0,160.0,286.0,0.0,2.0,108.0,1.0,1.5,2.0,?,?,2\n"
    )
    df = load_raw(raw)
    assert list(df.columns) == config.RAW_COLUMNS
    assert df.shape == (2, 14)
    assert df.loc[1, ["ca", "thal"]].isna().all()


def test_load_raw_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw(tmp_path / "nope.data")


def test_clean_binarises_target(raw_frame):
    clean = clean_data(raw_frame)
    assert config.TARGET in clean and "num" not in clean
    assert set(clean[config.TARGET].unique()) <= {0, 1}
    assert (clean[config.TARGET] == 1).sum() == (raw_frame["num"] > 0).sum()


def test_clean_invalidates_out_of_range_and_unknown_codes(raw_frame):
    raw = raw_frame.astype(object)
    raw.loc[0, "chol"] = 0  # impossible cholesterol (common in other UCI sites)
    raw.loc[1, "cp"] = 9  # unknown chest-pain code
    raw.loc[2, "trestbps"] = "abc"  # non-numeric garbage
    clean = clean_data(raw)
    assert pd.isna(clean.loc[0, "chol"])
    assert pd.isna(clean.loc[1, "cp"])
    assert pd.isna(clean.loc[2, "trestbps"])


def test_clean_drops_duplicates_and_missing_target(raw_frame):
    raw = pd.concat([raw_frame, raw_frame.head(5)], ignore_index=True)
    raw.loc[len(raw)] = raw_frame.iloc[10].to_dict() | {"num": np.nan, "age": 33}
    clean = clean_data(raw)
    assert len(clean) == len(raw_frame)


def test_clean_rejects_missing_columns(raw_frame):
    with pytest.raises(ValueError, match="Missing columns"):
        clean_data(raw_frame.drop(columns=["thal"]))


def test_split_features_target(clean_frame):
    X, y = split_features_target(clean_frame)
    assert list(X.columns) == config.FEATURES
    assert all(pd.api.types.is_float_dtype(dt) for dt in X.dtypes)
    assert len(X) == len(y)


def test_missing_value_report(clean_frame):
    df = clean_frame.copy()
    df.loc[:4, "ca"] = pd.NA
    report = missing_value_report(df)
    assert report.loc["ca", "missing"] == 5


@pytest.mark.skipif(not config.CLEAN_DATA_FILE.exists(), reason="cleaned dataset not generated")
def test_real_clean_dataset_contract():
    df = load_clean()
    assert len(df) == 303
    assert set(df.columns) == set(config.FEATURES + [config.TARGET])
    assert df[config.TARGET].isin([0, 1]).all()
    assert df[config.FEATURES].isna().sum().sum() <= 10  # only a handful of ca/thal gaps

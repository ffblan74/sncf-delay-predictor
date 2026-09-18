import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_pipeline import clean, engineer_features, build_feature_matrix
from generate_data import generate


@pytest.fixture(scope="module")
def raw_df():
    return generate(n_rows=500, seed=1)


def test_generate_has_expected_columns(raw_df):
    expected = {
        "date", "route", "departure_station", "arrival_station",
        "distance_km", "scheduled_hour", "delay_minutes", "cancelled",
    }
    assert expected.issubset(set(raw_df.columns))


def test_clean_drops_cancelled_trains(raw_df):
    cleaned = clean(raw_df)
    assert (cleaned["cancelled"] == 0).all()


def test_clean_removes_negative_or_extreme_delays(raw_df):
    dirty = raw_df.copy()
    dirty.loc[0, "delay_minutes"] = -5
    dirty.loc[1, "delay_minutes"] = 9999
    cleaned = clean(dirty)
    assert (cleaned["delay_minutes"] >= 0).all()
    assert (cleaned["delay_minutes"] < 300).all()


def test_engineer_features_adds_rush_hour_flag(raw_df):
    df = engineer_features(clean(raw_df))
    assert "is_rush_hour" in df.columns
    assert set(df["is_rush_hour"].unique()).issubset({0, 1})


def test_feature_matrix_shapes_match(raw_df):
    df = engineer_features(clean(raw_df))
    X, y, encoder = build_feature_matrix(df, fit_encoder=True)
    assert len(X) == len(y) == len(df)
    assert X.isna().sum().sum() == 0


def test_feature_matrix_inference_reuses_encoder(raw_df):
    df = engineer_features(clean(raw_df))
    X_train, y_train, encoder = build_feature_matrix(df, fit_encoder=True)
    X_infer, _, _ = build_feature_matrix(df.head(5), fit_encoder=False, encoder=encoder)
    assert list(X_infer.columns) == list(X_train.columns)

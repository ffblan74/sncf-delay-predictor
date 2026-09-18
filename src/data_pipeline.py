"""
ETL / feature engineering pipeline.

Kept separate from train.py so it can be unit-tested independently and
reused by the API at inference time (same transformations, no train/serve
skew).
"""

from __future__ import annotations

import pandas as pd

CATEGORICAL_COLS = ["season", "departure_station", "arrival_station"]
NUMERIC_FEATURES = [
    "distance_km", "scheduled_hour", "is_weekend",
    "weather_severity", "planned_trains", "is_rush_hour",
]
TARGET = "delay_minutes"


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop obviously invalid rows and cancelled trains (no meaningful delay)."""
    df = df.dropna(subset=["distance_km", "scheduled_hour", "delay_minutes"])
    df = df[df["cancelled"] == 0].copy()
    df = df[(df["delay_minutes"] >= 0) & (df["delay_minutes"] < 300)]
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_rush_hour"] = df["scheduled_hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
    df["month"] = pd.to_datetime(df["date"]).dt.month
    return df


def build_feature_matrix(df: pd.DataFrame, fit_encoder: bool = True, encoder=None):
    """
    One-hot encodes categorical columns and returns (X, y, encoder).
    Pass a fitted `encoder` at inference time to guarantee identical columns.
    """
    from sklearn.preprocessing import OneHotEncoder

    if fit_encoder:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        cat_matrix = encoder.fit_transform(df[CATEGORICAL_COLS])
    else:
        assert encoder is not None, "Must pass a fitted encoder for inference"
        cat_matrix = encoder.transform(df[CATEGORICAL_COLS])

    cat_df = pd.DataFrame(
        cat_matrix,
        columns=encoder.get_feature_names_out(CATEGORICAL_COLS),
        index=df.index,
    )

    X = pd.concat([df[NUMERIC_FEATURES].reset_index(drop=True),
                    cat_df.reset_index(drop=True)], axis=1)
    y = df[TARGET].reset_index(drop=True) if TARGET in df.columns else None
    return X, y, encoder


def run_pipeline(raw_csv_path: str):
    df = load_raw(raw_csv_path)
    df = clean(df)
    df = engineer_features(df)
    X, y, encoder = build_feature_matrix(df, fit_encoder=True)
    return X, y, encoder, df

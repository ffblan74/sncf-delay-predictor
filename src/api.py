"""
Minimal FastAPI service exposing the trained delay-prediction model.

Run locally:
    uvicorn src.api:app --reload
Then open http://127.0.0.1:8000/docs for interactive Swagger UI.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from data_pipeline import NUMERIC_FEATURES, CATEGORICAL_COLS, engineer_features, build_feature_matrix

app = FastAPI(
    title="SNCF Delay Predictor",
    description="Predicts expected train delay (minutes) from route and conditions.",
    version="0.1.0",
)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
model = joblib.load(MODEL_DIR / "model.joblib")
encoder = joblib.load(MODEL_DIR / "encoder.joblib")
with open(MODEL_DIR / "feature_columns.json") as f:
    FEATURE_COLUMNS = json.load(f)


class PredictionRequest(BaseModel):
    departure_station: str = Field(examples=["Paris"])
    arrival_station: str = Field(examples=["Lyon"])
    distance_km: float = Field(examples=[465])
    scheduled_hour: int = Field(ge=0, le=23, examples=[8])
    is_weekend: bool = False
    season: str = Field(examples=["winter"])
    weather_severity: float = Field(ge=0, le=1, examples=[0.3])
    planned_trains: int = Field(examples=[20])
    date: str = Field(examples=["2026-01-15"])


class PredictionResponse(BaseModel):
    predicted_delay_minutes: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    row = pd.DataFrame([{
        "date": req.date,
        "departure_station": req.departure_station,
        "arrival_station": req.arrival_station,
        "distance_km": req.distance_km,
        "scheduled_hour": req.scheduled_hour,
        "is_weekend": int(req.is_weekend),
        "season": req.season,
        "weather_severity": req.weather_severity,
        "planned_trains": req.planned_trains,
    }])
    row = engineer_features(row)
    X, _, _ = build_feature_matrix(row, fit_encoder=False, encoder=encoder)
    X = X.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    pred = float(model.predict(X)[0])
    return PredictionResponse(predicted_delay_minutes=round(pred, 1))

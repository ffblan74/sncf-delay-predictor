"""
FastAPI service exposing the monthly delay forecaster.

A caller cannot reasonably supply 35 lagged features, so the service ships
with the cleaned history panel written by `train.py` and builds the feature
row itself, through the exact same pipeline functions used for training --
that is what rules out train/serve skew. The client only provides what it
actually knows: an OD pair, a month, and optionally next month's timetable.

Run locally:
    cd src && uvicorn api:app --reload
Then open http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from data_pipeline import (
    KNOWN_IN_ADVANCE,
    OD,
    OD_KEY,
    PERIOD,
    TARGET,
    engineer_features,
    load_raw,
    predict_delays,
    to_panel,
)

# Overridable so a deployment (or a test) can point at another artifact set.
MODEL_DIR = Path(
    os.environ.get("SNCF_MODEL_DIR", Path(__file__).resolve().parent.parent / "models")
)

app = FastAPI(
    title="SNCF Delay Predictor",
    description=(
        "Forecasts the average arrival delay (minutes, all trains included) "
        "of a TGV origin-destination pair for a given month, from SNCF's "
        "public monthly regularity data."
    ),
    version="1.0.0",
)


class Artifacts:
    """Model, feature contract and history panel, loaded once."""

    def __init__(self, model_dir: Path):
        self.model = joblib.load(model_dir / "model.joblib")
        with open(model_dir / "feature_columns.json", encoding="utf-8") as fh:
            self.contract = json.load(fh)
        try:
            self.metrics = json.loads(
                (model_dir / "metrics.json").read_text(encoding="utf-8")
            )
        except FileNotFoundError:
            self.metrics = {}
        self.history = load_raw(str(model_dir / "history.csv.gz"))
        self.last_month = self.history[PERIOD].max()


@lru_cache(maxsize=1)
def artifacts() -> Artifacts:
    try:
        return Artifacts(MODEL_DIR)
    except FileNotFoundError as exc:  # pragma: no cover - deployment mistake
        raise RuntimeError(
            f"missing artifact in {MODEL_DIR}: {exc}. Run `python src/train.py` first."
        ) from exc


class PredictionRequest(BaseModel):
    gare_depart: str = Field(examples=["PARIS MONTPARNASSE"])
    gare_arrivee: str = Field(examples=["BORDEAUX ST JEAN"])
    month: str | None = Field(
        default=None,
        description="Month to forecast, YYYY-MM. Defaults to the month after "
                    "the last one in the data. A month already present in the "
                    "history is answered as a backtest.",
        examples=["2026-07"],
    )
    nb_train_prevu: int | None = Field(
        default=None, ge=1,
        description="Planned trains that month (from the timetable). "
                    "Defaults to the OD's recent average.",
    )
    duree_moyenne: float | None = Field(
        default=None, gt=0,
        description="Scheduled journey time in minutes. Defaults to the "
                    "OD's recent average.",
    )


class PredictionResponse(BaseModel):
    gare_depart: str
    gare_arrivee: str
    month: str
    predicted_delay_minutes: float
    recent_average_minutes: float | None = Field(
        description="The OD's own 6-month average, i.e. the baseline the "
                    "model corrects."
    )
    is_backtest: bool = Field(
        description="True when the requested month is already in the data, so "
                    "the actual value is known and returned for comparison."
    )
    actual_delay_minutes: float | None = None
    history_up_to: str


class RouteInfo(BaseModel):
    gare_depart: str
    gare_arrivee: str
    months_observed: int
    last_month: str
    recent_average_minutes: float


@app.get("/health")
def health() -> dict:
    art = artifacts()
    return {
        "status": "ok",
        "model": art.metrics.get("model", "unknown"),
        "history_up_to": str(art.last_month),
        "n_routes": int(art.history[OD].nunique()),
    }


@app.get("/metrics")
def metrics() -> dict:
    """Evaluation report of the currently served model."""
    return artifacts().metrics


@app.get("/routes", response_model=list[RouteInfo])
def routes() -> list[RouteInfo]:
    """Origin-destination pairs the model can forecast."""
    history = artifacts().history.sort_values(PERIOD)
    out = []
    for (dep, arr), group in history.groupby(OD_KEY, sort=True):
        out.append(RouteInfo(
            gare_depart=dep,
            gare_arrivee=arr,
            months_observed=len(group),
            last_month=str(group[PERIOD].max()),
            recent_average_minutes=round(float(group[TARGET].tail(6).mean()), 2),
        ))
    return out


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest) -> PredictionResponse:
    art = artifacts()
    history = art.history
    route = history[
        (history["gare_depart"] == req.gare_depart.strip().upper())
        & (history["gare_arrivee"] == req.gare_arrivee.strip().upper())
    ]
    if route.empty:
        raise HTTPException(
            status_code=404,
            detail=f"unknown route {req.gare_depart!r} -> {req.gare_arrivee!r}; "
                   "see GET /routes for the available pairs",
        )

    target_period = _parse_month(req.month) if req.month else art.last_month + 1
    if target_period < route[PERIOD].min() + 1:
        raise HTTPException(
            status_code=422,
            detail=f"no history before {target_period}; this route starts at "
                   f"{route[PERIOD].min()}",
        )
    if target_period > art.last_month + 1:
        raise HTTPException(
            status_code=422,
            detail=f"the model forecasts one month ahead: with data up to "
                   f"{art.last_month}, the furthest month is "
                   f"{art.last_month + 1}",
        )

    # Same code path as training: dense panel -> causal features -> predict.
    # Rows after `target_period` are dropped so a backtest cannot see them.
    panel = to_panel(history[history[PERIOD] < target_period], until=target_period)
    row_mask = (panel[OD] == route[OD].iloc[0]) & (panel[PERIOD] == target_period)

    # The timetable is published before the month starts, so it is a feature,
    # not a leak: take it from the request, else from the record when the
    # month is already past, else from the OD's recent level.
    on_record = route[route[PERIOD] == target_period]
    overrides = {"nb_train_prevu": req.nb_train_prevu,
                 "duree_moyenne": req.duree_moyenne}
    for col in KNOWN_IN_ADVANCE:
        value = overrides[col]
        if value is None and not on_record.empty:
            value = on_record[col].iloc[0]
        if value is None:
            value = route[col].tail(6).mean()
        panel.loc[row_mask, col] = value

    features = engineer_features(panel)
    row = features[
        (features[OD] == route[OD].iloc[0]) & (features[PERIOD] == target_period)
    ]
    prediction = float(predict_delays(art.model, row)[0])

    actual = route.loc[route[PERIOD] == target_period, TARGET]
    recent = row["roll6"].iloc[0]
    return PredictionResponse(
        gare_depart=req.gare_depart.strip().upper(),
        gare_arrivee=req.gare_arrivee.strip().upper(),
        month=str(target_period),
        predicted_delay_minutes=round(prediction, 1),
        recent_average_minutes=None if pd.isna(recent) else round(float(recent), 1),
        is_backtest=not actual.empty,
        actual_delay_minutes=None if actual.empty else round(float(actual.iloc[0]), 1),
        history_up_to=str(art.last_month),
    )


def _parse_month(month: str) -> pd.Period:
    try:
        return pd.Period(month, freq="M")
    except Exception as exc:
        raise HTTPException(
            status_code=422, detail=f"invalid month {month!r}, expected YYYY-MM"
        ) from exc

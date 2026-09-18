"""
End-to-end tests: synthetic export -> training -> HTTP inference.

They also pin down the property the whole design is about: a prediction
served by the API is bit-for-bit the prediction the training pipeline would
have produced for the same row (no train/serve skew).
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import data_pipeline as dp
from generate_data import ROUTES, write

DEP, ARR, _ = ROUTES[0]


@pytest.fixture(scope="module")
def client(tmp_path_factory, monkeypatch_module):
    import train as train_module

    data_dir = tmp_path_factory.mktemp("data")
    model_dir = tmp_path_factory.mktemp("models")
    csv = data_dir / "raw.csv"
    write(str(csv), n_months=84, seed=11)
    train_module.train(str(csv), str(model_dir), model_kind="hgb",
                       test_months=6, cv_folds=1)

    monkeypatch_module.setenv("SNCF_MODEL_DIR", str(model_dir))
    import api

    api.MODEL_DIR = Path(model_dir)
    api.artifacts.cache_clear()
    yield TestClient(api.app)
    api.artifacts.cache_clear()


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch

    patch = MonkeyPatch()
    yield patch
    patch.undo()


def test_health_reports_the_served_artifacts(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model"] == "hgb"
    assert body["n_routes"] == len(ROUTES)


def test_metrics_exposes_the_evaluation_report(client):
    body = client.get("/metrics").json()
    assert body["target"] == dp.TARGET
    assert body["test"]["n_rows"] > 0
    assert "mean_12_months" in body["baselines"]


def test_routes_lists_every_od_pair(client):
    body = client.get("/routes").json()
    assert len(body) == len(ROUTES)
    assert {(r["gare_depart"], r["gare_arrivee"]) for r in body} == {
        (dep, arr) for dep, arr, _ in ROUTES
    }
    assert all(r["recent_average_minutes"] >= 0 for r in body)


def test_predict_defaults_to_the_month_after_the_data(client):
    body = client.post("/predict", json={"gare_depart": DEP, "gare_arrivee": ARR}).json()
    assert body["month"] > body["history_up_to"]
    assert body["predicted_delay_minutes"] >= 0
    assert body["is_backtest"] is False
    assert body["actual_delay_minutes"] is None
    # a monthly average that far off would mean the anchor is not being used
    assert abs(body["predicted_delay_minutes"] - body["recent_average_minutes"]) < 10


def test_predict_on_a_known_month_is_a_backtest(client):
    last = client.get("/health").json()["history_up_to"]
    body = client.post(
        "/predict",
        json={"gare_depart": DEP, "gare_arrivee": ARR, "month": last},
    ).json()
    assert body["month"] == last
    assert body["is_backtest"] is True
    assert body["actual_delay_minutes"] is not None


def test_predict_accepts_a_timetable_override(client):
    payload = {"gare_depart": DEP, "gare_arrivee": ARR,
               "nb_train_prevu": 480, "duree_moyenne": 150}
    body = client.post("/predict", json=payload).json()
    assert body["predicted_delay_minutes"] >= 0


def test_predict_is_case_and_whitespace_insensitive(client):
    body = client.post(
        "/predict",
        json={"gare_depart": f"  {DEP.lower()} ", "gare_arrivee": ARR.lower()},
    ).json()
    assert body["gare_depart"] == DEP


def test_predict_rejects_an_unknown_route(client):
    response = client.post(
        "/predict", json={"gare_depart": "PARIS NORD", "gare_arrivee": "TOKYO"}
    )
    assert response.status_code == 404
    assert "/routes" in response.json()["detail"]


def test_predict_rejects_a_month_beyond_the_horizon(client):
    response = client.post(
        "/predict", json={"gare_depart": DEP, "gare_arrivee": ARR, "month": "2099-01"}
    )
    assert response.status_code == 422
    assert "one month ahead" in response.json()["detail"]


def test_predict_rejects_a_malformed_month(client):
    response = client.post(
        "/predict", json={"gare_depart": DEP, "gare_arrivee": ARR, "month": "juillet"}
    )
    assert response.status_code == 422


def test_served_prediction_matches_the_training_pipeline(client):
    """No train/serve skew: same row, same number, to the last digit."""
    import api

    art = api.artifacts()
    month = art.last_month
    served = client.post(
        "/predict",
        json={"gare_depart": DEP, "gare_arrivee": ARR, "month": str(month)},
    ).json()["predicted_delay_minutes"]

    engineered = dp.engineer_features(dp.to_panel(art.history))
    row = engineered[
        (engineered["gare_depart"] == DEP)
        & (engineered["gare_arrivee"] == ARR)
        & (engineered[dp.PERIOD] == month)
    ]
    offline = dp.predict_delays(art.model, row)[0]
    assert served == pytest.approx(round(float(offline), 1))

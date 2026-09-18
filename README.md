# 🚆 SNCF Delay Predictor

Predicts expected train delay (in minutes) from route, schedule, and weather
conditions, using a Random Forest regression model. End-to-end ML project:
data pipeline → training → evaluation → REST API → tests → CI.

## Why this project

Built to apply machine learning to a domain I know well (railway
infrastructure, from my apprenticeship at SNCF Réseau) while practicing the
full ML engineering lifecycle — not just a notebook, but code that could
plausibly ship: a reusable data pipeline, an inference API with no
train/serve skew, unit tests, and CI.

> **Note on data**: this repo ships with a synthetic data generator
> (`src/generate_data.py`) that mimics the schema of SNCF's public
> [open data](https://data.sncf.com) on train regularity, with realistic
> injected relationships (rush hour, weather, route length). This keeps the
> project fully reproducible without needing an API key or violating any
> data confidentiality. See [Using real data](#using-real-data) to swap in
> the actual open dataset.

## Architecture

```
generate_data.py  →  raw_regularity.csv
                            │
                            ▼
                     data_pipeline.py   (clean, feature engineer, encode)
                            │
                            ▼
                       train.py         (RandomForestRegressor, eval, save)
                            │
                            ▼
                  models/*.joblib + metrics.json
                            │
                            ▼
                        api.py          (FastAPI /predict endpoint)
```

## Results

On held-out test data (20% split, synthetic dataset):

| Metric | Value |
|---|---|
| MAE | ~3.2 minutes |
| RMSE | ~4.0 minutes |
| R² | ~0.60 |

Feature importance confirms the model learns sensible relationships: weather
severity and rush-hour timing dominate the prediction, as expected for this
domain.

## Quickstart

```bash
git clone https://github.com/<your-username>/sncf-delay-predictor.git
cd sncf-delay-predictor
pip install -r requirements.txt

# 1. Generate the synthetic dataset
cd src
python generate_data.py

# 2. Train the model
python train.py --data ../data/raw_regularity.csv --model-dir ../models

# 3. Serve predictions
uvicorn api:app --reload
# → open http://127.0.0.1:8000/docs
```

### Example request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "departure_station": "Paris",
    "arrival_station": "Lyon",
    "distance_km": 465,
    "scheduled_hour": 8,
    "is_weekend": false,
    "season": "winter",
    "weather_severity": 0.7,
    "planned_trains": 20,
    "date": "2026-01-15"
  }'
# → {"predicted_delay_minutes": 20.9}
```

## Running tests

```bash
pytest tests/ -v
```

## Docker

```bash
docker build -t sncf-delay-predictor .
docker run -p 8000:8000 sncf-delay-predictor
```

## Using real data

To train on actual SNCF open data instead of the synthetic generator:

1. Download a "régularité mensuelle" CSV from [data.sncf.com](https://data.sncf.com)
2. Map its columns to match `RAW_COLUMNS` in `src/generate_data.py`
3. Save it as `data/raw_regularity.csv`
4. Run `python train.py` as usual — the pipeline is agnostic to data source

## Possible extensions

- [ ] Swap Random Forest for Gradient Boosting (XGBoost/LightGBM) and compare
- [ ] Add MLflow experiment tracking
- [ ] Hyperparameter tuning with Optuna
- [ ] Deploy the API on Render / Hugging Face Spaces / Fly.io
- [ ] Add a simple Streamlit front-end for interactive predictions

## Tech stack

Python · pandas · scikit-learn · FastAPI · pytest · Docker · GitHub Actions

## License

MIT

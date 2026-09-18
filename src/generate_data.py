"""
Generates a synthetic dataset that mimics the structure of SNCF's public
"regularité mensuelle TGV/TER" open data (https://data.sncf.com).

Why synthetic data? This project is designed to run out-of-the-box for a
portfolio/demo without requiring API access. To use REAL data instead:
  1. Download a CSV from https://data.sncf.com (e.g. "regularite-mensuelle-ter")
  2. Rename columns to match RAW_COLUMNS below (or adapt data_pipeline.py)
  3. Skip this script and point train.py at your CSV

The synthetic generator injects realistic relationships (rush hour, season,
route length, weather) so the model has real signal to learn from -- it is
not pure noise.
"""

import numpy as np
import pandas as pd

RAW_COLUMNS = [
    "date", "route", "departure_station", "arrival_station",
    "distance_km", "scheduled_hour", "is_weekend", "season",
    "weather_severity", "planned_trains", "delay_minutes", "cancelled",
]

ROUTES = [
    ("Paris", "Lyon", 465), ("Paris", "Marseille", 775),
    ("Paris", "Lille", 225), ("Paris", "Bordeaux", 585),
    ("Paris", "Strasbourg", 490), ("Lyon", "Marseille", 315),
    ("Paris", "Nantes", 385), ("Paris", "Rennes", 350),
]

SEASONS = ["winter", "spring", "summer", "autumn"]


def _season_from_month(month: int) -> str:
    return SEASONS[(month % 12) // 3]


def generate(n_rows: int = 8000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    dates = pd.date_range("2022-01-01", "2025-12-31", freq="D")

    for _ in range(n_rows):
        date = rng.choice(dates)
        date = pd.Timestamp(date)
        dep, arr, dist = ROUTES[rng.integers(0, len(ROUTES))]
        hour = int(rng.integers(5, 23))
        is_weekend = date.dayofweek >= 5
        season = _season_from_month(date.month)
        # winter/autumn -> more weather disruption
        base_weather = {"winter": 0.6, "autumn": 0.4, "spring": 0.2, "summer": 0.1}[season]
        weather_severity = float(np.clip(rng.normal(base_weather, 0.2), 0, 1))
        planned_trains = int(rng.integers(4, 40))

        # --- signal the model should learn ---
        delay = 0.0
        delay += dist / 100 * 0.6                       # longer routes -> more delay
        delay += weather_severity * 18                   # weather impact
        if hour in (7, 8, 9, 17, 18, 19):                 # rush hour
            delay += 4.5
        if is_weekend:
            delay -= 2.0                                  # less congestion
        delay += rng.normal(0, 4)                          # noise
        delay = max(0.0, delay)

        cancel_prob = 0.01 + weather_severity * 0.07
        cancelled = int(rng.random() < cancel_prob)
        if cancelled:
            delay = 0.0  # cancelled trains aren't "delayed"

        rows.append([
            date.date().isoformat(), f"{dep}-{arr}", dep, arr, dist, hour,
            int(is_weekend), season, round(weather_severity, 2),
            planned_trains, round(delay, 1), cancelled,
        ])

    return pd.DataFrame(rows, columns=RAW_COLUMNS)


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/raw_regularity.csv", index=False)
    print(f"Wrote {len(df)} rows to data/raw_regularity.csv")
    print(df.head())

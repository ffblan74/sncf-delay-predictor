"""
Trains a delay-prediction regressor and saves the artifacts needed for
serving (model + fitted encoder + feature list) to models/.

Usage:
    python src/train.py --data data/raw_regularity.csv
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from data_pipeline import run_pipeline, NUMERIC_FEATURES, CATEGORICAL_COLS


def train(data_path: str, model_dir: str = "models"):
    Path(model_dir).mkdir(exist_ok=True)

    X, y, encoder, _ = run_pipeline(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )

    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    preds = model.predict(X_test)
    metrics = {
        "mae_minutes": round(mean_absolute_error(y_test, preds), 3),
        "rmse_minutes": round(float(np.sqrt(mean_squared_error(y_test, preds))), 3),
        "r2": round(r2_score(y_test, preds), 3),
        "train_time_seconds": round(train_time, 2),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    joblib.dump(model, f"{model_dir}/model.joblib")
    joblib.dump(encoder, f"{model_dir}/encoder.joblib")
    with open(f"{model_dir}/feature_columns.json", "w") as f:
        json.dump(list(X.columns), f)
    with open(f"{model_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # simple feature importance report
    importances = sorted(
        zip(X.columns, model.feature_importances_), key=lambda t: -t[1]
    )[:8]

    print("=== Evaluation on held-out test set ===")
    for k, v in metrics.items():
        print(f"{k:>18}: {v}")
    print("\n=== Top feature importances ===")
    for name, score in importances:
        print(f"{name:>25}: {score:.3f}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw_regularity.csv")
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()
    train(args.data, args.model_dir)

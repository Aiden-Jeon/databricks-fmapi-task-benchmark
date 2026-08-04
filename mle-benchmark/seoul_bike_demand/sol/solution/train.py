#!/usr/bin/env python3
"""Train the bike demand ensemble and create outputs/submission.csv."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor


SEED = 42
TARGET = "rented_bike_count"


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create calendar features using only information available at prediction time."""
    dt = pd.to_datetime(frame["date"], format="%d/%m/%Y")
    features = frame.drop(columns=["id", "date", TARGET], errors="ignore").copy()
    features["seasons"] = features["seasons"].map(
        {"Winter": 0, "Spring": 1, "Summer": 2, "Autumn": 3}
    )
    features["holiday"] = (features["holiday"] == "Holiday").astype(np.int8)
    features["functioning_day"] = (features["functioning_day"] == "Yes").astype(np.int8)
    features["day_index"] = (dt - pd.Timestamp("2017-12-01")).dt.days
    features["month"] = dt.dt.month
    features["day_of_week"] = dt.dt.dayofweek
    features["day_of_year"] = dt.dt.dayofyear
    features["hour_of_week"] = dt.dt.dayofweek * 24 + frame["hour"]

    cyclic_values = {
        "hour": (frame["hour"], 24),
        "dow": (dt.dt.dayofweek, 7),
        "year": (dt.dt.dayofyear, 365.25),
    }
    for name, (values, period) in cyclic_values.items():
        features[f"{name}_sin"] = np.sin(2 * np.pi * values / period)
        features[f"{name}_cos"] = np.cos(2 * np.pi * values / period)
    return features


def main() -> None:
    workspace = Path(__file__).resolve().parent.parent
    train = pd.read_csv(workspace / "train.csv")
    test = pd.read_csv(workspace / "test.csv")
    sample = pd.read_csv(workspace / "sample_submission.csv")

    if train["id"].duplicated().any() or test["id"].duplicated().any():
        raise ValueError("Input IDs must be unique")
    if train[TARGET].isna().any():
        raise ValueError("Training target contains missing values")

    X_train = make_features(train)
    X_test = make_features(test)
    y_train = train[TARGET].to_numpy()

    hist = HistGradientBoostingRegressor(
        learning_rate=0.06,
        max_iter=500,
        max_leaf_nodes=31,
        min_samples_leaf=15,
        l2_regularization=2.0,
        random_state=SEED,
    )
    extra = ExtraTreesRegressor(
        n_estimators=500,
        min_samples_leaf=1,
        max_features=0.9,
        n_jobs=-1,
        random_state=SEED,
    )
    hist.fit(X_train, y_train)
    extra.fit(X_train, y_train)

    predictions = 0.65 * hist.predict(X_test) + 0.35 * extra.predict(X_test)
    predictions = np.maximum(predictions, 0.0)
    predictions[test["functioning_day"].to_numpy() == "No"] = 0.0

    submission = pd.DataFrame({"id": test["id"], TARGET: predictions})
    if list(submission.columns) != list(sample.columns):
        raise ValueError("Submission columns do not match sample_submission.csv")
    if len(submission) != len(test) or set(submission["id"]) != set(test["id"]):
        raise ValueError("Submission IDs do not exactly match test IDs")
    if not np.isfinite(submission[TARGET]).all():
        raise ValueError("Predictions contain non-finite values")

    output_dir = workspace / "outputs"
    output_dir.mkdir(exist_ok=True)
    submission.to_csv(output_dir / "submission.csv", index=False)
    print(f"Wrote {len(submission)} predictions to {output_dir / 'submission.csv'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Final LightGBM ensemble model for Seoul Bike demand prediction.

Key insights from EDA:
- Test is Autumn (Sep 19 - Nov 30 2018) with cool temps (mean 12C).
- Train's Autumn is only Sep 1-18 (warm, 22C). The 'seasons' label is therefore
  misleading: in train "Autumn" means warm early-September, while in test
  "Autumn" spans warm-then-cool Sep-Nov. We EXCLUDE the 'seasons' feature
  (it has no variance in test anyway, and only encourages memorization of
  train's warm-autumn pattern).
- Similarly we drop month / dayofyear / day features: they encode calendar
  seasonality that does not transfer to the cooler test months. The model
  generalizes better when it must learn the temperature -> demand mapping
  rather than memorizing "this month -> this demand".
- functioning_day == 'No' -> always 0 in train, so we hard-set those to 0.
- We keep a time-trend feature (days_since_start) which captures growth.
- Validation: a "cool" holdout (Mar-Apr 2018, mean temp 10.5C, close to
  test's 12C) estimates test performance (~RMSE 210-220). A chronological
  holdout (last 14 days, warm autumn) gives ~RMSE 196. Test, which starts
  warm and cools, likely falls between these.

Model: 5-seed LightGBM ensemble with fixed 800 boosting rounds, regularized.
Expected test RMSE: ~210-230.
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASK = HERE.parent
TRAIN_CSV = TASK / "train.csv"
TEST_CSV = TASK / "test.csv"
SAMPLE_CSV = TASK / "sample_submission.csv"
OUT_CSV = TASK / "outputs" / "submission.csv"

SEEDS = [42, 1, 7, 100, 2024]
NUM_ROUNDS = 800
RANDOM_STATE = 42


def build_features(df, id_col, ref_date_min=None):
    """Engineer features. Drops calendar-seasonality features that do not
    transfer from train's warm autumn to test's cool autumn."""
    df = df.copy()
    ymd = df[id_col].str[:8]
    hour = df[id_col].str[9:11].astype(int)
    df["hour"] = df.get("hour", hour)
    date_dt = pd.to_datetime(ymd, format="%Y%m%d")
    df["year"] = date_dt.dt.year
    df["dayofweek"] = date_dt.dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    if ref_date_min is None:
        ref_date_min = date_dt.min()
    df["days_since_start"] = (date_dt - ref_date_min).dt.days
    df["days_since_start_norm"] = df["days_since_start"] / 365.0

    # Categorical (note: 'seasons' intentionally excluded - see module docstring)
    df["holiday"] = df["holiday"].astype("category")
    df["functioning_day"] = df["functioning_day"].astype("category")

    # Weather interactions / derived
    df["temp_humid"] = df["temperature_c"] * df["humidity_pct"] / 100.0
    df["discomfort"] = df["temperature_c"] + 0.55 * (1 - df["humidity_pct"] / 100.0) * (df["temperature_c"] - 14.5)
    df["wind_chill"] = df["temperature_c"] - 0.2 * df["wind_speed_ms"]
    df["has_rain"] = (df["rainfall_mm"] > 0).astype(int)
    df["has_snow"] = (df["snowfall_cm"] > 0).astype(int)
    df["rain_or_snow"] = ((df["rainfall_mm"] > 0) | (df["snowfall_cm"] > 0)).astype(int)
    df["log_rainfall"] = np.log1p(df["rainfall_mm"])
    df["log_snowfall"] = np.log1p(df["snowfall_cm"])
    df["log_solar"] = np.log1p(df["solar_radiation_mj"])
    df["low_visibility"] = (df["visibility_10m"] < 2000).astype(int)
    df["log_visibility"] = np.log1p(df["visibility_10m"])

    # Temperature bin as categorical - captures non-linear temp effect
    df["temp_bin"] = pd.cut(
        df["temperature_c"],
        bins=[-50, -10, 0, 10, 20, 30, 50],
        labels=["v_cold", "cold", "cool", "mild", "warm", "hot"],
    ).astype("category")

    # Cyclical encoding of hour and day-of-week (these transfer across seasons)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    # Rush hour / time-of-day flags
    df["is_rush_morning"] = ((df["hour"] >= 7) & (df["hour"] <= 9)).astype(int)
    df["is_rush_evening"] = ((df["hour"] >= 17) & (df["hour"] <= 19)).astype(int)
    df["is_peak_evening"] = (df["hour"] == 18).astype(int)
    df["is_night"] = ((df["hour"] >= 0) & (df["hour"] <= 5)).astype(int)
    df["is_daytime"] = ((df["hour"] >= 7) & (df["hour"] <= 20)).astype(int)

    # Interactions
    df["rush_morning_weekend"] = df["is_rush_morning"] * df["is_weekend"]
    df["rush_evening_weekend"] = df["is_rush_evening"] * df["is_weekend"]
    df["hour_holiday"] = df["hour"] * (df["holiday"] == "Holiday").astype(int)

    return df


# NOTE: 'seasons', 'month', 'dayofyear', 'day' intentionally excluded -
# they encode calendar seasonality that does not generalize from train's
# warm autumn to test's cool autumn (see module docstring).
CAT_COLS = ["holiday", "functioning_day", "temp_bin"]
NUM_COLS = [
    "hour", "temperature_c", "humidity_pct", "wind_speed_ms", "visibility_10m",
    "dew_point_c", "solar_radiation_mj", "rainfall_mm", "snowfall_cm",
    "dayofweek", "is_weekend",
    "days_since_start", "days_since_start_norm",
    "temp_humid", "discomfort", "wind_chill",
    "has_rain", "has_snow", "rain_or_snow",
    "log_rainfall", "log_snowfall", "log_solar", "low_visibility", "log_visibility",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_rush_morning", "is_rush_evening", "is_peak_evening", "is_night", "is_daytime",
    "rush_morning_weekend", "rush_evening_weekend", "hour_holiday",
]
FEAT_COLS = NUM_COLS + CAT_COLS


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def get_params(seed):
    return {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.04,
        "num_leaves": 48,
        "min_data_in_leaf": 30,
        "feature_fraction": 0.6,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "lambda_l1": 0.5,
        "lambda_l2": 10.0,
        "verbose": -1,
        "seed": seed,
    }


def evaluate_val(X, y, feat_train, mask, label):
    """Train on non-mask, evaluate on mask (with functioning_day=No -> 0)."""
    Xtr, Xval = X[~mask][FEAT_COLS], X[mask][FEAT_COLS]
    ytr, yval = y[~mask], y[mask]
    nm = (feat_train.loc[mask, "functioning_day"] == "No").values
    preds_sum = np.zeros(len(Xval))
    for s in SEEDS:
        dtr = lgb.Dataset(Xtr, ytr, categorical_feature=CAT_COLS, free_raw_data=False)
        m = lgb.train(get_params(s), dtr, num_boost_round=NUM_ROUNDS,
                       valid_sets=[], callbacks=[lgb.log_evaluation(0)])
        preds_sum += m.predict(Xval)
    pred = preds_sum / len(SEEDS)
    pred[nm] = 0.0
    pred = np.clip(pred, 0, None)
    r = rmse(yval.values, pred)
    print(f"  [{label}] val RMSE = {r:.4f}  (n={mask.sum()}, temp_mean={feat_train.loc[mask,'temperature_c'].mean():.1f})")
    return r


def main():
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    sample = pd.read_csv(SAMPLE_CSV)

    ref_date_min = pd.to_datetime(train["id"].str[:8], format="%Y%m%d").min()
    feat_train = build_features(train, "id", ref_date_min)
    feat_test = build_features(test, "id", ref_date_min)

    X = feat_train[FEAT_COLS].copy()
    y = feat_train["rented_bike_count"].astype(float)

    print("=== Validation (5-seed ensemble, fixed %d rounds) ===" % NUM_ROUNDS)
    # Cool-temp holdout (Mar-Apr 2018, mean temp 10.5C ~ test 12C): primary estimate
    mask_cool = (feat_train["id"] >= "20180301_00") & (feat_train["id"] < "20180501_00")
    evaluate_val(X, y, feat_train, mask_cool, "cool (Mar-Apr)")
    # Chronological holdout (last 14 days, warm autumn): secondary
    mask_chrono = feat_train["id"] >= "20180905_00"
    evaluate_val(X, y, feat_train, mask_chrono, "chrono (last 14d)")

    print("\n=== Training final ensemble on ALL train data ===")
    X_full = X[FEAT_COLS]
    y_full = y
    X_test = feat_test[FEAT_COLS].copy()

    preds_sum = np.zeros(len(X_test))
    for i, s in enumerate(SEEDS, 1):
        dtr = lgb.Dataset(X_full, y_full, categorical_feature=CAT_COLS, free_raw_data=False)
        m = lgb.train(get_params(s), dtr, num_boost_round=NUM_ROUNDS,
                       valid_sets=[], callbacks=[lgb.log_evaluation(0)])
        preds_sum += m.predict(X_test)
        print(f"  trained seed {s} ({i}/{len(SEEDS)})")
    preds = preds_sum / len(SEEDS)

    # functioning_day == 'No' -> 0 (always 0 in train)
    no_mask = (feat_test["functioning_day"] == "No").values
    preds[no_mask] = 0.0
    preds = np.clip(preds, 0, None)

    # Build submission aligned with sample order
    test_ids = feat_test["id"].values
    pred_df = pd.DataFrame({"id": test_ids, "rented_bike_count": preds})
    sub = sample[["id"]].merge(pred_df, on="id", how="left")
    sub["rented_bike_count"] = sub["rented_bike_count"].fillna(0)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV} with {len(sub)} rows")
    print("Prediction stats:")
    print(sub["rented_bike_count"].describe())

    # Feature importance (from first seed model, for inspection)
    dtr0 = lgb.Dataset(X_full, y_full, categorical_feature=CAT_COLS, free_raw_data=False)
    m0 = lgb.train(get_params(SEEDS[0]), dtr0, num_boost_round=NUM_ROUNDS,
                    valid_sets=[], callbacks=[lgb.log_evaluation(0)])
    imp = pd.DataFrame({
        "feat": m0.feature_name(),
        "imp": m0.feature_importance(importance_type="gain"),
    }).sort_values("imp", ascending=False).head(15)
    print("\nTop 15 features (seed 0):")
    print(imp.to_string(index=False))


if __name__ == "__main__":
    main()

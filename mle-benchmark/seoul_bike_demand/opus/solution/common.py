"""Shared feature engineering for t5_bike (Seoul bike hourly demand)."""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = "rented_bike_count"
ID = "id"

WEATHER = ["temperature_c", "humidity_pct", "wind_speed_ms", "visibility_10m",
           "dew_point_c", "solar_radiation_mj", "rainfall_mm", "snowfall_cm"]


def load(path_train=None, path_test=None):
    tr = pd.read_csv(path_train or os.path.join(ROOT, "train.csv"))
    te = pd.read_csv(path_test or os.path.join(ROOT, "test.csv"))
    return tr, te


def _base(df):
    df = df.copy()
    d = pd.to_datetime(df["date"], dayfirst=True)
    df["dt"] = d + pd.to_timedelta(df["hour"], unit="h")
    df["dayofweek"] = d.dt.dayofweek
    df["month"] = d.dt.month
    df["day"] = d.dt.day
    df["doy"] = d.dt.dayofyear
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["holiday_f"] = (df["holiday"] == "Holiday").astype(int)
    df["func"] = (df["functioning_day"] == "Yes").astype(int)
    # non-working day = weekend or holiday
    df["nonwork"] = ((df["is_weekend"] == 1) | (df["holiday_f"] == 1)).astype(int)
    return df


def build_features(tr_raw, te_raw):
    """Build features on the concatenated timeline (weather-only lags -> no target leakage)."""
    tr = _base(tr_raw)
    te = _base(te_raw)
    tr["_is_test"] = 0
    te["_is_test"] = 1
    te[TARGET] = np.nan
    all_df = pd.concat([tr, te], ignore_index=True, sort=False)
    all_df = all_df.sort_values("dt").reset_index(drop=True)

    # cyclic encodings
    all_df["hour_sin"] = np.sin(2 * np.pi * all_df["hour"] / 24)
    all_df["hour_cos"] = np.cos(2 * np.pi * all_df["hour"] / 24)
    all_df["doy_sin"] = np.sin(2 * np.pi * all_df["doy"] / 365)
    all_df["doy_cos"] = np.cos(2 * np.pi * all_df["doy"] / 365)

    # weather transforms
    all_df["rain_flag"] = (all_df["rainfall_mm"] > 0).astype(int)
    all_df["snow_flag"] = (all_df["snowfall_cm"] > 0).astype(int)
    all_df["log_rain"] = np.log1p(all_df["rainfall_mm"])
    all_df["temp_hum"] = all_df["temperature_c"] * all_df["humidity_pct"] / 100.0
    all_df["temp_minus_dew"] = all_df["temperature_c"] - all_df["dew_point_c"]
    # apparent temperature (simple wind-chill / heat proxy)
    t, w, h = all_df["temperature_c"], all_df["wind_speed_ms"], all_df["humidity_pct"]
    all_df["apparent_temp"] = t - 0.7 * w + 0.05 * (h - 50) * np.maximum(t - 20, 0) / 10.0

    # lag / lead weather (weather is known for test period -> legitimate)
    for c in ["temperature_c", "rainfall_mm", "humidity_pct", "solar_radiation_mj"]:
        all_df[f"{c}_lag1"] = all_df[c].shift(1)
        all_df[f"{c}_lag3"] = all_df[c].shift(3)
        all_df[f"{c}_lead1"] = all_df[c].shift(-1)
    all_df["rain_roll3"] = all_df["rainfall_mm"].rolling(3, min_periods=1).sum()
    all_df["rain_roll6"] = all_df["rainfall_mm"].rolling(6, min_periods=1).sum()
    all_df["rain_roll24"] = all_df["rainfall_mm"].rolling(24, min_periods=1).sum()
    all_df["temp_roll24"] = all_df["temperature_c"].rolling(24, min_periods=1).mean()

    # daily weather aggregates
    day_key = all_df["dt"].dt.normalize()
    all_df["_dk"] = day_key
    g = all_df.groupby("_dk")
    all_df["temp_day_mean"] = g["temperature_c"].transform("mean")
    all_df["temp_day_max"] = g["temperature_c"].transform("max")
    all_df["temp_day_min"] = g["temperature_c"].transform("min")
    all_df["rain_day_sum"] = g["rainfall_mm"].transform("sum")
    all_df["solar_day_sum"] = g["solar_radiation_mj"].transform("sum")
    all_df["hum_day_mean"] = g["humidity_pct"].transform("mean")
    all_df["temp_dev_day"] = all_df["temperature_c"] - all_df["temp_day_mean"]

    # trend (days since start of data)
    all_df["t_idx"] = (all_df["dt"] - all_df["dt"].min()).dt.total_seconds() / 86400.0

    all_df = all_df.drop(columns=["_dk"])
    tr_out = all_df[all_df["_is_test"] == 0].reset_index(drop=True)
    te_out = all_df[all_df["_is_test"] == 1].reset_index(drop=True)
    return tr_out, te_out


FEATS = (
    WEATHER
    + ["hour", "dayofweek", "month", "day", "doy", "is_weekend", "holiday_f", "nonwork",
       "hour_sin", "hour_cos", "doy_sin", "doy_cos",
       "rain_flag", "snow_flag", "log_rain", "temp_hum", "temp_minus_dew", "apparent_temp",
       "rain_roll3", "rain_roll6", "rain_roll24", "temp_roll24",
       "temp_day_mean", "temp_day_max", "temp_day_min", "rain_day_sum", "solar_day_sum",
       "hum_day_mean", "temp_dev_day"]
    + [f"{c}_{s}" for c in ["temperature_c", "rainfall_mm", "humidity_pct", "solar_radiation_mj"]
       for s in ["lag1", "lag3", "lead1"]]
)


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))

"""공통 특성 엔지니어링 및 시계열 검증 유틸."""
import numpy as np
import pandas as pd

TARGET = "rented_bike_count"
ID_COL = "id"


def load_data():
    train = pd.read_csv("train.csv", parse_dates=["date"], dayfirst=True)
    test = pd.read_csv("test.csv", parse_dates=["date"], dayfirst=True)
    train["_is_train"] = 1
    test["_is_train"] = 0
    df = pd.concat([train, test], ignore_index=True, sort=False)
    df = df.sort_values(["date", "hour"]).reset_index(drop=True)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dow"] = df["date"].dt.dayofweek
    df["weekend"] = (df["dow"] >= 5).astype(int)
    df["dayofyear"] = df["date"].dt.dayofyear
    # 시간/주기 주기적 인코딩
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    # 범주형 -> 수치
    df["seasons"] = df["seasons"].map(
        {"Spring": 0, "Summer": 1, "Autumn": 2, "Winter": 3}
    ).astype(int)
    df["holiday"] = (df["holiday"] == "Holiday").astype(int)
    df["functioning_day"] = (df["functioning_day"] == "Yes").astype(int)
    # 파생 기상 특성
    df["temp_humidity"] = df["temperature_c"] * df["humidity_pct"]
    df["rain_flag"] = (df["rainfall_mm"] > 0).astype(int)
    df["snow_flag"] = (df["snowfall_cm"] > 0).astype(int)
    df["bad_weather"] = ((df["rainfall_mm"] > 0) | (df["snowfall_cm"] > 0)).astype(int)
    # 출퇴근 피크 지표
    df["peak_morning"] = df["hour"].isin([7, 8, 9]).astype(int)
    df["peak_evening"] = df["hour"].isin([17, 18, 19]).astype(int)
    return df


FEATURES = [
    "hour", "temperature_c", "humidity_pct", "wind_speed_ms",
    "visibility_10m", "dew_point_c", "solar_radiation_mj",
    "rainfall_mm", "snowfall_cm", "seasons", "holiday", "functioning_day",
    "year", "month", "day", "dow", "weekend", "dayofyear",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos", "dow_sin", "dow_cos",
    "temp_humidity", "rain_flag", "snow_flag", "bad_weather",
    "peak_morning", "peak_evening",
]


def time_folds(df, n_folds=4, val_days=21):
    """시간순(미래 누출 없는) fold 인덱스 생성.

    마지막 val_days * n_folds 일 구간을 fold로 나누어 검증.
    """
    dates = np.sort(df.loc[df["_is_train"] == 1, "date"].unique())
    cutoffs = []
    for i in range(n_folds, 0, -1):
        cutoffs.append(dates[-val_days * i])
    folds = []
    for c in cutoffs:
        tr_idx = df.index[(df["_is_train"] == 1) & (df["date"] < c)].to_numpy()
        va_idx = df.index[(df["_is_train"] == 1) & (df["date"] >= c)].to_numpy()
        folds.append((tr_idx, va_idx))
    return folds

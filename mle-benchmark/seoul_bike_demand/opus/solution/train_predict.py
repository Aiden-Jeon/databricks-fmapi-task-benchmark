"""Final reproducible pipeline for t5_bike (Seoul hourly bike demand).

Design decisions (all validated with rolling-origin chronological CV, see exp0*.py):
  * Only rows with functioning_day == 'Yes' are used for training; the target is
    identically 0 whenever functioning_day == 'No', so those test rows are forced to 0.
  * Calendar-seasonal features (month / day / dayofyear / their cyclic encodings) are
    EXCLUDED: the test period (Sep 19 - Nov 30) lies entirely outside the training
    day-of-year range, and CV folds showed catastrophic degradation (RMSE 505 vs 340
    on one fold) when trees extrapolate on those columns. Seasonality is carried by
    the weather channels instead (temperature, solar radiation, humidity, ...).
  * No explicit adoption-trend extrapolation: leave-one-month-out diagnostics showed a
    weather-only model is already well calibrated (level ratio ~1.0), and forced trend
    offsets badly hurt every long-horizon fold except the earliest one.
  * Poisson objective (log link, models the conditional mean) beat L2 on raw / sqrt /
    log targets for RMSE. Final prediction = multi-seed blend of Poisson + Tweedie(1.3)
    + sqrt-L2 LightGBM models.

Usage:  python solution/train_predict.py
Writes: outputs/submission.csv
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb

from common import ROOT, load, build_features, FEATS, TARGET, rmse

SEASONAL = ["month", "day", "doy", "doy_sin", "doy_cos"]
F = [f for f in FEATS if f not in SEASONAL]

BASE = dict(learning_rate=0.04, num_leaves=31, min_data_in_leaf=30, feature_fraction=0.8,
            bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=2)

# (name, weight, n_rounds, extra params, target transform)
# equal-weight blend chosen by rolling-origin CV (pooled RMSE 266.5, worst fold 326)
MEMBERS = [
    ("po_ff5", 1.0, 1800, dict(objective="poisson", feature_fraction=0.5), None),
    ("po_reg", 1.0, 2500, dict(objective="poisson", num_leaves=15, min_data_in_leaf=50), None),
    ("tweedie13", 1.0, 1400, dict(objective="tweedie", tweedie_variance_power=1.3), None),
]
SEEDS = [7, 42, 2024, 31337, 555]


def fit_predict(dtr, X, params, nr, tf, seed):
    y = dtr[TARGET].values.astype(float)
    yt = y if tf is None else {"sqrt": np.sqrt(y), "log": np.log1p(y)}[tf]
    p = dict(BASE); p.update(params); p["seed"] = seed
    p["bagging_seed"] = seed + 1
    p["feature_fraction_seed"] = seed + 2
    model = lgb.train(p, lgb.Dataset(dtr[F], yt), num_boost_round=nr)
    q = model.predict(X)
    if tf == "sqrt":
        q = np.maximum(q, 0) ** 2
    elif tf == "log":
        q = np.expm1(q)
    return np.clip(q, 0, None), model


def ensemble(dtr, X):
    total = np.zeros(len(X))
    wsum = 0.0
    for name, w, nr, params, tf in MEMBERS:
        acc = np.zeros(len(X))
        for s in SEEDS:
            p, _ = fit_predict(dtr, X, params, nr, tf, s)
            acc += p
        acc /= len(SEEDS)
        total += w * acc
        wsum += w
        print(f"  member {name:10s} w={w} mean_pred={acc.mean():8.1f}")
    return total / wsum


def main():
    tr_raw, te_raw = load()
    tr, te = build_features(tr_raw, te_raw)
    trf = tr[tr["func"] == 1].reset_index(drop=True)
    print(f"train rows={len(tr)} functioning={len(trf)} | test rows={len(te)} "
          f"non-functioning(forced 0)={(te['func'] == 0).sum()}")

    # --- honest holdout report: last 28 days of train ---
    cut = trf["dt"].max() - pd.Timedelta(days=28)
    ho_tr, ho_va = trf[trf["dt"] <= cut], trf[trf["dt"] > cut]
    if len(ho_va) > 100:
        p = ensemble(ho_tr, ho_va[F])
        print(f"[holdout last 28d] n={len(ho_va)} RMSE={rmse(ho_va[TARGET], p):.2f} "
              f"(mean_y={ho_va[TARGET].mean():.1f}, mean_p={p.mean():.1f})")

    # --- final model on all functioning training rows ---
    print("training final ensemble on full train ...")
    pred = ensemble(trf, te[F])
    pred = np.where(te["func"].values == 1, pred, 0.0)
    pred = np.clip(pred, 0, None)

    sub = pd.DataFrame({"id": te["id"].values, TARGET: np.round(pred, 2)})
    samp = pd.read_csv(os.path.join(ROOT, "sample_submission.csv"))
    sub = samp[["id"]].merge(sub, on="id", how="left")
    assert len(sub) == len(samp), "row count mismatch"
    assert sub[TARGET].notna().all(), "missing predictions"
    assert sub["id"].is_unique, "duplicate ids"
    out = os.path.join(ROOT, "outputs", "submission.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sub.to_csv(out, index=False)
    print("wrote", out)
    print(sub[TARGET].describe().round(1).to_string())


if __name__ == "__main__":
    main()

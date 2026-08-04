"""Diagnose level bias / adoption trend: residual ratio by month + fold scale factors."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from common import load, build_features, FEATS, TARGET, rmse

tr_raw, te_raw = load()
tr, te = build_features(tr_raw, te_raw)
trf = tr[tr["func"] == 1].reset_index(drop=True)
SEASONAL = ["month", "day", "doy", "doy_sin", "doy_cos"]
F = [f for f in FEATS if f not in SEASONAL]  # weather+hour+dow only, no calendar season

PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=42, num_threads=2)

# ---- 1) out-of-fold residual ratio by month using K-fold random CV (no time info in feats)
from sklearn.model_selection import KFold
oof = np.zeros(len(trf))
for i, (a, b) in enumerate(KFold(5, shuffle=True, random_state=0).split(trf)):
    m = lgb.train(PARAMS, lgb.Dataset(trf.iloc[a][F], np.log1p(trf.iloc[a][TARGET])), num_boost_round=800)
    oof[b] = np.expm1(m.predict(trf.iloc[b][F]))
trf["oof"] = np.clip(oof, 0, None)
g = trf.groupby(trf["dt"].dt.to_period("M")).apply(
    lambda d: pd.Series({"actual": d[TARGET].mean(), "pred": d["oof"].mean(),
                         "ratio": d[TARGET].mean() / d["oof"].mean(), "n": len(d)}))
print("=== random-KFold OOF (weather-only feats) monthly ratio ===")
print(g.round(3).to_string())
print("random-KFold OOF rmse:", round(rmse(trf[TARGET], trf["oof"]), 2))

# ---- 2) chronological folds: bias direction of future periods
FOLDS = [("2018-04-20", "2018-05-19"), ("2018-05-20", "2018-06-19"), ("2018-06-20", "2018-07-19"),
         ("2018-07-20", "2018-08-18"), ("2018-08-19", "2018-09-19")]
print("\n=== chronological folds: needed scale factor k = mean(y)/mean(pred) ===")
for a, b in FOLDS:
    m_tr = trf["dt"] < a
    m_va = (trf["dt"] >= a) & (trf["dt"] < b)
    m = lgb.train(PARAMS, lgb.Dataset(trf[m_tr][F], np.log1p(trf[m_tr][TARGET])), num_boost_round=800)
    p = np.clip(np.expm1(m.predict(trf[m_va][F])), 0, None)
    y = trf.loc[m_va, TARGET].values
    print(f"{a} n_tr={m_tr.sum():5d} k={y.mean()/p.mean():.3f} rmse={rmse(y,p):7.2f} "
          f"rmse_k={rmse(y,p*(y.mean()/p.mean())):7.2f} mean_y={y.mean():7.1f} mean_p={p.mean():7.1f}")

# ---- 3) temperature-matched comparison across periods (weekday, daytime)
sub = trf[(trf["nonwork"] == 0) & (trf["hour"].between(7, 21)) & (trf["rainfall_mm"] == 0)].copy()
sub["tbin"] = pd.cut(sub["temperature_c"], [-20, 0, 5, 10, 15, 20, 25, 30, 40])
sub["per"] = np.select(
    [sub["dt"] < "2018-01-01", sub["dt"] < "2018-03-01", sub["dt"] < "2018-05-01",
     sub["dt"] < "2018-07-01", sub["dt"] < "2018-09-01"],
    ["2017-12", "2018-01-02", "2018-03-04", "2018-05-06", "2018-07-08"], "2018-09")
pv = sub.pivot_table(index="tbin", columns="per", values=TARGET, aggfunc="mean", observed=False)
print("\n=== mean count by temp bin x period (weekday daytime, no rain) ===")
print(pv.round(0).to_string())
print("\ncounts:")
print(sub.pivot_table(index="tbin", columns="per", values=TARGET, aggfunc="size", observed=False).to_string())

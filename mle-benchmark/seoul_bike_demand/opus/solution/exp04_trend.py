"""Leave-one-month-out level ratios + long-horizon folds for trend handling strategies."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from common import load, build_features, FEATS, TARGET, rmse

tr_raw, te_raw = load()
tr, te = build_features(tr_raw, te_raw)
trf = tr[tr["func"] == 1].reset_index(drop=True)
SEASONAL = ["month", "day", "doy", "doy_sin", "doy_cos"]
F = [f for f in FEATS if f not in SEASONAL]

P = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=20,
         feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
         verbose=-1, seed=42, num_threads=2)
NR = 800


def fit(df, feats, y, w=None):
    return lgb.train(P, lgb.Dataset(df[feats], y, weight=w), num_boost_round=NR)


# ---------- 1) leave-one-month-out level ratio (weather-only model) ----------
print("=== leave-one-month-out: actual/pred level ratio (weather-only model) ===")
trf["ym"] = trf["dt"].dt.to_period("M")
for ym in sorted(trf["ym"].unique()):
    m_va = trf["ym"] == ym
    m = fit(trf[~m_va], F, np.log1p(trf.loc[~m_va, TARGET]))
    p = np.clip(np.expm1(m.predict(trf.loc[m_va, F])), 0, None)
    y = trf.loc[m_va, TARGET].values
    print(f"{ym}  n={m_va.sum():4d} mean_y={y.mean():7.1f} mean_p={p.mean():7.1f} "
          f"ratio={y.mean()/p.mean():5.3f} logdiff={np.mean(np.log1p(y)-np.log1p(p)):+.3f} rmse={rmse(y,p):7.1f}")

# ---------- 2) long-horizon folds: strategies ----------
HFOLDS = [("2018-05-01", "2018-07-15"), ("2018-06-01", "2018-08-15"), ("2018-07-01", "2018-09-19")]


def strategies(dtr, dva):
    out = {}
    y = np.log1p(dtr[TARGET].values)
    tmax = dtr["t_idx"].max()
    # (a) plain weather model
    m = fit(dtr, F, y)
    out["plain"] = np.clip(np.expm1(m.predict(dva[F])), 0, None)
    # (b) + t_idx feature (trees clamp at train max)
    Ft = F + ["t_idx"]
    m = fit(dtr, Ft, y)
    out["t_feat"] = np.clip(np.expm1(m.predict(dva[Ft])), 0, None)
    # (c) with all calendar features
    m = fit(dtr, FEATS, y)
    out["full_cal"] = np.clip(np.expm1(m.predict(dva[FEATS])), 0, None)
    # (d) recency weighting (half-life in days)
    for hl in [60, 120]:
        w = 0.5 ** ((tmax - dtr["t_idx"].values) / hl)
        m = fit(dtr, F, y, w=w)
        out[f"recency{hl}"] = np.clip(np.expm1(m.predict(dva[F])), 0, None)
    # (e) trend-offset: target = log1p(y) - beta*t, add back beta*t_valid (linear extrapolation)
    for beta in [0.001, 0.002, 0.003, 0.005]:
        m = fit(dtr, F, y - beta * dtr["t_idx"].values)
        p = np.expm1(m.predict(dva[F]) + beta * dva["t_idx"].values)
        out[f"trend{beta}"] = np.clip(p, 0, None)
    return out


res = {}
for a, b in HFOLDS:
    dtr = trf[trf["dt"] < a]
    dva = trf[(trf["dt"] >= a) & (trf["dt"] < b)]
    yv = dva[TARGET].values
    outs = strategies(dtr, dva)
    print(f"\n--- fold train<{a} valid {a}..{b} (n_tr={len(dtr)}, n_va={len(dva)}, mean_y={yv.mean():.0f}) ---")
    for k, p in outs.items():
        r = rmse(yv, p)
        res.setdefault(k, []).append(r)
        print(f"  {k:12s} rmse={r:7.2f} k_opt={yv.mean()/p.mean():5.3f} rmse_kopt={rmse(yv,p*yv.mean()/p.mean()):7.2f}")

print("\n=== mean over long-horizon folds ===")
for k, v in sorted(res.items(), key=lambda x: np.mean(x[1])):
    print(f"{k:12s} {np.mean(v):7.2f}  {[round(x) for x in v]}")

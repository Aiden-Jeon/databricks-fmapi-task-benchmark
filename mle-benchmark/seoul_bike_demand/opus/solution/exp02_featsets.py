"""Compare feature sets / transforms with chronological expanding-window CV."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from common import load, build_features, FEATS, TARGET, rmse

tr_raw, te_raw = load()
tr, te = build_features(tr_raw, te_raw)
trf = tr[tr["func"] == 1].reset_index(drop=True)
print("target on functioning days: min", trf[TARGET].min(), "zeros", (trf[TARGET] == 0).sum())

SEASONAL = ["month", "day", "doy", "doy_sin", "doy_cos"]
FS = {
    "full": FEATS,
    "no_seasonal": [f for f in FEATS if f not in SEASONAL],
    "no_seasonal_t": [f for f in FEATS if f not in SEASONAL] + ["t_idx"],
    "cyc_only": [f for f in FEATS if f not in ("month", "day", "doy")],
}

FOLDS = [("2018-05-20", "2018-06-19"), ("2018-06-20", "2018-07-19"),
         ("2018-07-20", "2018-08-18"), ("2018-08-19", "2018-09-19")]

PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=42, num_threads=2)


def run(train_df, valid_df, feats, transform, n=1000):
    y = train_df[TARGET].values.astype(float)
    yt = {"log": np.log1p(y), "sqrt": np.sqrt(y), "raw": y}[transform]
    m = lgb.train(PARAMS, lgb.Dataset(train_df[feats], yt), num_boost_round=n)
    p = m.predict(valid_df[feats])
    if transform == "log":
        p = np.expm1(p)
    elif transform == "sqrt":
        p = np.maximum(p, 0) ** 2
    return np.clip(p, 0, None)


rows = []
for name, feats in FS.items():
    for transform in ["log", "sqrt"]:
        sc, scc = [], []
        for a, b in FOLDS:
            m_tr = trf["dt"] < a
            m_va = (trf["dt"] >= a) & (trf["dt"] < b)
            p = run(trf[m_tr], trf[m_va], feats, transform)
            yv = trf.loc[m_va, TARGET].values
            sc.append(rmse(yv, p))
            # in-sample-calibrated multiplicative correction (oracle upper bound check)
            k = yv.mean() / max(p.mean(), 1e-9)
            scc.append(rmse(yv, p * k))
        rows.append((name, transform, np.mean(sc), np.mean(scc), [round(s) for s in sc]))
        print(f"{name:14s} {transform:4s} rmse={np.mean(sc):7.2f} oracle_scaled={np.mean(scc):7.2f} {rows[-1][4]}")

print()
res = pd.DataFrame(rows, columns=["fs", "tf", "rmse", "oracle", "folds"]).sort_values("rmse")
print(res.to_string(index=False))

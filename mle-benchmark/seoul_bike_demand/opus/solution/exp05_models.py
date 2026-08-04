"""Robust rolling-origin CV (12 folds, 4-week validation) comparing models & blends."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from common import load, build_features, FEATS, TARGET, rmse

tr_raw, te_raw = load()
tr, te = build_features(tr_raw, te_raw)
trf = tr[tr["func"] == 1].reset_index(drop=True)

SEASONAL = ["month", "day", "doy", "doy_sin", "doy_cos"]
F_NS = [f for f in FEATS if f not in SEASONAL]
F_CYC = [f for f in FEATS if f not in ("month", "day", "doy")]

# rolling origin: validation start every 3 weeks from Mar 15, 28-day window
starts = pd.date_range("2018-03-15", "2018-08-22", freq="21D")
FOLDS = [(s, s + pd.Timedelta(days=28)) for s in starts]
print("folds:", [(str(a.date()), str(b.date())) for a, b in FOLDS])

LP = dict(objective="regression", learning_rate=0.04, num_leaves=63, min_data_in_leaf=20,
          feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
          verbose=-1, num_threads=2)


def lgbm(dtr, dva, feats, tf="log", seed=42, nr=900, **kw):
    y = dtr[TARGET].values.astype(float)
    yt = {"log": np.log1p(y), "sqrt": np.sqrt(y), "raw": y}[tf]
    p = dict(LP); p.update(kw); p["seed"] = seed
    m = lgb.train(p, lgb.Dataset(dtr[feats], yt), num_boost_round=nr)
    q = m.predict(dva[feats])
    if tf == "log":
        q = np.expm1(q)
    elif tf == "sqrt":
        q = np.maximum(q, 0) ** 2
    return np.clip(q, 0, None)


def sk(model, dtr, dva, feats, tf="log"):
    y = dtr[TARGET].values.astype(float)
    yt = {"log": np.log1p(y), "sqrt": np.sqrt(y), "raw": y}[tf]
    X = dtr[feats].fillna(-999).values
    model.fit(X, yt)
    q = model.predict(dva[feats].fillna(-999).values)
    if tf == "log":
        q = np.expm1(q)
    elif tf == "sqrt":
        q = np.maximum(q, 0) ** 2
    return np.clip(q, 0, None)


CANDS = {
    "lgb_ns_log": lambda a, b: lgbm(a, b, F_NS, "log"),
    "lgb_cyc_log": lambda a, b: lgbm(a, b, F_CYC, "log"),
    "lgb_full_log": lambda a, b: lgbm(a, b, FEATS, "log"),
    "lgb_ns_sqrt": lambda a, b: lgbm(a, b, F_NS, "sqrt"),
    "lgb_ns_log_deep": lambda a, b: lgbm(a, b, F_NS, "log", num_leaves=127, min_data_in_leaf=10),
    "lgb_ns_log_shal": lambda a, b: lgbm(a, b, F_NS, "log", num_leaves=31, min_data_in_leaf=40, nr=1400),
    "et_ns_log": lambda a, b: sk(ExtraTreesRegressor(500, min_samples_leaf=2, n_jobs=2, random_state=0), a, b, F_NS, "log"),
    "rf_ns_log": lambda a, b: sk(RandomForestRegressor(400, min_samples_leaf=2, n_jobs=2, random_state=0), a, b, F_NS, "log"),
    "hgb_ns_log": lambda a, b: sk(HistGradientBoostingRegressor(max_iter=600, learning_rate=0.06, max_leaf_nodes=63, min_samples_leaf=20, l2_regularization=1.0, random_state=0), a, b, F_NS, "log"),
}

store = {k: [] for k in CANDS}
ys = []
for a, b in FOLDS:
    dtr = trf[trf["dt"] < a]
    dva = trf[(trf["dt"] >= a) & (trf["dt"] < b)]
    ys.append(dva[TARGET].values)
    for k, fn in CANDS.items():
        store[k].append(fn(dtr, dva))
    print(f"fold {a.date()} n_tr={len(dtr)} n_va={len(dva)} " +
          " ".join(f"{k}={rmse(ys[-1], store[k][-1]):.0f}" for k in CANDS))

print("\n=== per-model mean RMSE over folds (and pooled) ===")
scores = {}
for k in CANDS:
    per = [rmse(y, p) for y, p in zip(ys, store[k])]
    pooled = rmse(np.concatenate(ys), np.concatenate(store[k]))
    scores[k] = (np.mean(per), pooled)
for k, v in sorted(scores.items(), key=lambda x: x[1][1]):
    print(f"{k:18s} mean={v[0]:7.2f} pooled={v[1]:7.2f}")

print("\n=== simple blends (pooled RMSE) ===")
import itertools
names = list(CANDS)
best = []
for r in [2, 3, 4]:
    for combo in itertools.combinations(names, r):
        p = np.mean([np.concatenate(store[k]) for k in combo], axis=0)
        best.append((rmse(np.concatenate(ys), p), combo))
best.sort()
for s, c in best[:15]:
    print(f"{s:7.2f}  {'+'.join(c)}")
np.save("/tmp/cv_ys.npy", np.concatenate(ys))

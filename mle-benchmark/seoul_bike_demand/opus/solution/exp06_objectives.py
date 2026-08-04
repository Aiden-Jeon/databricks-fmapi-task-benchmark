"""Compare LightGBM objectives (log-link count objectives model the mean -> better for RMSE)."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from common import load, build_features, FEATS, TARGET, rmse

tr_raw, te_raw = load()
tr, te = build_features(tr_raw, te_raw)
trf = tr[tr["func"] == 1].reset_index(drop=True)
SEASONAL = ["month", "day", "doy", "doy_sin", "doy_cos"]
F_NS = [f for f in FEATS if f not in SEASONAL]
F_CYC = [f for f in FEATS if f not in ("month", "day", "doy")]

starts = pd.date_range("2018-03-15", "2018-08-22", freq="21D")
FOLDS = [(s, s + pd.Timedelta(days=28)) for s in starts]

BASE = dict(learning_rate=0.04, num_leaves=63, min_data_in_leaf=20, feature_fraction=0.8,
            bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1,
            num_threads=2, seed=42)


def run(dtr, dva, feats, obj, tf=None, nr=900, **kw):
    y = dtr[TARGET].values.astype(float)
    p = dict(BASE); p["objective"] = obj; p.update(kw)
    yt = y if tf is None else ({"log": np.log1p(y), "sqrt": np.sqrt(y)}[tf])
    m = lgb.train(p, lgb.Dataset(dtr[feats], yt), num_boost_round=nr)
    q = m.predict(dva[feats])
    if tf == "log":
        q = np.expm1(q)
    elif tf == "sqrt":
        q = np.maximum(q, 0) ** 2
    return np.clip(q, 0, None)


CANDS = {
    "l2_log": lambda a, b: run(a, b, F_NS, "regression", "log"),
    "l2_sqrt": lambda a, b: run(a, b, F_NS, "regression", "sqrt"),
    "l2_raw": lambda a, b: run(a, b, F_NS, "regression", None),
    "poisson": lambda a, b: run(a, b, F_NS, "poisson", None),
    "tweedie11": lambda a, b: run(a, b, F_NS, "tweedie", None, tweedie_variance_power=1.1),
    "tweedie13": lambda a, b: run(a, b, F_NS, "tweedie", None, tweedie_variance_power=1.3),
    "tweedie15": lambda a, b: run(a, b, F_NS, "tweedie", None, tweedie_variance_power=1.5),
    "gamma": lambda a, b: run(a, b, F_NS, "gamma", None),
    "poisson_cyc": lambda a, b: run(a, b, F_CYC, "poisson", None),
    "poisson_full": lambda a, b: run(a, b, FEATS, "poisson", None),
    "tweedie13_cyc": lambda a, b: run(a, b, F_CYC, "tweedie", None, tweedie_variance_power=1.3),
    "huber_raw": lambda a, b: run(a, b, F_NS, "huber", None, alpha=200),
}

store = {k: [] for k in CANDS}
ys = []
for a, b in FOLDS:
    dtr = trf[trf["dt"] < a]
    dva = trf[(trf["dt"] >= a) & (trf["dt"] < b)]
    ys.append(dva[TARGET].values)
    for k, fn in CANDS.items():
        store[k].append(fn(dtr, dva))
    print(f"fold {a.date()} " + " ".join(f"{k}={rmse(ys[-1], store[k][-1]):.0f}" for k in CANDS), flush=True)

yall = np.concatenate(ys)
print("\n=== pooled RMSE ===")
sc = {k: (np.mean([rmse(y, p) for y, p in zip(ys, store[k])]), rmse(yall, np.concatenate(store[k]))) for k in CANDS}
for k, v in sorted(sc.items(), key=lambda x: x[1][1]):
    print(f"{k:15s} mean={v[0]:7.2f} pooled={v[1]:7.2f}")

import itertools, pickle
print("\n=== blends ===")
res = []
for r in [2, 3, 4]:
    for c in itertools.combinations(CANDS, r):
        res.append((rmse(yall, np.mean([np.concatenate(store[k]) for k in c], axis=0)), c))
res.sort()
for s, c in res[:15]:
    print(f"{s:7.2f} {'+'.join(c)}")
with open("/tmp/exp06_store.pkl", "wb") as f:
    pickle.dump({"store": {k: np.concatenate(v) for k, v in store.items()}, "y": yall}, f)

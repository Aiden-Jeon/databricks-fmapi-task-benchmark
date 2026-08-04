"""Tune n_rounds / lr for poisson, and pick blend weights (no-seasonal features only)."""
import itertools
import numpy as np
import pandas as pd
import lightgbm as lgb
from common import load, build_features, FEATS, TARGET, rmse

tr_raw, te_raw = load()
tr, te = build_features(tr_raw, te_raw)
trf = tr[tr["func"] == 1].reset_index(drop=True)
SEASONAL = ["month", "day", "doy", "doy_sin", "doy_cos"]
F_NS = [f for f in FEATS if f not in SEASONAL]

starts = pd.date_range("2018-03-15", "2018-08-22", freq="21D")
FOLDS = [(s, s + pd.Timedelta(days=28)) for s in starts]

BASE = dict(learning_rate=0.04, num_leaves=63, min_data_in_leaf=20, feature_fraction=0.8,
            bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=2, seed=42)


def run(dtr, dva, obj="poisson", tf=None, nr=900, feats=F_NS, **kw):
    y = dtr[TARGET].values.astype(float)
    p = dict(BASE); p["objective"] = obj; p.update(kw)
    yt = y if tf is None else {"log": np.log1p(y), "sqrt": np.sqrt(y)}[tf]
    m = lgb.train(p, lgb.Dataset(dtr[feats], yt), num_boost_round=nr)
    q = m.predict(dva[feats])
    if tf == "sqrt":
        q = np.maximum(q, 0) ** 2
    elif tf == "log":
        q = np.expm1(q)
    return np.clip(q, 0, None)


CANDS = {
    "po_600": lambda a, b: run(a, b, nr=600),
    "po_900": lambda a, b: run(a, b, nr=900),
    "po_1500": lambda a, b: run(a, b, nr=1500),
    "po_lr02_2000": lambda a, b: run(a, b, nr=2000, learning_rate=0.02),
    "po_leaf31": lambda a, b: run(a, b, nr=1400, num_leaves=31, min_data_in_leaf=30),
    "po_leaf127": lambda a, b: run(a, b, nr=700, num_leaves=127, min_data_in_leaf=15),
    "po_ff5": lambda a, b: run(a, b, nr=1200, feature_fraction=0.5),
    "tw13": lambda a, b: run(a, b, obj="tweedie", nr=900, tweedie_variance_power=1.3),
    "sqrt": lambda a, b: run(a, b, obj="regression", tf="sqrt", nr=900),
    "po_full": lambda a, b: run(a, b, nr=900, feats=FEATS),
}

store = {k: [] for k in CANDS}
ys, fold_rmse = [], {k: [] for k in CANDS}
for a, b in FOLDS:
    dtr = trf[trf["dt"] < a]
    dva = trf[(trf["dt"] >= a) & (trf["dt"] < b)]
    ys.append(dva[TARGET].values)
    for k, fn in CANDS.items():
        p = fn(dtr, dva)
        store[k].append(p)
        fold_rmse[k].append(rmse(ys[-1], p))
    print(f"fold {a.date()} " + " ".join(f"{k}={fold_rmse[k][-1]:.0f}" for k in CANDS), flush=True)

yall = np.concatenate(ys)
print("\n=== pooled / mean / worst fold ===")
rows = [(k, np.mean(fold_rmse[k]), rmse(yall, np.concatenate(store[k])), max(fold_rmse[k])) for k in CANDS]
for k, m_, p_, w_ in sorted(rows, key=lambda r: r[2]):
    print(f"{k:14s} mean={m_:7.2f} pooled={p_:7.2f} worst={w_:7.2f}")

print("\n=== equal-weight blends (pooled, worst-fold) ===")
res = []
for r in range(2, 5):
    for c in itertools.combinations(CANDS, r):
        P = [np.mean([store[k][i] for k in c], axis=0) for i in range(len(FOLDS))]
        res.append((rmse(yall, np.concatenate(P)), max(rmse(y, p) for y, p in zip(ys, P)), c))
res.sort()
for s, w, c in res[:15]:
    print(f"pooled={s:7.2f} worst={w:7.2f}  {'+'.join(c)}")

"""Final blend selection around the regularized poisson config, incl. seed averaging."""
import itertools
import numpy as np
import pandas as pd
import lightgbm as lgb
from common import load, build_features, FEATS, TARGET, rmse

tr_raw, te_raw = load()
tr, te = build_features(tr_raw, te_raw)
trf = tr[tr["func"] == 1].reset_index(drop=True)
SEASONAL = ["month", "day", "doy", "doy_sin", "doy_cos"]
F = [f for f in FEATS if f not in SEASONAL]

starts = pd.date_range("2018-03-15", "2018-08-22", freq="21D")
FOLDS = [(s, s + pd.Timedelta(days=28)) for s in starts]

BASE = dict(learning_rate=0.04, num_leaves=31, min_data_in_leaf=30, feature_fraction=0.8,
            bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=2)


def run(dtr, dva, tf=None, nr=1400, seeds=(42,), **kw):
    y = dtr[TARGET].values.astype(float)
    yt = y if tf is None else {"sqrt": np.sqrt(y), "log": np.log1p(y)}[tf]
    acc = np.zeros(len(dva))
    for s in seeds:
        p = dict(BASE); p.update(kw)
        p.update(seed=s, bagging_seed=s + 1, feature_fraction_seed=s + 2)
        m = lgb.train(p, lgb.Dataset(dtr[F], yt), num_boost_round=nr)
        q = m.predict(dva[F])
        if tf == "sqrt":
            q = np.maximum(q, 0) ** 2
        elif tf == "log":
            q = np.expm1(q)
        acc += np.clip(q, 0, None)
    return acc / len(seeds)


S3 = (42, 7, 2024)
CANDS = {
    "po1": lambda a, b: run(a, b, objective="poisson"),
    "po3s": lambda a, b: run(a, b, objective="poisson", seeds=S3),
    "po3s_ff5": lambda a, b: run(a, b, objective="poisson", seeds=S3, feature_fraction=0.5, nr=1800),
    "po3s_reg": lambda a, b: run(a, b, objective="poisson", seeds=S3, num_leaves=15,
                                 min_data_in_leaf=50, nr=2500),
    "tw3s": lambda a, b: run(a, b, objective="tweedie", tweedie_variance_power=1.3, seeds=S3),
    "sq3s": lambda a, b: run(a, b, objective="regression", tf="sqrt", seeds=S3),
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
print("\n=== singles ===")
for k in sorted(CANDS, key=lambda k: rmse(yall, np.concatenate(store[k]))):
    fr = [rmse(y, p) for y, p in zip(ys, store[k])]
    print(f"{k:10s} pooled={rmse(yall, np.concatenate(store[k])):7.2f} mean={np.mean(fr):7.2f} worst={max(fr):7.2f}")

print("\n=== blends ===")
res = []
for r in range(2, len(CANDS) + 1):
    for c in itertools.combinations(CANDS, r):
        P = [np.mean([store[k][i] for k in c], axis=0) for i in range(len(FOLDS))]
        fr = [rmse(y, p) for y, p in zip(ys, P)]
        res.append((rmse(yall, np.concatenate(P)), np.mean(fr), max(fr), c))
res.sort()
for s, m_, w, c in res[:12]:
    print(f"pooled={s:7.2f} mean={m_:7.2f} worst={w:7.2f}  {'+'.join(c)}")

# weighted blend: poisson-heavy
print("\n=== weighted (po3s, tw3s, sq3s) ===")
for wts in [(1, 0, 0), (.6, .2, .2), (.5, .25, .25), (.7, .15, .15), (.34, .33, .33), (.6, .4, 0), (.6, 0, .4)]:
    P = [wts[0] * store["po3s"][i] + wts[1] * store["tw3s"][i] + wts[2] * store["sq3s"][i]
         for i in range(len(FOLDS))]
    fr = [rmse(y, p) for y, p in zip(ys, P)]
    print(f"{wts} pooled={rmse(yall, np.concatenate(P)):7.2f} mean={np.mean(fr):7.2f} worst={max(fr):7.2f}")

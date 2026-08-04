"""Quick baseline + chronological CV comparison of target transforms."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from common import load, build_features, FEATS, TARGET, rmse

tr_raw, te_raw = load()
tr, te = build_features(tr_raw, te_raw)

# only functioning days are learnable; non-functioning => 0
trf = tr[tr["func"] == 1].reset_index(drop=True)

FOLDS = [("2018-06-20", "2018-07-19"), ("2018-07-20", "2018-08-18"), ("2018-08-19", "2018-09-19")]

PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=42, num_threads=2)


def fit_predict(train_df, valid_X, transform, n=1200):
    y = train_df[TARGET].values.astype(float)
    if transform == "log":
        yt = np.log1p(y)
    elif transform == "sqrt":
        yt = np.sqrt(y)
    else:
        yt = y
    m = lgb.train(PARAMS, lgb.Dataset(train_df[FEATS], yt), num_boost_round=n)
    p = m.predict(valid_X)
    if transform == "log":
        p = np.expm1(p)
    elif transform == "sqrt":
        p = np.maximum(p, 0) ** 2
    return np.clip(p, 0, None), m


for transform in ["raw", "sqrt", "log"]:
    scores = []
    for a, b in FOLDS:
        m_tr = trf["dt"] < a
        m_va = (trf["dt"] >= a) & (trf["dt"] < b)
        p, _ = fit_predict(trf[m_tr], trf.loc[m_va, FEATS], transform)
        scores.append(rmse(trf.loc[m_va, TARGET], p))
    print(f"{transform:5s} folds={[round(s,1) for s in scores]} mean={np.mean(scores):.2f}")

# (submission writing removed: use solution/train_predict.py as the entry point)

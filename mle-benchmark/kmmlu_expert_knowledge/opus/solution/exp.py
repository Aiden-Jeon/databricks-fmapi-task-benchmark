"""Fast model-selection experiments on pre-computed hand-crafted features.

Compares point-wise (option-level binary) vs list-wise (question-level 4-class)
formulations and a few hyper-parameter settings with repeated 5-fold CV.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import build_features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tr = pd.read_csv(os.path.join(ROOT, "train.csv"))
n = len(tr)
lab = tr["label"].values - 1

F, _ = build_features(tr)
cols = [c for c in F.columns if c not in ("qidx", "opt")]
X = F[cols].values
n_feat = len(cols)


def rel_block(X):
    idx = [cols.index(c) for c in ["len", "sim_q", "sim_oth_mean", "nwords", "o_digits"]]
    out = []
    for c in idx:
        v = X[:, c].reshape(-1, 4)
        out.append((v - v.mean(1, keepdims=True)).reshape(-1))
        out.append(np.argsort(np.argsort(v, 1), 1).astype(float).reshape(-1))
    return np.stack(out, 1)


X = np.hstack([X, rel_block(X)])
y = np.zeros(n * 4)
y[np.arange(n) * 4 + lab] = 1

# list-wise design: one row per question, features of all four options concatenated
Xq = X.reshape(n, -1)


def cv_pointwise(make_model, seeds=(0, 1)):
    accs = []
    for sd in seeds:
        kf = KFold(5, shuffle=True, random_state=sd)
        oof = np.zeros(n * 4)
        for a, b in kf.split(np.arange(n)):
            ra = (a[:, None] * 4 + np.arange(4)).ravel()
            rb = (b[:, None] * 4 + np.arange(4)).ravel()
            m = make_model()
            m.fit(X[ra], y[ra])
            oof[rb] = m.predict_proba(X[rb])[:, 1]
        accs.append((oof.reshape(n, 4).argmax(1) == lab).mean())
    return float(np.mean(accs)), accs


def cv_listwise(make_model, seeds=(0, 1)):
    accs = []
    for sd in seeds:
        kf = KFold(5, shuffle=True, random_state=sd)
        oof = np.zeros((n, 4))
        for a, b in kf.split(np.arange(n)):
            m = make_model()
            m.fit(Xq[a], lab[a])
            P = m.predict_proba(Xq[b])
            oof[b] = P
        accs.append((oof.argmax(1) == lab).mean())
    return float(np.mean(accs)), accs


if __name__ == "__main__":
    print("prior baseline:", round(np.bincount(lab).max() / n, 4))

    variants = {
        "pw_hgb_lr04_lf15": lambda: HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.04, max_leaf_nodes=15,
            min_samples_leaf=40, l2_regularization=1.0, random_state=0),
        "pw_hgb_lr05_lf31_i200": lambda: HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=31,
            min_samples_leaf=20, l2_regularization=0.0, random_state=0),
        "pw_hgb_shallow": lambda: HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.05, max_leaf_nodes=6,
            min_samples_leaf=60, l2_regularization=2.0, random_state=0),
        "pw_hgb_verylight": lambda: HistGradientBoostingClassifier(
            max_iter=120, learning_rate=0.06, max_leaf_nodes=4,
            min_samples_leaf=100, l2_regularization=5.0, random_state=0),
    }
    for name, mk in variants.items():
        m, a = cv_pointwise(mk)
        print(f"{name:24s} {m:.4f}  {np.round(a,4)}")

    lvariants = {
        "lw_hgb_lf15": lambda: HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
            min_samples_leaf=30, l2_regularization=1.0, random_state=0),
        "lw_hgb_shallow": lambda: HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=6,
            min_samples_leaf=60, l2_regularization=2.0, random_state=0),
        "lw_rf": lambda: RandomForestClassifier(
            n_estimators=500, min_samples_leaf=10, max_features="sqrt",
            n_jobs=-1, random_state=0),
    }
    for name, mk in lvariants.items():
        m, a = cv_listwise(mk)
        print(f"{name:24s} {m:.4f}  {np.round(a,4)}")

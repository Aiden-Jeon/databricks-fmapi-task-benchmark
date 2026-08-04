"""Blend cached base-model OOF/test predictions.

Two strategies, both scored by 5-fold CV on the OOF matrix:
  1. weighted geometric mean (weights >= 0, sum 1) optimised with SLSQP
  2. logistic-regression stacker on log-probabilities
The better one (by CV) is written to outputs/submission.csv.
"""
import glob
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

CLASSES = ["EAP", "HPL", "MWS"]
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "oof_cache")
EPS = 1e-6
SEED = 42
N_FOLDS = 5


def load():
    names, oofs, tests = [], [], []
    for p in sorted(glob.glob(os.path.join(CACHE, "*.npz"))):
        d = np.load(p)
        names.append(os.path.basename(p)[:-4])
        oofs.append(np.clip(d["oof"], EPS, 1))
        tests.append(np.clip(d["test"], EPS, 1))
    oofs = [o / o.sum(1, keepdims=True) for o in oofs]
    tests = [t / t.sum(1, keepdims=True) for t in tests]
    return names, np.stack(oofs), np.stack(tests)


def geo_blend(P, w):
    """P: (m, n, 3) probabilities, w: (m,) weights -> normalised geometric mean."""
    L = np.tensordot(w, np.log(P), axes=(0, 0))
    L -= L.max(1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(1, keepdims=True)


def fit_weights(P, y):
    m = P.shape[0]

    def obj(w):
        return log_loss(y, geo_blend(P, w), labels=CLASSES)

    best = None
    for start in (np.ones(m) / m, np.full(m, 1.0 / max(m, 1))):
        r = minimize(obj, start, method="SLSQP",
                     bounds=[(0, 5)] * m,
                     constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                     options={"maxiter": 300, "ftol": 1e-9})
        if best is None or r.fun < best.fun:
            best = r
    return np.clip(best.x, 0, None)


def cv_eval_geo(P, y, folds):
    oof = np.zeros((P.shape[1], 3))
    for tr, va in folds:
        w = fit_weights(P[:, tr], y[tr])
        oof[va] = geo_blend(P[:, va], w)
    return log_loss(y, oof, labels=CLASSES), oof


def stack_features(P):
    return np.log(P).transpose(1, 0, 2).reshape(P.shape[1], -1)


def cv_eval_stack(P, y, folds, C=1.0):
    X = stack_features(P)
    oof = np.zeros((len(y), 3))
    for tr, va in folds:
        clf = LogisticRegression(C=C, max_iter=4000)
        clf.fit(X[tr], y[tr])
        cls = list(clf.classes_)
        oof[va] = clf.predict_proba(X[va])[:, [cls.index(c) for c in CLASSES]]
    return log_loss(y, oof, labels=CLASSES), oof


def greedy_select(names, P, y, folds):
    """Forward selection of models maximising CV of the geometric blend."""
    singles = [(log_loss(y, P[i], labels=CLASSES), i) for i in range(len(names))]
    singles.sort()
    chosen = [singles[0][1]]
    best, _ = cv_eval_geo(P[chosen], y, folds)
    print(f"  start {names[chosen[0]]}: {best:.5f}")
    improved = True
    while improved:
        improved = False
        cand_best, cand_i = best, None
        for i in range(len(names)):
            if i in chosen:
                continue
            sc, _ = cv_eval_geo(P[chosen + [i]], y, folds)
            if sc < cand_best - 1e-5:
                cand_best, cand_i = sc, i
        if cand_i is not None:
            chosen.append(cand_i)
            best = cand_best
            improved = True
            print(f"  + {names[cand_i]:14s} -> {best:.5f}")
    return chosen, best


def main():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    y = train["author"].values
    names, P, T = load()
    print(f"{len(names)} base models")
    for i, n in enumerate(names):
        print(f"  {n:14s} {log_loss(y, P[i], labels=CLASSES):.5f}")

    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=7).split(y, y))

    print("greedy geometric blend:")
    chosen, geo_cv = greedy_select(names, P, y, folds)
    print("  chosen:", [names[i] for i in chosen])

    results = [("geo", geo_cv, chosen, None)]
    for C in (0.1, 0.3, 1.0, 3.0):
        sc, _ = cv_eval_stack(P, y, folds, C=C)
        print(f"stack all  C={C}: {sc:.5f}")
        results.append((f"stack_all_C{C}", sc, list(range(len(names))), C))
    for C in (0.3, 1.0, 3.0):
        sc, _ = cv_eval_stack(P[chosen], y, folds, C=C)
        print(f"stack sel  C={C}: {sc:.5f}")
        results.append((f"stack_sel_C{C}", sc, chosen, C))

    results.sort(key=lambda r: r[1])
    kind, score, idx, C = results[0]
    print(f"\nBEST: {kind}  CV={score:.5f}  models={[names[i] for i in idx]}")

    if kind == "geo":
        w = fit_weights(P[idx], y)
        print("weights:", {names[i]: round(float(x), 4) for i, x in zip(idx, w)})
        pred = geo_blend(T[idx], w)
    else:
        Xtr = stack_features(P[idx])
        Xte = stack_features(T[idx])
        clf = LogisticRegression(C=C, max_iter=4000).fit(Xtr, y)
        cls = list(clf.classes_)
        pred = clf.predict_proba(Xte)[:, [cls.index(c) for c in CLASSES]]

    pred = np.clip(pred, 1e-7, 1)
    pred /= pred.sum(1, keepdims=True)
    sub = pd.DataFrame(pred, columns=CLASSES)
    sub.insert(0, "id", test["id"].values)
    os.makedirs("outputs", exist_ok=True)
    sub.to_csv("outputs/submission.csv", index=False)
    print("wrote outputs/submission.csv", sub.shape)


if __name__ == "__main__":
    main()

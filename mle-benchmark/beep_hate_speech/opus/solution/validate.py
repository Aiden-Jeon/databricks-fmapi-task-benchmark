#!/usr/bin/env python
"""Honest validation of the final stacking + calibration stage.

Reuses the exact feature/model pool from run.py. The base-model
out-of-fold matrix is built once, then the level-1 stacker + prior
calibration are evaluated with repeated (5 seeds x 5 folds) CV.
"""
import os, sys, time
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run import (BASE_POOL, CLASSES, C2I, NFOLD, SEED, build_features,
                 fit_prior_weights, proba)

CACHE = "solution/_oof_cache.npz"


def build_oof():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    y = train.label.map(C2I).values
    ntr = len(train)
    blocks = build_features(train, test)
    fds = list(StratifiedKFold(NFOLD, shuffle=True,
                               random_state=SEED).split(np.zeros(ntr), y))
    cols = []
    for tag, names, factory in BASE_POOL:
        M = hstack([blocks[n] for n in names]).tocsr()[:ntr]
        P = np.zeros((ntr, 3))
        for tri, vai in fds:
            P[vai] = proba(factory().fit(M[tri], y[tri]), M[vai])
        print(f"  {tag:24s} {f1_score(y, P.argmax(1), average='macro'):.4f}",
              flush=True)
        cols.append(np.log(P))
    Sx = np.concatenate(cols, axis=1)
    np.savez_compressed(CACHE, Sx=Sx, y=y)
    return Sx, y


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        Sx, y = d["Sx"], d["y"]
    else:
        print("base models (oof macro F1):")
        Sx, y = build_oof()

    def mk():
        return LogisticRegression(C=0.03, max_iter=3000,
                                  class_weight="balanced", random_state=SEED)

    for calib in (False, True):
        scores = []
        for s in (0, 1, 2, 7, 42):
            skf = StratifiedKFold(5, shuffle=True, random_state=s)
            pred = np.zeros(len(y), dtype=int)
            for tri, vai in skf.split(Sx, y):
                m = mk().fit(Sx[tri], y[tri])
                if calib:
                    lw, _ = fit_prior_weights(m.predict_proba(Sx[tri]), y[tri])
                    pred[vai] = (np.log(np.clip(m.predict_proba(Sx[vai]),
                                                1e-9, None)) + lw).argmax(1)
                else:
                    pred[vai] = m.predict(Sx[vai])
            scores.append(f1_score(y, pred, average="macro"))
        tag = "stack + prior-calib" if calib else "stack only         "
        print(f"{tag}  macroF1 = {np.mean(scores):.4f} +/- {np.std(scores):.4f}"
              f"   {np.round(scores, 4)}")

    # single best base model for reference
    best = max(f1_score(y, Sx[:, 3 * i:3 * i + 3].argmax(1), average="macro")
               for i in range(Sx.shape[1] // 3))
    print(f"best single base model            = {best:.4f}")


if __name__ == "__main__":
    main()

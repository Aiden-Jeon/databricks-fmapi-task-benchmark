"""Experiment: test stronger models on the improved feature set quickly.

We add a character-bigram-only model, an SGDClassifier with hinge loss,
and a passive-aggressive classifier.  We run a single fast CV split to
gauge whether they beat the current LinearSVC ensemble (~68.75%).
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import build_pair_features

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def quick_cv(Xtr, y, name, clf, n_splits=3, seed=42, proba=False):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for fold, (tr_idx, va_idx) in enumerate(skf.split(Xtr, y)):
        clf.fit(Xtr[tr_idx], y[tr_idx])
        if proba:
            oof[va_idx] = clf.predict_proba(Xtr[va_idx])[:, 1]
        else:
            oof[va_idx] = clf.decision_function(Xtr[va_idx])
        print(f"  {name} fold {fold} acc={accuracy_score(y[va_idx], (oof[va_idx]>0).astype(int)):.4f}", flush=True)
    acc = accuracy_score(y, (oof > 0).astype(int))
    print(f"{name} OOF acc={acc:.4f}", flush=True)
    return oof, acc


def main():
    t0 = time.time()
    train_df = pd.read_csv(os.path.join(ROOT, "train.csv"))
    test_df = pd.read_csv(os.path.join(ROOT, "test.csv"))
    y = train_df["label"].values
    print("building features...", flush=True)
    Xtr, Xte, _ = build_pair_features(train_df, test_df)
    print(f"combined: {Xtr.shape} {Xte.shape}", flush=True)

    models = {
        "LR_c4": (LogisticRegression(C=4.0, max_iter=3000, solver="liblinear"), True),
        "LR_c1": (LogisticRegression(C=1.0, max_iter=3000, solver="liblinear"), True),
        "SVC_c1": (LinearSVC(C=1.0, max_iter=5000, dual="auto"), False),
        "SVC_c05": (LinearSVC(C=0.5, max_iter=5000, dual="auto"), False),
        "SGD_hinge": (SGDClassifier(loss="hinge", alpha=1e-5, max_iter=1000, random_state=42, n_jobs=-1), False),
        "SGD_log": (SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=1000, random_state=42, n_jobs=-1), True),
        "PA": (PassiveAggressiveClassifier(C=1.0, max_iter=1000, random_state=42, n_jobs=-1), False),
        "Ridge": (RidgeClassifier(alpha=1.0), False),
    }
    oofs = {}
    for name, (clf, proba) in models.items():
        oof, acc = quick_cv(Xtr, y, name, clf, n_splits=3, proba=proba)
        oofs[name] = (oof, acc, proba)
    print(f"elapsed {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Threshold tuning experiment for Korean UnSmile.

Goal: improve macro F1 over the default baseline by tuning:
  - per-class thresholds (joint macro-F1 grid)
  - the clean-handling rule (if any hate set -> clean=0; if none -> clean=1)
  - C hyperparameter
"""

import os
import re
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

TASK_DIR = "/tmp/kmle/M3_t24_kor_unsmile_full_20260804_033857/task"
TRAIN_CSV = os.path.join(TASK_DIR, "train.csv")
N_CLASSES = 10


def normalize(text):
    text = str(text).lower()
    text = text.replace("ㆍ", " ").replace("·", " ")
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    return text


def macro_f1(y_true, y_pred):
    f1s = []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        if yt.sum() == 0:
            continue
        tp = int((yt & yp).sum())
        fp = int((~yt.astype(bool) & yp.astype(bool)).sum())
        fn = int((yt.astype(bool) & ~yp.astype(bool)).sum())
        if tp == 0:
            f1s.append(0.0)
        else:
            p = tp / (tp + fp)
            r = tp / (tp + fn)
            f1s.append(2 * p * r / (p + r))
    return float(np.mean(f1s)), f1s


def apply_clean_rule(pred):
    pred = pred.copy()
    pred[:, 9] = pred[:, 9] * (pred[:, :9].sum(axis=1) == 0).astype(int)
    no_label = pred.sum(axis=1) == 0
    pred[no_label, 9] = 1
    return pred


def run_C(C):
    train = pd.read_csv(TRAIN_CSV, dtype={"labels": str})
    train_text = train["sentence"].map(normalize).tolist()
    y = np.array([list(s) for s in train["labels"]]).astype(int)

    word_vec = TfidfVectorizer(analyzer="word", token_pattern=r"\S+",
                              ngram_range=(1, 2), min_df=2,
                              sublinear_tf=True, max_features=50000)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                              min_df=2, sublinear_tf=True, max_features=60000)
    Xt_word = word_vec.fit_transform(train_text)
    Xt_char = char_vec.fit_transform(train_text)
    Xt = hstack([Xt_word, Xt_char]).tocsr()

    primary = y.argmax(axis=1)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros_like(y, dtype=float)
    for tr, va in skf.split(Xt, primary):
        for c in range(N_CLASSES):
            if y[tr, c].sum() < 2:
                continue
            clf = LogisticRegression(C=C, max_iter=2000, solver="liblinear",
                                     class_weight="balanced")
            clf.fit(Xt[tr], y[tr, c])
            oof[va, c] = clf.predict_proba(Xt[va])[:, 1]

    # Per-class threshold tune
    thr = np.full(N_CLASSES, 0.5)
    for c in range(N_CLASSES):
        if y[:, c].sum() == 0:
            continue
        best_t, best_f = 0.5, -1
        for t in np.linspace(0.05, 0.95, 91):
            p = (oof[:, c] >= t).astype(int)
            tp = int((p & y[:, c]).sum())
            fp = int((p & ~y[:, c].astype(bool)).sum())
            fn = int((~p.astype(bool) & y[:, c].astype(bool)).sum())
            f = 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)
            if f > best_f:
                best_f, best_t = f, t
        thr[c] = best_t
    pred = (oof >= thr).astype(int)
    pred = apply_clean_rule(pred)
    f1, f1s = macro_f1(y, pred)
    print(f"C={C} thr={thr.round(2)} OOF_f1={f1:.4f}")
    print(f"  per-class: {[f'{x:.3f}' for x in f1s]}")
    return f1, oof, thr


if __name__ == "__main__":
    for C in [1.0, 2.0, 4.0, 8.0]:
        run_C(C)

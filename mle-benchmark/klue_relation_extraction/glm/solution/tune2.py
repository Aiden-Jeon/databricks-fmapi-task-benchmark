#!/usr/bin/env python3
"""Tuning v2: best SVC (none) combined with balanced LR; vary C and weights."""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from scipy.sparse import hstack

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solution import build_features, word_tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.dirname(HERE)
TRAIN_PATH = os.path.join(TASK_DIR, "train.csv")

def main():
    train = pd.read_csv(TRAIN_PATH)
    y = train["label"].values
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_text = build_features(train)
    word_vec = TfidfVectorizer(analyzer="word", tokenizer=word_tokenizer,
        ngram_range=(1,2), min_df=2, max_df=0.95, sublinear_tf=True,
        strip_accents=None, lowercase=False)
    Xw = word_vec.fit_transform(X_text)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,5),
        min_df=3, max_df=0.95, sublinear_tf=True, lowercase=False)
    Xc = char_vec.fit_transform(X_text)
    X = hstack([Xw, Xc]).tocsr()
    print("X shape", X.shape, flush=True)

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    folds = list(skf.split(X, y_enc))

    # Precompute calibrated SVC probabilities per fold for several SVC settings
    svc_settings = [
        ("svc_none_C1", LinearSVC(C=1.0, class_weight=None, random_state=42, max_iter=3000)),
        ("svc_bal_C1", LinearSVC(C=1.0, class_weight="balanced", random_state=42, max_iter=3000)),
    ]
    lr_settings = [
        ("lr_bal_C4", LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced",
            n_jobs=-1, random_state=42, solver="liblinear")),
        ("lr_none_C10", LogisticRegression(C=10.0, max_iter=2000, class_weight=None,
            n_jobs=-1, random_state=42, solver="liblinear")),
        ("lr_bal_C1", LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced",
            n_jobs=-1, random_state=42, solver="liblinear")),
    ]

    svc_proba = {n: [] for n,_ in svc_settings}
    lr_proba = {n: [] for n,_ in lr_settings}
    y_va_all = []
    for tr, va in folds:
        y_va_all.append(y_enc[va])
        for name, svc in svc_settings:
            cal = CalibratedClassifierCV(clone(svc), cv=3, method="sigmoid")
            cal.fit(X[tr], y_enc[tr])
            svc_proba[name].append(cal.predict_proba(X[va]))
        for name, lr in lr_settings:
            m = clone(lr)
            m.fit(X[tr], y_enc[tr])
            lr_proba[name].append(m.predict_proba(X[va]))

    y_va = np.concatenate(y_va_all)
    for sname,_ in svc_settings:
        ps = np.concatenate(svc_proba[sname])
        pred = np.argmax(ps, axis=1)
        print(f"{sname}: {accuracy_score(y_va, pred):.4f}", flush=True)
    for lname,_ in lr_settings:
        ps = np.concatenate(lr_proba[lname])
        pred = np.argmax(ps, axis=1)
        print(f"{lname}: {accuracy_score(y_va, pred):.4f}", flush=True)

    print("\n--- Ensembles (concat folds) ---", flush=True)
    for sname,_ in svc_settings:
        ps = np.concatenate(svc_proba[sname])
        for lname,_ in lr_settings:
            pl = np.concatenate(lr_proba[lname])
            for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
                proba = w*ps + (1-w)*pl
                pred = np.argmax(proba, axis=1)
                acc = accuracy_score(y_va, pred)
                print(f"  svc={sname} lr={lname} w_svc={w}: {acc:.4f}", flush=True)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Tuning script: try different class_weight and C settings for accuracy."""
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

    configs = [
        ("lr_balanced_C4", LogisticRegression(C=4.0, max_iter=2000,
            class_weight="balanced", n_jobs=-1, random_state=42, solver="liblinear"), "lr"),
        ("lr_none_C4", LogisticRegression(C=4.0, max_iter=2000,
            class_weight=None, n_jobs=-1, random_state=42, solver="liblinear"), "lr"),
        ("lr_none_C1", LogisticRegression(C=1.0, max_iter=2000,
            class_weight=None, n_jobs=-1, random_state=42, solver="liblinear"), "lr"),
        ("lr_none_C10", LogisticRegression(C=10.0, max_iter=2000,
            class_weight=None, n_jobs=-1, random_state=42, solver="liblinear"), "lr"),
    ]

    results = {}
    for name, model, kind in configs:
        accs = []
        for fold, (tr, va) in enumerate(folds):
            model_cloned = sklearn_clone(model)
            model_cloned.fit(X[tr], y_enc[tr])
            if kind == "lr":
                proba = model_cloned.predict_proba(X[va])
            else:
                proba = model_cloned.predict_proba(X[va])
            pred = np.argmax(proba, axis=1)
            acc = accuracy_score(y_enc[va], pred)
            accs.append(acc)
        mean_acc = np.mean(accs)
        results[name] = mean_acc
        print(f"{name}: {mean_acc:.4f}  folds={[f'{a:.4f}' for a in accs]}", flush=True)

    # SVC variations
    svc_configs = [
        ("svc_balanced_C1", LinearSVC(C=1.0, class_weight="balanced", random_state=42, max_iter=3000)),
        ("svc_none_C1", LinearSVC(C=1.0, class_weight=None, random_state=42, max_iter=3000)),
        ("svc_none_C05", LinearSVC(C=0.5, class_weight=None, random_state=42, max_iter=3000)),
    ]
    for name, svc in svc_configs:
        accs = []
        for fold, (tr, va) in enumerate(folds):
            cal = CalibratedClassifierCV(svc, cv=3, method="sigmoid")
            cal.fit(X[tr], y_enc[tr])
            proba = cal.predict_proba(X[va])
            pred = np.argmax(proba, axis=1)
            acc = accuracy_score(y_enc[va], pred)
            accs.append(acc)
        mean_acc = np.mean(accs)
        results[name] = mean_acc
        print(f"{name}: {mean_acc:.4f}  folds={[f'{a:.4f}' for a in accs]}", flush=True)

    # Ensembles of best LR + best SVC
    print("\n--- Ensembles ---", flush=True)
    lr_none_C4 = LogisticRegression(C=4.0, max_iter=2000, class_weight=None,
        n_jobs=-1, random_state=42, solver="liblinear")
    svc_none_C1 = LinearSVC(C=1.0, class_weight=None, random_state=42, max_iter=3000)
    accs = []
    for fold, (tr, va) in enumerate(folds):
        lr_none_C4.fit(X[tr], y_enc[tr])
        p_lr = lr_none_C4.predict_proba(X[va])
        cal = CalibratedClassifierCV(svc_none_C1, cv=3, method="sigmoid")
        cal.fit(X[tr], y_enc[tr])
        p_svc = cal.predict_proba(X[va])
        for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
            proba = w*p_lr + (1-w)*p_svc
            pred = np.argmax(proba, axis=1)
            acc = accuracy_score(y_enc[va], pred)
            print(f"  fold{fold} w_lr={w}: {acc:.4f}", flush=True)

def sklearn_clone(model):
    from sklearn.base import clone
    return clone(model)

if __name__ == "__main__":
    main()

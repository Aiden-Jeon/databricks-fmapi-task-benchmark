#!/usr/bin/env python3
"""Quick CV estimate for the KLUE-RE TF-IDF model."""
import os, re, sys, warnings
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

    # Fit vectorizers on full data (leakage is minor for vectorizer; but to be clean
    # we could fit per fold. For speed, fit once and split indices.)
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
    accs_lr, accs_svc, accs_ens = [], [], []
    for fold, (tr, va) in enumerate(skf.split(X, y_enc)):
        lr = LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced",
            n_jobs=-1, random_state=42, solver="liblinear")
        lr.fit(X[tr], y_enc[tr])
        p_lr = lr.predict_proba(X[va])
        svc = LinearSVC(C=1.0, class_weight="balanced", random_state=42, max_iter=3000)
        cal = CalibratedClassifierCV(svc, cv=3, method="sigmoid")
        cal.fit(X[tr], y_enc[tr])
        p_svc = cal.predict_proba(X[va])
        proba = 0.5*p_lr + 0.5*p_svc
        pred = np.argmax(proba, axis=1)
        a_lr = accuracy_score(y_enc[va], np.argmax(p_lr, axis=1))
        a_svc = accuracy_score(y_enc[va], np.argmax(p_svc, axis=1))
        a_ens = accuracy_score(y_enc[va], pred)
        accs_lr.append(a_lr); accs_svc.append(a_svc); accs_ens.append(a_ens)
        print(f"fold {fold}: lr={a_lr:.4f} svc={a_svc:.4f} ens={a_ens:.4f}", flush=True)
    print(f"mean: lr={np.mean(accs_lr):.4f} svc={np.mean(accs_svc):.4f} ens={np.mean(accs_ens):.4f}", flush=True)

if __name__ == "__main__":
    main()

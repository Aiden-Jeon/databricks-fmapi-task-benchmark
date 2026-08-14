# -*- coding: utf-8 -*-
"""홀드아웃 기반 하이퍼파라미터/앙상블 실험 스크립트 (모델 선택용)."""
import os
import re
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def clean_text(s):
    if not isinstance(s, str):
        return ""
    s = s.replace("\n", " ").replace("\t", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build(train_texts, val_texts, word_cfg, char_cfg):
    wv = TfidfVectorizer(analyzer="word", sublinear_tf=True,
                         token_pattern=r"(?u)\S+", **word_cfg)
    cv = TfidfVectorizer(analyzer="char_wb", sublinear_tf=True, **char_cfg)
    Xtr = sparse.hstack([wv.fit_transform(train_texts),
                         cv.fit_transform(train_texts)]).tocsr()
    Xva = sparse.hstack([wv.transform(val_texts),
                         cv.transform(val_texts)]).tocsr()
    return Xtr, Xva


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    train = pd.read_csv(os.path.join(BASE_DIR, "train.csv"), dtype={"id": str})
    train["document"] = train["document"].fillna("").map(clean_text)
    y = train["label"].astype(int).values
    tr, va = train_test_split(np.arange(len(train)), test_size=0.1,
                              random_state=42, stratify=y)
    txt_tr = train["document"].iloc[tr]
    txt_va = train["document"].iloc[va]
    ytr, yva = y[tr], y[va]

    feats = {
        "base": (dict(ngram_range=(1, 2), min_df=2, max_features=400000),
                 dict(ngram_range=(2, 5), min_df=3, max_features=400000)),
        "w3":   (dict(ngram_range=(1, 3), min_df=2, max_features=600000),
                 dict(ngram_range=(2, 5), min_df=3, max_features=400000)),
        "c6":   (dict(ngram_range=(1, 2), min_df=2, max_features=400000),
                 dict(ngram_range=(2, 6), min_df=3, max_features=600000)),
    }

    for name, (wc, cc) in feats.items():
        t0 = time.time()
        Xtr, Xva = build(txt_tr, txt_va, wc, cc)
        print(f"\n== features: {name} shape={Xtr.shape} ({time.time()-t0:.0f}s)")
        probs = {}
        for C in [2.0, 4.0, 8.0]:
            lr = LogisticRegression(C=C, solver="liblinear", max_iter=200,
                                    random_state=42)
            lr.fit(Xtr, ytr)
            acc = lr.score(Xva, yva)
            probs[f"lr{C}"] = lr.predict_proba(Xva)[:, 1]
            print(f"  LR C={C}: {acc:.5f}")
        for C in [0.5, 1.0, 2.0]:
            sv = LinearSVC(C=C, random_state=42)
            sv.fit(Xtr, ytr)
            acc = sv.score(Xva, yva)
            probs[f"svc{C}"] = sigmoid(sv.decision_function(Xva))
            print(f"  SVC C={C}: {acc:.5f}")
        # ensembles
        ens = {
            "lr4+svc1": 0.5 * probs["lr4.0"] + 0.5 * probs["svc1.0"],
            "lr2+lr4+svc05+svc1": 0.25 * (probs["lr2.0"] + probs["lr4.0"]
                                          + probs["svc0.5"] + probs["svc1.0"]),
        }
        for ename, p in ens.items():
            acc = ((p >= 0.5).astype(int) == yva).mean()
            print(f"  ENS {ename}: {acc:.5f}")


if __name__ == "__main__":
    main()

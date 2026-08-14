# -*- coding: utf-8 -*-
"""Round 4: SVC C sweep on combined view; ensemble weight search; seeds."""
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solution import mark_aspect, softmax, LABELS

train = pd.read_csv("train.csv")
tr_texts = [mark_aspect(s, a) for s, a in zip(train.sentence, train.aspect)]
y = train.label.map({l: i for i, l in enumerate(LABELS)}).values


def build_X():
    v1 = TfidfVectorizer(analyzer="char", ngram_range=(3, 6), min_df=3, sublinear_tf=True, max_features=300000)
    v2 = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=300000)
    return hstack([v1.fit_transform(tr_texts), v2.fit_transform(tr_texts)]).tocsr(), (v1, v2)


def cv_oofs(X, y, models, n=5, seed=42):
    skf = StratifiedKFold(n_splits=n, shuffle=True, random_state=seed)
    oofs = [np.zeros((len(y), 3)) for _ in models]
    for tr, va in skf.split(X, y):
        for i, mfn in enumerate(models):
            m = mfn()
            m.fit(X[tr], y[tr])
            if hasattr(m, "predict_proba"):
                oofs[i][va] = m.predict_proba(X[va])
            else:
                oofs[i][va] = softmax(m.decision_function(X[va]))
    return oofs


def f1(o):
    return f1_score(y, o.argmax(1), average="macro")


X, _ = build_X()

# SVC C sweep on combined
for C in [0.1, 0.15, 0.2, 0.3, 0.5]:
    o = cv_oofs(X, y, [lambda C=C: LinearSVC(C=C, class_weight="balanced")])
    print(f"comb SVC C={C}: {f1(o[0]):.4f}", flush=True)

# LR C sweep on combined
for C in [1.0, 1.5, 2.0, 3.0]:
    o = cv_oofs(X, y, [lambda C=C: LogisticRegression(C=C, max_iter=3000, class_weight="balanced")])
    print(f"comb LR C={C}: {f1(o[0]):.4f}", flush=True)

# ensemble weight search
oofs = cv_oofs(X, y, [
    lambda: LogisticRegression(C=2.0, max_iter=3000, class_weight="balanced"),
    lambda: LinearSVC(C=0.2, class_weight="balanced"),
])
best = (0, None)
for w in np.arange(0.0, 1.05, 0.1):
    f = f1(w * oofs[0] + (1 - w) * oofs[1])
    print(f"w_LR={w:.1f}: {f:.4f}", flush=True)
    if f > best[0]:
        best = (f, w)
print("best", best)

# seed averaging
for seed in [1, 7]:
    o2 = cv_oofs(X, y, [lambda: LinearSVC(C=0.2, class_weight="balanced")], seed=seed)
    print(f"seed{seed} SVC: {f1(o2[0]):.4f}", flush=True)

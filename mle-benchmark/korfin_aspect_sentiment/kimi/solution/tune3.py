# -*- coding: utf-8 -*-
"""Third-round: combine multiple char views + tune C on char(3,6)."""
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


# 1) C tuning on char(3,6) min3
v = TfidfVectorizer(analyzer="char", ngram_range=(3, 6), min_df=3,
                    sublinear_tf=True, max_features=300000)
X36 = v.fit_transform(tr_texts)
for C in [1.0, 2.0, 3.0, 4.0]:
    o = cv_oofs(X36, y, [lambda C=C: LogisticRegression(C=C, max_iter=3000, class_weight="balanced")])
    print(f"char(3,6) LR C={C}: {f1(o[0]):.4f}", flush=True)
for C in [0.05, 0.1, 0.2]:
    o = cv_oofs(X36, y, [lambda C=C: LinearSVC(C=C, class_weight="balanced")])
    print(f"char(3,6) SVC C={C}: {f1(o[0]):.4f}", flush=True)

# 2) combined views
v1 = TfidfVectorizer(analyzer="char", ngram_range=(3, 6), min_df=3, sublinear_tf=True, max_features=300000)
v2 = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=300000)
v3 = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=5, sublinear_tf=True, max_features=200000)
Xc = hstack([v1.fit_transform(tr_texts), v2.fit_transform(tr_texts)]).tocsr()
Xc3 = hstack([v1.fit_transform(tr_texts), v2.fit_transform(tr_texts), v3.fit_transform(tr_texts)]).tocsr()
models = [
    lambda: LogisticRegression(C=2.0, max_iter=3000, class_weight="balanced"),
    lambda: LinearSVC(C=0.1, class_weight="balanced"),
]
for name, X in [("char36+char25", Xc), ("char36+char25+wb25", Xc3)]:
    oofs = cv_oofs(X, y, models)
    ens = 0.5 * oofs[0] + 0.5 * oofs[1]
    print(f"{name}: LR={f1(oofs[0]):.4f} SVC={f1(oofs[1]):.4f} ENS={f1(ens):.4f}", flush=True)

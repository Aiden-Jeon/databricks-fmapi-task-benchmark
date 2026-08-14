# -*- coding: utf-8 -*-
"""Second-round tuning: char vectorizer variants + ensembles."""
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


def cv_ens(X, y, models, weights=None, n=5, seed=42):
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
    f1s = [f1_score(y, o.argmax(1), average="macro") for o in oofs]
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    ens = sum(w * o for w, o in zip(weights, oofs))
    f_ens = f1_score(y, ens.argmax(1), average="macro")
    return f1s, f_ens


for name, kw in [
    ("wb(2,5) min2", dict(analyzer="char_wb", ngram_range=(2, 5), min_df=2)),
    ("wb(2,5) min3", dict(analyzer="char_wb", ngram_range=(2, 5), min_df=3)),
    ("wb(2,5) min5", dict(analyzer="char_wb", ngram_range=(2, 5), min_df=5)),
    ("wb(1,5) min2", dict(analyzer="char_wb", ngram_range=(1, 5), min_df=2)),
    ("wb(2,6) min3", dict(analyzer="char_wb", ngram_range=(2, 6), min_df=3)),
    ("char(2,5) min2", dict(analyzer="char", ngram_range=(2, 5), min_df=2)),
    ("char(3,6) min3", dict(analyzer="char", ngram_range=(3, 6), min_df=3)),
]:
    v = TfidfVectorizer(sublinear_tf=True, max_features=300000, **kw)
    X = v.fit_transform(tr_texts)
    f1s, f_ens = cv_ens(X, y, [
        lambda: LogisticRegression(C=2.0, max_iter=3000, class_weight="balanced"),
        lambda: LinearSVC(C=0.1, class_weight="balanced"),
    ])
    print(f"{name}: LR={f1s[0]:.4f} SVC={f1s[1]:.4f} ENS={f_ens:.4f}", flush=True)

# -*- coding: utf-8 -*-
"""Round 5: extra features (sentiment lexicon stats, aspect-window) + seed variance."""
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
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

POS = ["상승", "호조", "개선", "성장", "확대", "긍정", "수혜", "호재", "양호", "증가", "회복",
       "강세", "흑자", "최대", "기대", "돌파", "호실적", "늘어", "올라", "반등", "좋", "높"]
NEG = ["하락", "부진", "악화", "감소", "우려", "적자", "손실", "위축", "둔화", "리스크", "급락",
       "약세", "부담", "축소", "하향", "불안", "피해", "낮", "줄어", "떨어", "압박", "타격"]


def extra_features(df):
    rows = []
    for s, a in zip(df.sentence, df.aspect):
        s = str(s); a = str(a)
        pos = sum(s.count(w) for w in POS)
        neg = sum(s.count(w) for w in NEG)
        idx = s.find(a)
        wpos = wneg = 0
        if idx >= 0:
            win = s[max(0, idx - 40): idx + len(a) + 40]
            wpos = sum(win.count(w) for w in POS)
            wneg = sum(win.count(w) for w in NEG)
        rows.append([pos, neg, pos - neg, wpos, wneg, wpos - wneg,
                     len(s), s.count(","), 1 if a in s else 0])
    return csr_matrix(np.array(rows, dtype=np.float64))


def build_X(texts, vecs=None, extra=None):
    if vecs is None:
        v1 = TfidfVectorizer(analyzer="char", ngram_range=(3, 6), min_df=3, sublinear_tf=True, max_features=300000)
        v2 = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=300000)
        X = hstack([v1.fit_transform(texts), v2.fit_transform(texts)]).tocsr()
        vecs = (v1, v2)
    else:
        v1, v2 = vecs
        X = hstack([v1.transform(texts), v2.transform(texts)]).tocsr()
    if extra is not None:
        X = hstack([X, extra]).tocsr()
    return X, vecs


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


X_base, vecs = build_X(tr_texts)
X_ext, _ = build_X(tr_texts, extra=extra_features(train))

models = [
    lambda: LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced"),
    lambda: LinearSVC(C=0.1, class_weight="balanced"),
]
for name, X in [("base", X_base), ("base+lex", X_ext)]:
    oofs = cv_oofs(X, y, models)
    ens = 0.3 * oofs[0] + 0.7 * oofs[1]
    print(f"{name}: LR={f1(oofs[0]):.4f} SVC={f1(oofs[1]):.4f} ENS={f1(ens):.4f}", flush=True)

# seed variance of SVC on combined
for seed in [1, 7, 42, 100, 2024]:
    o = cv_oofs(X_base, y, [lambda: LinearSVC(C=0.1, class_weight="balanced")], seed=seed)
    print(f"SVC seed={seed}: {f1(o[0]):.4f}", flush=True)

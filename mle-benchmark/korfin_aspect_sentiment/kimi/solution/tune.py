# -*- coding: utf-8 -*-
"""Hyperparameter / feature search via 5-fold CV."""
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solution import mark_aspect, softmax, LABELS

train = pd.read_csv("train.csv")
tr_texts = [mark_aspect(s, a) for s, a in zip(train.sentence, train.aspect)]
y = train.label.map({l: i for i, l in enumerate(LABELS)}).values


def make_X(cfg):
    vecs = []
    feats = []
    if cfg.get("char"):
        v = TfidfVectorizer(analyzer="char_wb", ngram_range=cfg["char"],
                            min_df=cfg.get("min_df", 2), sublinear_tf=True,
                            max_features=cfg.get("maxf", 200000))
        feats.append(v.fit_transform(tr_texts))
        vecs.append(v)
    if cfg.get("word"):
        v = TfidfVectorizer(analyzer="word", ngram_range=cfg["word"],
                            token_pattern=r"(?u)\S+", min_df=cfg.get("min_df", 2),
                            sublinear_tf=True)
        feats.append(v.fit_transform(tr_texts))
        vecs.append(v)
    return hstack(feats).tocsr(), vecs


def cv_model(X, y, model_fn, n=5, seed=42):
    skf = StratifiedKFold(n_splits=n, shuffle=True, random_state=seed)
    oof = np.zeros((len(y), 3))
    for tr, va in skf.split(X, y):
        m = model_fn()
        m.fit(X[tr], y[tr])
        if hasattr(m, "predict_proba"):
            oof[va] = m.predict_proba(X[va])
        else:
            oof[va] = softmax(m.decision_function(X[va]))
    return f1_score(y, oof.argmax(1), average="macro"), oof


results = {}

configs = {
    "char(2,5)+word(1,2)": {"char": (2, 5), "word": (1, 2)},
    "char(2,4)+word(1,2)": {"char": (2, 4), "word": (1, 2)},
    "char(3,5)+word(1,1)": {"char": (3, 5), "word": (1, 1)},
    "char(2,6)+word(1,2)": {"char": (2, 6), "word": (1, 2)},
    "char(2,5) only": {"char": (2, 5)},
}
for name, cfg in configs.items():
    X, _ = make_X(cfg)
    f, _ = cv_model(X, y, lambda: LogisticRegression(C=2.0, max_iter=3000, class_weight="balanced"))
    results[name] = f
    print(f"{name}: LR f1={f:.4f}", flush=True)

# Use best feature config for C tuning
best = max(results, key=results.get)
print("best features:", best)
X, _ = make_X(configs[best])
for C in [0.5, 1.0, 2.0, 4.0, 8.0]:
    f, _ = cv_model(X, y, lambda C=C: LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))
    print(f"LR C={C}: f1={f:.4f}", flush=True)
for C in [0.1, 0.25, 0.5, 1.0]:
    f, _ = cv_model(X, y, lambda C=C: LinearSVC(C=C, class_weight="balanced"))
    print(f"SVC C={C}: f1={f:.4f}", flush=True)

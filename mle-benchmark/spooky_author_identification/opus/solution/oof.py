"""Compute out-of-fold and test predictions for a set of base models.

Caches results to solution/oof_cache/<name>.npz so blending can be iterated cheaply.
Usage: python solution/oof.py [model_name ...]   (no args = all)
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

CLASSES = ["EAP", "HPL", "MWS"]
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "oof_cache")
N_FOLDS = 5
SEED = 42


# ---------------------------------------------------------------- NB-features
class NBTransformer(BaseEstimator):
    """Multiply features by per-class log-count ratio (NBSVM style, one-vs-rest)."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        rs = []
        for c in self.classes_:
            p = np.asarray(X[y == c].sum(0)).ravel() + self.alpha
            q = np.asarray(X[y != c].sum(0)).ravel() + self.alpha
            r = np.log((p / p.sum()) / (q / q.sum()))
            rs.append(r)
        self.r_ = np.vstack(rs)  # (n_classes, n_features)
        return self

    def transform(self, X):
        return sp.hstack([X.multiply(r) for r in self.r_]).tocsr()


class HandFeatures(BaseEstimator):
    """Simple stylometric features."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = []
        for t in X:
            n = len(t)
            words = t.split()
            nw = max(len(words), 1)
            uniq = len(set(w.lower() for w in words)) / nw
            out.append([
                n, nw, n / nw, uniq,
                t.count(","), t.count(";"), t.count(":"), t.count('"'),
                t.count("'"), t.count("."), t.count("?"), t.count("!"),
                t.count("-"), t.count("("),
                sum(c.isupper() for c in t) / max(n, 1),
                sum(c.isdigit() for c in t) / max(n, 1),
                np.mean([len(w) for w in words]) if words else 0.0,
                max([len(w) for w in words]) if words else 0.0,
            ])
        return np.asarray(out, dtype=np.float64)


def word_tfidf(**kw):
    d = dict(ngram_range=(1, 2), sublinear_tf=True, min_df=2, strip_accents="unicode",
             token_pattern=r"\w{1,}", analyzer="word")
    d.update(kw)
    return TfidfVectorizer(**d)


def char_tfidf(**kw):
    d = dict(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=3,
             strip_accents="unicode")
    d.update(kw)
    return TfidfVectorizer(**d)


def build_models():
    m = {}
    # --- word tf-idf + logistic regression
    m["lr_word"] = make_pipeline(word_tfidf(), LogisticRegression(C=10, max_iter=3000))
    m["lr_word1"] = make_pipeline(word_tfidf(ngram_range=(1, 1), min_df=1),
                                  LogisticRegression(C=6, max_iter=3000))
    # --- char tf-idf + logistic regression
    m["lr_char"] = make_pipeline(char_tfidf(), LogisticRegression(C=10, max_iter=3000))
    m["lr_char25"] = make_pipeline(
        TfidfVectorizer(analyzer="char", ngram_range=(1, 4), sublinear_tf=True, min_df=3,
                        strip_accents="unicode"),
        LogisticRegression(C=6, max_iter=3000))
    # --- union word+char
    m["lr_union"] = Pipeline([
        ("f", FeatureUnion([("w", word_tfidf()), ("c", char_tfidf())])),
        ("clf", LogisticRegression(C=10, max_iter=3000)),
    ])
    # --- multinomial naive bayes
    m["mnb_word"] = make_pipeline(
        CountVectorizer(ngram_range=(1, 2), min_df=1, strip_accents="unicode",
                        token_pattern=r"\w{1,}"),
        MultinomialNB(alpha=0.05))
    m["mnb_tfidf"] = make_pipeline(word_tfidf(min_df=1), MultinomialNB(alpha=0.02))
    m["mnb_char"] = make_pipeline(
        CountVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                        strip_accents="unicode"),
        MultinomialNB(alpha=0.2))
    m["cnb_word"] = make_pipeline(word_tfidf(min_df=1), ComplementNB(alpha=0.3))
    # --- NBSVM style
    m["nb_lr_word"] = Pipeline([
        ("v", CountVectorizer(ngram_range=(1, 2), min_df=2, strip_accents="unicode",
                              binary=True, token_pattern=r"\w{1,}")),
        ("nb", NBTransformer(alpha=1.0)),
        ("clf", LogisticRegression(C=1.0, max_iter=3000)),
    ])
    # --- SGD (modified huber gives probabilities)
    m["sgd_word"] = make_pipeline(
        word_tfidf(),
        SGDClassifier(loss="modified_huber", alpha=1e-5, max_iter=60, tol=1e-4,
                      random_state=SEED))
    # --- calibrated linear SVM on union features
    m["svc_word"] = make_pipeline(
        word_tfidf(),
        CalibratedClassifierCV(LinearSVC(C=0.5, dual=True, max_iter=5000), cv=3,
                               method="sigmoid"))
    m["svc_char"] = make_pipeline(
        char_tfidf(),
        CalibratedClassifierCV(LinearSVC(C=0.5, dual=True, max_iter=5000), cv=3,
                               method="sigmoid"))
    # --- SVD + hand features + LR (dense, different inductive bias)
    m["svd_hand_lr"] = Pipeline([
        ("f", FeatureUnion([
            ("svd", make_pipeline(word_tfidf(), TruncatedSVD(150, random_state=SEED))),
            ("hand", HandFeatures()),
        ])),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=3000)),
    ])
    return m


def run(name, model, train, test, y):
    path = os.path.join(CACHE, name + ".npz")
    if os.path.exists(path):
        d = np.load(path)
        print(f"{name:14s} cached  logloss={log_loss(y, d['oof'], labels=CLASSES):.5f}")
        return
    t0 = time.time()
    cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(train), 3))
    Xtr_all = train["text"].values
    Xte = test["text"].values
    test_p = np.zeros((len(test), 3))
    for tr, va in cv.split(Xtr_all, y):
        mdl = clone(model)
        mdl.fit(Xtr_all[tr], y[tr])
        cls = list(mdl.classes_)
        idx = [cls.index(c) for c in CLASSES]
        oof[va] = mdl.predict_proba(Xtr_all[va])[:, idx]
        test_p += mdl.predict_proba(Xte)[:, idx] / N_FOLDS
    ll = log_loss(y, oof, labels=CLASSES)
    np.savez_compressed(path, oof=oof, test=test_p)
    print(f"{name:14s} logloss={ll:.5f}  ({time.time() - t0:.0f}s)")


def main():
    os.makedirs(CACHE, exist_ok=True)
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    y = train["author"].values
    models = build_models()
    wanted = sys.argv[1:] or list(models)
    for name in wanted:
        run(name, models[name], train, test, y)


if __name__ == "__main__":
    main()

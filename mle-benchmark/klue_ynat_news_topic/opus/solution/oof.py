"""Compute OOF + test decision-function matrices for diverse base models.

Usage: python oof.py <model_key>
Saves /tmp/opencode/oof_<key>.npz  with arrays oof (n_train x 7), test (n_test x 7)
"""
import sys, os, time
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier, SGDClassifier, LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.base import BaseEstimator, TransformerMixin
import scipy.sparse as sp


class NBTransform(BaseEstimator, TransformerMixin):
    """Scale features by the one-vs-rest NB log-count ratio, averaged over classes.

    Gives an NB-SVM style model (Wang & Manning 2012) generalised to multiclass by
    using the max |log-ratio| across classes as the per-feature weight.
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        y = np.asarray(y)
        R = []
        for c in np.unique(y):
            m = (y == c)
            p = np.asarray(X[m].sum(0)).ravel() + self.alpha
            q = np.asarray(X[~m].sum(0)).ravel() + self.alpha
            R.append(np.log((p / p.sum()) / (q / q.sum())))
        self.r_ = np.abs(np.vstack(R)).max(0)
        return self

    def transform(self, X):
        return X.multiply(sp.csr_matrix(self.r_)).tocsr()

TASK = "/tmp/kmle/M1_t3_ynat_full_20260804_033458/task"
CACHE = "/tmp/opencode"
SEED = 42
NFOLD = 5


def tv(analyzer, ng, min_df=1, sub=True):
    return TfidfVectorizer(analyzer=analyzer, ngram_range=ng, min_df=min_df, sublinear_tf=sub)


def build(key):
    """Return (feature_union, classifier)."""
    W12 = ("word", (1, 2))
    if key == "A":   # best single
        f = make_union(tv(*W12), tv("char_wb", (1, 3)))
        c = LinearSVC(C=0.2, dual=True)
    elif key == "B":
        f = make_union(tv(*W12), tv("char_wb", (2, 5)))
        c = LinearSVC(C=0.2, dual=True)
    elif key == "C":
        f = make_union(tv("char", (1, 5)))
        c = LinearSVC(C=0.2, dual=True)
    elif key == "D":
        f = make_union(tv("word", (1, 3)))
        c = LinearSVC(C=0.3, dual=True)
    elif key == "E":  # ridge on best feats
        f = make_union(tv(*W12), tv("char_wb", (1, 3)))
        c = RidgeClassifier(alpha=1.0)
    elif key == "F":  # ComplementNB
        f = make_union(tv(*W12), tv("char_wb", (1, 4)))
        c = ComplementNB(alpha=1.0)
    elif key == "G":  # SGD modified_huber, different loss surface
        f = make_union(tv(*W12), tv("char_wb", (1, 4)))
        c = SGDClassifier(loss="modified_huber", alpha=2e-6, max_iter=30,
                          tol=1e-4, random_state=SEED)
    elif key == "H":  # balanced class weights -> helps macro F1
        f = make_union(tv(*W12), tv("char_wb", (1, 3)))
        c = LinearSVC(C=0.2, dual=True, class_weight="balanced")
    elif key == "I":  # non-sublinear tf, char_wb 1-4
        f = make_union(tv("word", (1, 2), sub=False), tv("char_wb", (1, 4), sub=False))
        c = LinearSVC(C=0.2, dual=True)
    elif key == "J":  # LogReg (probabilities), fast solver on best feats
        f = make_union(tv(*W12), tv("char_wb", (1, 3)))
        c = LogisticRegression(C=8, max_iter=1000, solver="liblinear")
    elif key == "K":  # binary presence features
        f = make_union(
            TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, binary=True, use_idf=False, norm="l2"),
            TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), min_df=1, binary=True, use_idf=False, norm="l2"))
        c = LinearSVC(C=0.3, dual=True)
    elif key == "M":  # NB-SVM: features reweighted by NB log-count ratio
        f = make_pipeline(make_union(tv(*W12), tv("char_wb", (1, 4))), NBTransform())
        c = LinearSVC(C=0.2, dual=True)
    elif key == "N":  # char_wb only, wider window
        f = make_union(tv("char_wb", (2, 6), min_df=2))
        c = LinearSVC(C=0.3, dual=True)
    elif key == "P":  # cosine kNN -- very different inductive bias
        f = make_union(tv(*W12), tv("char_wb", (1, 4), min_df=2))
        c = KNeighborsClassifier(n_neighbors=40, metric="cosine", weights="distance",
                                 algorithm="brute", n_jobs=2)
    else:
        raise ValueError(key)
    return f, c


def scores(clf, Xt):
    if hasattr(clf, "decision_function"):
        d = clf.decision_function(Xt)
    else:
        d = np.log(np.clip(clf.predict_proba(Xt), 1e-9, None))
    return d


def run(key):
    tr = pd.read_csv(f"{TASK}/train.csv")
    te = pd.read_csv(f"{TASK}/test.csv")
    X, y = tr.title.values, tr.label.values
    classes = np.array(sorted(tr.label.unique()))
    oof = np.zeros((len(X), len(classes)))
    tst = np.zeros((len(te), len(classes)))
    skf = StratifiedKFold(NFOLD, shuffle=True, random_state=SEED)
    t0 = time.time()
    for f_idx, (a, b) in enumerate(skf.split(X, y)):
        fu, cl = build(key)
        pipe = make_pipeline(fu, cl)
        pipe.fit(X[a], y[a])
        assert list(pipe[-1].classes_) == list(classes)
        oof[b] = scores(pipe[-1], pipe[:-1].transform(X[b]))
    # full-train model for test predictions
    fu, cl = build(key)
    pipe = make_pipeline(fu, cl)
    pipe.fit(X, y)
    tst = scores(pipe[-1], pipe[:-1].transform(te.title.values))
    s = f1_score(y, classes[oof.argmax(1)], average="macro")
    np.savez_compressed(f"{CACHE}/oof_{key}.npz", oof=oof, test=tst, classes=classes)
    print(f"[{key}] oof macroF1={s:.5f} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    run(sys.argv[1])

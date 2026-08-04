"""Shared model definitions for the KoBEST SentiNeg solution."""
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import normalize as l2_normalize
from sklearn.svm import LinearSVC, SVC

from features import jamo_norm, normalize

VIEWS = ("norm", "jamo")


def make_views(sentences):
    s = pd.Series(sentences).astype(str)
    return {"norm": s.map(normalize).values, "jamo": s.map(jamo_norm).values}


def _tv(**kw):
    kw.setdefault("sublinear_tf", True)
    return TfidfVectorizer(**kw)


class Featurizer:
    """Fits several TF-IDF blocks, each on a chosen text view."""

    def __init__(self, blocks):
        # blocks: list of (view, TfidfVectorizer kwargs)
        self.blocks = blocks

    def fit_transform(self, views):
        self.vecs_ = []
        mats = []
        for view, kw in self.blocks:
            v = _tv(**kw)
            mats.append(v.fit_transform(views[view]))
            self.vecs_.append((view, v))
        return hstack(mats).tocsr()

    def transform(self, views):
        mats = [v.transform(views[view]) for view, v in self.vecs_]
        return hstack(mats).tocsr()


class NBSVM(BaseEstimator, ClassifierMixin):
    """NB-SVM (Wang & Manning 2012): scale features by the NB log-count ratio,
    L2-renormalise, then fit a linear SVM."""

    def __init__(self, C=0.3, alpha=1.0):
        self.C = C
        self.alpha = alpha

    def _ratio(self, X, y):
        p = np.asarray(X[y == 1].sum(axis=0)).ravel() + self.alpha
        q = np.asarray(X[y == 0].sum(axis=0)).ravel() + self.alpha
        return np.log((p / p.sum()) / (q / q.sum()))

    def _scale(self, X):
        return l2_normalize(X.multiply(self.r_).tocsr())

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.r_ = self._ratio(X, y)
        self.est_ = LinearSVC(C=self.C, dual=True, max_iter=8000, random_state=0)
        self.est_.fit(self._scale(X), y)
        return self

    def decision_function(self, X):
        return self.est_.decision_function(self._scale(X))

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)


CHAR14 = ("norm", dict(analyzer="char", ngram_range=(1, 4), min_df=2))
CHAR15 = ("norm", dict(analyzer="char", ngram_range=(1, 5), min_df=2))
CHARWB = ("norm", dict(analyzer="char_wb", ngram_range=(2, 5), min_df=2))
JAMO26 = ("jamo", dict(analyzer="char", ngram_range=(2, 6), min_df=2))
JAMO28 = ("jamo", dict(analyzer="char", ngram_range=(2, 8), min_df=2))
WORD12 = ("norm", dict(analyzer="word", ngram_range=(1, 2), min_df=1))


def model_zoo():
    """name -> (blocks, estimator factory, score_fn)"""
    z = {}
    z["A_lr"] = ([CHAR14, JAMO26], lambda: LogisticRegression(C=3, max_iter=5000))
    z["B_lr"] = ([CHAR14, JAMO26, WORD12], lambda: LogisticRegression(C=3, max_iter=5000))
    z["C_svc"] = ([CHAR14, JAMO26], lambda: LinearSVC(C=0.3, dual=True, max_iter=8000, random_state=0))
    z["D_svc"] = ([CHAR15, JAMO28, WORD12], lambda: LinearSVC(C=0.3, dual=True, max_iter=8000, random_state=0))
    z["E_ridge"] = ([CHAR14, JAMO26], lambda: RidgeClassifier(alpha=0.5))
    z["F_cnb"] = ([CHAR14, JAMO26], lambda: ComplementNB(alpha=0.3))
    z["G_rbf"] = ([CHAR14, JAMO26], lambda: SVC(C=10, kernel="rbf", gamma="scale"))
    z["H_sgd"] = ([CHAR14, JAMO26, WORD12],
                  lambda: SGDClassifier(loss="modified_huber", alpha=1e-5,
                                        max_iter=3000, tol=1e-4, random_state=0))
    z["I_lr_wb"] = ([CHARWB, JAMO26], lambda: LogisticRegression(C=3, max_iter=5000))
    z["J_nbsvm"] = ([CHAR14, JAMO26], lambda: NBSVM(C=0.3))
    return z


def decision(est, X):
    """Signed score, standardised so blending is scale-comparable."""
    if hasattr(est, "predict_proba"):
        p = est.predict_proba(X)[:, 1]
        return np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6))
    return est.decision_function(X)

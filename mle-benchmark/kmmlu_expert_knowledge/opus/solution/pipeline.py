"""Shared pipeline pieces: feature assembly (dense + LSA + retrieval) and models."""
import numpy as np
from scipy import sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

from feats import build_features, OPTS

RS = 42
K = 25
HGB_CONFIGS = [
    dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=40,
         l2_regularization=1.0),
    dict(max_iter=450, learning_rate=0.04, max_leaf_nodes=8, min_samples_leaf=60,
         l2_regularization=2.0),
    dict(max_iter=200, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=30,
         l2_regularization=1.0),
]


def opt_texts(df):
    return [str(df[c].iloc[i]) for i in range(len(df)) for c in OPTS]


def expand(idx):
    return (np.asarray(idx)[:, None] * 4 + np.arange(4)).ravel()


def zgroup(s):
    s = np.asarray(s, dtype=float).reshape(-1, 4)
    return ((s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-9)).ravel()


class TextSpace:
    """Unsupervised tfidf + LSA space fitted on all available text."""

    def __init__(self, texts):
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3,
                                   sublinear_tf=True)
        M = self.vec.fit_transform(texts)
        self.svd = TruncatedSVD(200, random_state=RS).fit(M)

    def sparse(self, texts):
        return normalize(self.vec.transform(texts))

    def dense(self, texts):
        return normalize(self.svd.transform(self.vec.transform(texts)))


def lsa_sim(space, questions, options):
    Qs = space.dense(questions)
    Ps = space.dense(options)
    return (Ps * np.repeat(Qs, 4, axis=0)).sum(1)


def retrieval_feats(space, q_pool, opt_pool, y_pool, q_query, opt_query, k=K,
                    exclude_self=False):
    """For each query option: similarity to correct answers vs distractors of the
    most similar pool questions. y_pool = 0..3 index of correct option."""
    Qp = space.sparse(q_pool)
    Pp = space.sparse(opt_pool)
    Qq = space.sparse(q_query)
    Pq = space.sparse(opt_query)
    npool = len(q_pool)
    nq = len(q_query)
    corr = np.arange(npool) * 4 + y_pool
    Pc = Pp[corr]
    dis_idx = np.array([[i * 4 + j for j in range(4) if j != y_pool[i]]
                        for i in range(npool)])
    RF = np.zeros((nq * 4, 6))
    S = (Qq @ Qp.T).toarray()
    kk = min(k, npool - 1)
    for a in range(nq):
        row = S[a]
        if exclude_self:
            row = row.copy()
            row[a] = -1e9
        nb = np.argpartition(-row, kk)[:kk]
        w = row[nb]
        o = np.argsort(-w)
        nb, w = nb[o], np.maximum(w[o], 0)
        Po = Pq[a * 4:a * 4 + 4]
        simc = (Po @ Pc[nb].T).toarray()
        simd = (Po @ Pp[dis_idx[nb].ravel()].T).toarray().reshape(4, kk, 3).mean(2)
        ww = w / (w.sum() + 1e-9)
        fc, fd = simc @ ww, simd @ ww
        sl = slice(a * 4, a * 4 + 4)
        RF[sl, 0] = fc
        RF[sl, 1] = fd
        RF[sl, 2] = fc - fd
        RF[sl, 3] = simc[:, :5].mean(1) - simd[:, :5].mean(1)
        RF[sl, 4] = simc.max(1)
        RF[sl, 5] = w[0]
    return RF


def hgb_scores(Xtr, ybin, Xte, seeds=(0, 1, 2)):
    """Average predicted P(correct) over configs x seeds."""
    out = np.zeros(Xte.shape[0])
    cnt = 0
    for cfg in HGB_CONFIGS:
        for s in seeds:
            m = HistGradientBoostingClassifier(random_state=RS + 100 * s, **cfg)
            m.fit(Xtr, ybin)
            out += zgroup(m.predict_proba(Xte)[:, 1])
            cnt += 1
    return out / cnt


def text_lr_scores(train_texts, ybin, test_texts):
    """Char + word tfidf logistic on option text only."""
    outs = []
    for kw in [dict(analyzer="char_wb", ngram_range=(2, 4), min_df=3, sublinear_tf=True),
               dict(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                    token_pattern=r"\S+")]:
        v = TfidfVectorizer(**kw)
        A = v.fit_transform(train_texts)
        B = v.transform(test_texts)
        m = LogisticRegression(C=1.0, max_iter=2000)
        m.fit(A, ybin)
        outs.append(zgroup(m.decision_function(B)))
    return outs

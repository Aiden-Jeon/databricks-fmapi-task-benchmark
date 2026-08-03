"""Leak-free training pipeline for the KMMLU 4-choice task.

`fit_predict(train_df, test_df)` trains everything on `train_df` only and returns
a score matrix (n_test x 4); the predicted label is argmax + 1.

Components
----------
1. Hand-crafted surface features (features.build_features) plus within-question
   relative / rank transforms so that a point-wise learner can behave list-wise.
2. Stacked TF-IDF text scores (option text, question+option text) added as
   features; their values on the training rows are produced with inner
   cross-fitting so the booster never sees in-sample text scores.
3. A retrieval feature: similarity of (question, option) to the correct answers
   of the training set minus similarity to the wrong ones (also cross-fitted).
4. Point-wise HistGradientBoosting ensemble (several seeds) + logistic
   regression, averaged after per-question standardisation.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import build_features, option_texts

N_INNER = 5


def _labels_to_y(labels, n):
    y = np.zeros(n * 4)
    y[np.arange(n) * 4 + (labels - 1)] = 1
    return y


def _rel_block(X, cols):
    """Within-question mean-centred values and ranks for the given columns."""
    n = X.shape[0] // 4
    out = []
    for c in cols:
        v = X[:, c].reshape(n, 4)
        out.append((v - v.mean(1, keepdims=True)).reshape(-1))
        out.append(np.argsort(np.argsort(v, 1), 1).astype(float).reshape(-1))
    return np.stack(out, 1)


# ---------------------------------------------------------------- text scorers
def _text_scores(tr_txt, tr_y, te_txt, kind):
    if kind == "char_opt":
        v = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), min_df=3,
                            sublinear_tf=True)
        C = 0.5
    elif kind == "char_qo":
        v = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3,
                            sublinear_tf=True)
        C = 0.3
    else:
        raise ValueError(kind)
    A = v.fit_transform(tr_txt)
    B = v.transform(te_txt)
    m = LogisticRegression(C=C, max_iter=3000)
    m.fit(A, tr_y)
    return m.decision_function(B)


def _retrieval_scores(tr_df, tr_lab, te_qo):
    """max cosine sim to train correct answers minus to train wrong answers."""
    q = tr_df["question"].astype(str).values
    O = [tr_df[o].astype(str).values for o in ["A", "B", "C", "D"]]
    corr, wrong = [], []
    for r in range(len(tr_df)):
        for i in range(4):
            t = q[r] + " || " + O[i][r]
            (corr if i == tr_lab[r] - 1 else wrong).append(t)
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=2,
                        sublinear_tf=True)
    v.fit(corr + wrong)
    Kc, Kw = v.transform(corr), v.transform(wrong)
    T = v.transform(te_qo)
    sc = np.asarray((T @ Kc.T).max(1).todense()).ravel()
    sw = np.asarray((T @ Kw.T).max(1).todense()).ravel()
    return np.stack([sc, sw, sc - sw], 1)


def _stacked_columns(tr_df, te_df):
    """Cross-fitted stack columns for train rows + fitted-on-all for test rows."""
    n_tr, n_te = len(tr_df), len(te_df)
    lab_tr = tr_df["label"].values
    y_tr = _labels_to_y(lab_tr, n_tr)
    opt_tr, qo_tr = option_texts(tr_df)
    opt_te, qo_te = option_texts(te_df)
    opt_tr = np.array(opt_tr, dtype=object); qo_tr = np.array(qo_tr, dtype=object)

    S_tr = np.zeros((n_tr * 4, 5))
    kf = KFold(n_splits=N_INNER, shuffle=True, random_state=7)
    for a, b in kf.split(np.arange(n_tr)):
        ra = (a[:, None] * 4 + np.arange(4)).ravel()
        rb = (b[:, None] * 4 + np.arange(4)).ravel()
        S_tr[rb, 0] = _text_scores(opt_tr[ra], y_tr[ra], opt_tr[rb], "char_opt")
        S_tr[rb, 1] = _text_scores(qo_tr[ra], y_tr[ra], qo_tr[rb], "char_qo")
        S_tr[rb, 2:] = _retrieval_scores(tr_df.iloc[a], lab_tr[a], qo_tr[rb])

    S_te = np.zeros((n_te * 4, 5))
    S_te[:, 0] = _text_scores(opt_tr, y_tr, opt_te, "char_opt")
    S_te[:, 1] = _text_scores(qo_tr, y_tr, qo_te, "char_qo")
    S_te[:, 2:] = _retrieval_scores(tr_df, lab_tr, qo_te)
    return S_tr, S_te


def _augment(X, S):
    n = X.shape[0] // 4
    Z = np.hstack([X, S])
    # relative transforms of the stack columns
    rel = []
    for j in range(S.shape[1]):
        v = S[:, j].reshape(n, 4)
        rel.append((v - v.mean(1, keepdims=True)).reshape(-1))
        rel.append(np.argsort(np.argsort(v, 1), 1).astype(float).reshape(-1))
        sd = v.std(1, keepdims=True) + 1e-9
        rel.append(((v - v.mean(1, keepdims=True)) / sd).reshape(-1))
    return np.hstack([Z, np.stack(rel, 1)])


def _per_q_z(s, n):
    v = s.reshape(n, 4)
    v = (v - v.mean(1, keepdims=True)) / (v.std(1, keepdims=True) + 1e-9)
    return v.reshape(-1)


def fit_predict(tr_df, te_df, seeds=(0, 1, 2, 3, 4), use_stack=True,
                use_lr=True, return_parts=False):
    n_tr, n_te = len(tr_df), len(te_df)
    y_tr = _labels_to_y(tr_df["label"].values, n_tr)

    Ftr, vec = build_features(tr_df)
    Fte, _ = build_features(te_df, vec=vec)
    cols = [c for c in Ftr.columns if c not in ("qidx", "opt")]
    Xtr = Ftr[cols].values
    Xte = Fte[cols].values

    rel_src = [cols.index(c) for c in
               ["len", "sim_q", "sim_oth_mean", "nwords", "o_digits"] if c in cols]
    Xtr = np.hstack([Xtr, _rel_block(Xtr, rel_src)])
    Xte = np.hstack([Xte, _rel_block(Xte, rel_src)])

    if use_stack:
        Str, Ste = _stacked_columns(tr_df, te_df)
        Xtr = _augment(Xtr, Str)
        Xte = _augment(Xte, Ste)

    parts = {}
    acc = np.zeros(n_te * 4)
    for sd in seeds:
        m = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.04, max_leaf_nodes=15,
            min_samples_leaf=40, l2_regularization=1.0,
            max_features=0.8, random_state=sd)
        m.fit(Xtr, y_tr)
        acc += _per_q_z(m.predict_proba(Xte)[:, 1], n_te)
    gbm = acc / len(seeds)
    parts["gbm"] = gbm
    score = gbm.copy()

    if use_lr:
        sc = StandardScaler()
        A = sc.fit_transform(np.nan_to_num(Xtr))
        B = sc.transform(np.nan_to_num(Xte))
        lr = LogisticRegression(C=0.05, max_iter=3000)
        lr.fit(A, y_tr)
        s = _per_q_z(lr.decision_function(B), n_te)
        parts["lr"] = s
        score = 0.8 * gbm + 0.2 * s

    if return_parts:
        return score.reshape(n_te, 4), parts
    return score.reshape(n_te, 4)

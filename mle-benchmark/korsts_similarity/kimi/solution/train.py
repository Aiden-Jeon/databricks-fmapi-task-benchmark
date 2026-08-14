"""Train KorSTS similarity model and generate submission.

Feature groups:
  A. Hand-crafted pair features (word/char n-gram overlaps, sequence matching,
     length stats, number-token overlap)              [features.py]
  B. Korean-normalized overlap features: number words -> digits and
     particle (josa) stripping                         [textnorm.py]
  C. TF-IDF cosine similarities (word / char / stemmed-word / num-normalized-char)
  D. SVD pair features (abs-diff + product) on word & char TF-IDF

Model: 8 base regressors (HGB/ET/GBR/RF x2 seeds + Ridge) with 5-fold OOF
stacking via Ridge. Final predictions clipped to [0, 5].
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from sklearn.base import clone
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import build_matrix  # noqa: E402
from textnorm import tokens, normalize_number_text  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_SPLITS = 5
SEED = 42


def cosine_rows(a, b):
    num = a.multiply(b).sum(axis=1).A.ravel()
    na = np.sqrt(a.multiply(a).sum(axis=1)).A.ravel()
    nb = np.sqrt(b.multiply(b).sum(axis=1)).A.ravel()
    d = na * nb
    return np.where(d > 0, num / np.maximum(d, 1e-9), 0.0)


def svd_pair(a, b, at, bt, n_comp):
    svd = TruncatedSVD(n_components=n_comp, random_state=SEED)
    svd.fit(sp.vstack([a, b, at, bt]))
    xa, xb = svd.transform(a), svd.transform(b)
    xat, xbt = svd.transform(at), svd.transform(bt)
    tr = np.hstack([np.abs(xa - xb), xa * xb]).astype(np.float32)
    te = np.hstack([np.abs(xat - xbt), xat * xbt]).astype(np.float32)
    return tr, te


def _jac(a, b):
    a, b = set(a), set(b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _dice(a, b):
    a, b = set(a), set(b)
    if not a or not b:
        return 0.0
    return 2 * len(a & b) / (len(a) + len(b))


def _contain(a, b):
    a, b = set(a), set(b)
    if not a or not b:
        return 0.0
    s, l = (a, b) if len(a) <= len(b) else (b, a)
    return len(s & l) / len(s)


def normalized_overlap_features(df):
    """number-normalized / particle-stripped token overlap features."""
    out = []
    for s1, s2 in zip(df["sentence1"], df["sentence2"]):
        r1, r2 = tokens(s1), tokens(s2)
        p1, p2 = tokens(s1, stem=True), tokens(s2, stem=True)
        out.append([
            _jac(r1, r2), _dice(r1, r2), _contain(r1, r2),
            _jac(p1, p2), _dice(p1, p2), _contain(p1, p2),
            len(set(p1) & set(p2)),
        ])
    return np.asarray(out, dtype=np.float32)


def pearson(a, b):
    return stats.pearsonr(a, b)[0]


def build_features(train, test):
    corpus = pd.concat([train["sentence1"], train["sentence2"],
                        test["sentence1"], test["sentence2"]]).tolist()

    Xc_tr = build_matrix(train)
    Xc_te = build_matrix(test)
    Xn_tr = normalized_overlap_features(train)
    Xn_te = normalized_overlap_features(test)

    blocks_tr, blocks_te = [Xc_tr, Xn_tr], [Xc_te, Xn_te]

    specs = [
        # (name, vectorizer, svd_components or None)
        ("word", TfidfVectorizer(analyzer="word", tokenizer=lambda s: tokens(s),
                                 ngram_range=(1, 2), min_df=2, sublinear_tf=True), 48),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                 min_df=2, sublinear_tf=True), 48),
        ("stem", TfidfVectorizer(analyzer="word",
                                 tokenizer=lambda s: tokens(s, stem=True),
                                 ngram_range=(1, 2), min_df=2, sublinear_tf=True), None),
        ("nchar", TfidfVectorizer(analyzer="char_wb",
                                  preprocessor=normalize_number_text,
                                  ngram_range=(2, 4), min_df=2, sublinear_tf=True), None),
    ]
    for name, vec, n_comp in specs:
        vec.fit(corpus)
        a = vec.transform(train["sentence1"])
        b = vec.transform(train["sentence2"])
        at = vec.transform(test["sentence1"])
        bt = vec.transform(test["sentence2"])
        blocks_tr.append(cosine_rows(a, b).reshape(-1, 1))
        blocks_te.append(cosine_rows(at, bt).reshape(-1, 1))
        if n_comp:
            ftr, fte = svd_pair(a, b, at, bt, n_comp)
            blocks_tr.append(ftr)
            blocks_te.append(fte)
        print(f"  {name}: vocab={len(vec.vocabulary_)}", flush=True)

    X_tr = np.nan_to_num(np.hstack(blocks_tr), nan=0.0, posinf=0.0, neginf=0.0)
    X_te = np.nan_to_num(np.hstack(blocks_te), nan=0.0, posinf=0.0, neginf=0.0)
    return X_tr, X_te


def get_models():
    return {
        "hgb42": HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.05, min_samples_leaf=15,
            l2_regularization=1.0, random_state=42),
        "hgb7": HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.07, max_leaf_nodes=63,
            min_samples_leaf=10, l2_regularization=0.5, random_state=7),
        "et42": ExtraTreesRegressor(
            n_estimators=200, min_samples_leaf=2, n_jobs=4, random_state=42),
        "et7": ExtraTreesRegressor(
            n_estimators=200, min_samples_leaf=2, n_jobs=4, random_state=7),
        "gbr42": GradientBoostingRegressor(
            n_estimators=400, learning_rate=0.04, max_depth=3,
            subsample=0.8, random_state=42),
        "gbr7": GradientBoostingRegressor(
            n_estimators=500, learning_rate=0.03, max_depth=4,
            subsample=0.7, random_state=7),
        "rf42": RandomForestRegressor(
            n_estimators=150, min_samples_leaf=2, n_jobs=4, random_state=42),
        "ridge": Ridge(alpha=10.0),
    }


def fit_model(name, model, Xtr, ytr, Xva):
    m = clone(model)
    if name == "ridge":
        sc = StandardScaler().fit(Xtr)
        m.fit(sc.transform(Xtr), ytr)
        return m.predict(sc.transform(Xva))
    m.fit(Xtr, ytr)
    return m.predict(Xva)


def main():
    t0 = time.time()
    train = pd.read_csv(os.path.join(ROOT, "train.csv"))
    test = pd.read_csv(os.path.join(ROOT, "test.csv"))
    y = train["score"].values.astype(np.float32)
    print(f"train={train.shape} test={test.shape} mean={y.mean():.3f} std={y.std():.3f}",
          flush=True)

    print("building features ...", flush=True)
    X_tr, X_te = build_features(train, test)
    print(f"feature matrix: train={X_tr.shape} test={X_te.shape} ({time.time()-t0:.0f}s)",
          flush=True)

    models = get_models()
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = {n: np.zeros(len(train), dtype=np.float32) for n in models}
    test_preds = {n: np.zeros(len(test), dtype=np.float32) for n in models}

    for fold, (tr, va) in enumerate(kf.split(X_tr)):
        for n, mdl in models.items():
            oof[n][va] = fit_model(n, mdl, X_tr[tr], y[tr], X_tr[va])
            test_preds[n] += fit_model(n, mdl, X_tr[tr], y[tr], X_te) / N_SPLITS
        fs = {n: round(pearson(y[va], oof[n][va]), 4) for n in models}
        print(f"fold {fold}: {fs}  ({time.time()-t0:.0f}s)", flush=True)

    for n in models:
        print(f"OOF {n}: {pearson(y, oof[n]):.4f}", flush=True)

    stack_tr = np.vstack([oof[n] for n in models]).T
    stack_te = np.vstack([test_preds[n] for n in models]).T

    meta = Ridge(alpha=1.0)
    ms = []
    for tr, va in kf.split(stack_tr):
        meta.fit(stack_tr[tr], y[tr])
        ms.append(pearson(y[va], meta.predict(stack_tr[va])))
    print(f"stacked OOF pearson: {np.mean(ms):.4f} +- {np.std(ms):.4f}", flush=True)
    print(f"simple-avg OOF pearson: {pearson(y, stack_tr.mean(axis=1)):.4f}", flush=True)

    meta.fit(stack_tr, y)
    final = meta.predict(stack_te)
    if pearson(y, stack_tr.mean(axis=1)) > np.mean(ms):
        final = stack_te.mean(axis=1)
        print("using simple average (better OOF)")
    final = np.clip(final, 0.0, 5.0)

    out = pd.DataFrame({"id": test["id"], "score": final})
    assert out["id"].is_unique and len(out) == len(test)
    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    path = os.path.join(ROOT, "outputs", "submission.csv")
    out.to_csv(path, index=False)
    print(f"saved {path}  ({time.time()-t0:.0f}s total)")
    print(out.describe())


if __name__ == "__main__":
    main()

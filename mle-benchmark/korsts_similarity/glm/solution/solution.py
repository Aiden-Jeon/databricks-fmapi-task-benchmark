"""
KorSTS sentence similarity solution.

Approach: feature-based regression with a stacking ensemble.
Features (per sentence pair):
  - Hand features: token lengths, Jaccard, overlap ratio, char-ngram Jaccard,
    char-length stats.
  - TF-IDF blocks (word 1-2/1-3gram, char_wb 2-4/3-5gram, char 2-3/3-4gram,
    and a Jamo(NFD)-decomposed char 2-4gram block). For each block we compute
    cosine similarity, sum-of-squared differences, max difference, and
    common/union nonzero ratios.
Models (base learners, 5-fold OOF):
  - Ridge, SVR(rbf) x2 (C=2,5), GradientBoostingRegressor, RandomForest,
    ExtraTrees, HistGradientBoostingRegressor. Distance-based learners use
    StandardScaler-normalized features.
Stacking meta-learner:
  - Ridge or non-negative Lasso on the OOF base predictions, alpha chosen by
    nested 5-fold CV; the best of {single best base, simple average, stacking}
    is selected by CV Pearson correlation.
The metric is Pearson correlation; predictions are clipped to [0, 5].
Only train.csv is used; no internet / external weights.
"""
import os
import re
import sys
import unicodedata
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import (GradientBoostingRegressor, RandomForestRegressor,
                              ExtraTreesRegressor, HistGradientBoostingRegressor)
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import paired_cosine_distances
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr


def jamo(s):
    return unicodedata.normalize("NFD", str(s))


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "train.csv")
TEST = os.path.join(ROOT, "test.csv")
SUB = os.path.join(ROOT, "outputs", "submission.csv")
RNG = 42


def tokenize(s):
    s = str(s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"([^가-힣A-Za-z0-9])", r" \1 ", s)
    return s.split()


def jaccard(a, b):
    sa, sb = set(tokenize(a)), set(tokenize(b))
    if not sa and not sb:
        return 1.0
    u = len(sa | sb)
    return len(sa & sb) / u if u else 0.0


def overlap_ratio(a, b):
    sa, sb = set(tokenize(a)), set(tokenize(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / min(len(sa), len(sb))


def char_ngrams(s, n):
    s = " " + str(s).lower().strip() + " "
    return set(s[i:i+n] for i in range(len(s) - n + 1))


def char_jaccard(a, b, n):
    sa, sb = char_ngrams(a, n), char_ngrams(b, n)
    if not sa and not sb:
        return 1.0
    u = len(sa | sb)
    return len(sa & sb) / u if u else 0.0


def hand_features(df):
    rows = []
    for s1, s2 in zip(df["sentence1"], df["sentence2"]):
        t1, t2 = tokenize(s1), tokenize(s2)
        n1, n2 = len(t1), len(t2)
        jac = jaccard(s1, s2)
        ovr = overlap_ratio(s1, s2)
        inter = len(set(t1) & set(t2))
        union = len(set(t1) | set(t2))
        c2 = char_jaccard(s1, s2, 2)
        c3 = char_jaccard(s1, s2, 3)
        c4 = char_jaccard(s1, s2, 4)
        rows.append([
            n1, n2, abs(n1 - n2), n1 + n2,
            min(n1, n2) / max(n1, n2) if max(n1, n2) else 0.0,
            jac, ovr, inter, union,
            c2, c3, c4,
            len(str(s1)), len(str(s2)), abs(len(str(s1)) - len(str(s2))),
        ])
    return np.array(rows)


def tfidf_block(t1tr, t2tr, t1te, t2te, params):
    vec = TfidfVectorizer(**params)
    all_text = t1tr + t2tr
    vec.fit(all_text)
    return (vec.transform(t1tr), vec.transform(t2tr),
            vec.transform(t1te), vec.transform(t2te))


def sparse_pair_feats(Xa, Xb):
    cos = (1.0 - paired_cosine_distances(Xa, Xb)).ravel()
    diff = Xa - Xb
    absdiff = diff.multiply(diff)
    sumsq = np.asarray(absdiff.sum(axis=1)).ravel()
    maxval = np.asarray(absdiff.max(axis=1).todense()).ravel()
    inter = np.asarray((Xa.multiply(Xb) > 0).sum(axis=1)).ravel()
    union = np.asarray(((Xa > 0).astype(int) + (Xb > 0).astype(int) > 0)
                       .sum(axis=1)).ravel()
    overlap_ratio = inter / np.maximum(union, 1)
    return np.column_stack([cos, sumsq, maxval, inter, union, overlap_ratio])


def build_features(train_df, test_df):
    t1tr = train_df["sentence1"].astype(str).tolist()
    t2tr = train_df["sentence2"].astype(str).tolist()
    t1te = test_df["sentence1"].astype(str).tolist()
    t2te = test_df["sentence2"].astype(str).tolist()

    blocks = [
        dict(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=2),
        dict(analyzer="word", ngram_range=(1, 3), sublinear_tf=True, min_df=3),
        dict(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, min_df=2),
        dict(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=3),
        dict(analyzer="char", ngram_range=(2, 3), sublinear_tf=True, min_df=2),
        dict(analyzer="char", ngram_range=(3, 4), sublinear_tf=True, min_df=3),
    ]

    ftr_list, fte_list = [], []
    for cfg in blocks:
        A1, A2, B1, B2 = tfidf_block(t1tr, t2tr, t1te, t2te, cfg)
        ftr_list.append(sparse_pair_feats(A1, A2))
        fte_list.append(sparse_pair_feats(B1, B2))

    # jamo-decomposed char n-gram block
    j1tr = [jamo(s) for s in t1tr]
    j2tr = [jamo(s) for s in t2tr]
    j1te = [jamo(s) for s in t1te]
    j2te = [jamo(s) for s in t2te]
    A1, A2, B1, B2 = tfidf_block(j1tr, j2tr, j1te, j2te,
                                 dict(analyzer="char", ngram_range=(2, 4),
                                      sublinear_tf=True, min_df=2))
    ftr_list.append(sparse_pair_feats(A1, A2))
    fte_list.append(sparse_pair_feats(B1, B2))

    hf_tr = hand_features(train_df)
    hf_te = hand_features(test_df)

    X_tr = np.hstack([hf_tr] + ftr_list)
    X_te = np.hstack([hf_te] + fte_list)
    return X_tr, X_te


def run_cv_model(fac, X, y, X_te, kf, n_splits):
    oof = np.zeros(len(y))
    test_pred = np.zeros(X_te.shape[0])
    for tr_idx, va_idx in kf.split(X):
        m = fac()
        m.fit(X[tr_idx], y[tr_idx])
        oof[va_idx] = m.predict(X[va_idx])
        test_pred += m.predict(X_te) / n_splits
    return oof, test_pred


def main():
    train_df = pd.read_csv(TRAIN)
    test_df = pd.read_csv(TEST)
    for col in ["sentence1", "sentence2"]:
        train_df[col] = train_df[col].astype(str)
        test_df[col] = test_df[col].astype(str)
    y = train_df["score"].values.astype(float)

    X_tr, X_te = build_features(train_df, test_df)
    print(f"Feature dim: {X_tr.shape[1]}", file=sys.stderr)

    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RNG)

    # scale features for distance-based models
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    base_factories = {
        "ridge": lambda: Ridge(alpha=2.0, random_state=RNG),
        "svr1": lambda: SVR(C=2.0, kernel="rbf", gamma="scale"),
        "svr2": lambda: SVR(C=5.0, kernel="rbf", gamma="scale"),
        "gbr": lambda: GradientBoostingRegressor(
            n_estimators=500, max_depth=3, learning_rate=0.04,
            subsample=0.8, random_state=RNG),
        "rf": lambda: RandomForestRegressor(
            n_estimators=500, max_depth=None, min_samples_leaf=2,
            n_jobs=-1, random_state=RNG),
        "et": lambda: ExtraTreesRegressor(
            n_estimators=500, max_depth=None, min_samples_leaf=2,
            n_jobs=-1, random_state=RNG),
        "hgb": lambda: HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.05, max_depth=None,
            max_leaf_nodes=31, l2_regularization=1.0, random_state=RNG),
    }

    # which models use scaled features
    scaled_models = {"svr1", "svr2", "ridge"}

    oof_dict, test_pred_dict = {}, {}
    for name, fac in base_factories.items():
        X_use = X_tr_s if name in scaled_models else X_tr
        X_te_use = X_te_s if name in scaled_models else X_te
        oof, tp = run_cv_model(fac, X_use, y, X_te_use, kf, n_splits)
        oof_dict[name] = oof
        test_pred_dict[name] = tp
        print(f"[CV] {name}: pearson={pearsonr(oof, y)[0]:.4f}", file=sys.stderr)

    names = list(base_factories.keys())
    oof_avg = np.mean([oof_dict[n] for n in names], axis=0)
    test_avg = np.mean([test_pred_dict[n] for n in names], axis=0)
    print(f"[CV] avg ensemble: pearson={pearsonr(oof_avg, y)[0]:.4f}", file=sys.stderr)

    meta_X = np.column_stack([oof_dict[n] for n in names])
    meta_X_te = np.column_stack([test_pred_dict[n] for n in names])

    # CV the meta across several alphas; pick best
    best_meta = ("ridge", 0.0, None)
    for alpha in [0.5, 1.0, 2.0, 5.0, 10.0]:
        oof_meta = np.zeros(len(y))
        for tr_idx, va_idx in kf.split(meta_X):
            mr = Ridge(alpha=alpha, random_state=RNG)
            mr.fit(meta_X[tr_idx], y[tr_idx])
            oof_meta[va_idx] = mr.predict(meta_X[va_idx])
        sc = pearsonr(oof_meta, y)[0]
        if sc > best_meta[1]:
            best_meta = ("ridge", sc, alpha)
        print(f"[CV] stack(ridge a={alpha}): pearson={sc:.4f}", file=sys.stderr)

    for alpha in [0.001, 0.003, 0.005, 0.01]:
        oof_meta_l = np.zeros(len(y))
        for tr_idx, va_idx in kf.split(meta_X):
            mr = Lasso(alpha=alpha, random_state=RNG, positive=True, max_iter=20000)
            mr.fit(meta_X[tr_idx], y[tr_idx])
            oof_meta_l[va_idx] = mr.predict(meta_X[va_idx])
        sc = pearsonr(oof_meta_l, y)[0]
        if sc > best_meta[1]:
            best_meta = ("lasso+", sc, alpha)
        print(f"[CV] stack(lasso+ a={alpha}): pearson={sc:.4f}", file=sys.stderr)

    meta_score = best_meta[1]
    print(f"[BEST META] {best_meta[0]} a={best_meta[2]}: {meta_score:.4f}", file=sys.stderr)

    # choose best
    scores = {
        "avg": pearsonr(oof_avg, y)[0],
        "stack": meta_score,
    }
    for n in names:
        scores[n] = pearsonr(oof_dict[n], y)[0]
    best_name = max(scores, key=scores.get)
    print(f"[BEST] {best_name}: {scores[best_name]:.4f}", file=sys.stderr)

    if best_name == "avg":
        pred = test_avg
    elif best_name == "stack":
        if best_meta[0] == "ridge":
            mr = Ridge(alpha=best_meta[2], random_state=RNG)
        else:
            mr = Lasso(alpha=best_meta[2], random_state=RNG, positive=True,
                       max_iter=20000)
        mr.fit(meta_X, y)
        pred = mr.predict(meta_X_te)
    else:
        pred = test_pred_dict[best_name]

    # safety: avg as fallback if it's the highest
    if scores["avg"] >= scores[best_name]:
        pred = test_avg
        print("[USE] avg (highest CV)", file=sys.stderr)
    else:
        print(f"[USE] {best_name}", file=sys.stderr)

    pred = np.clip(pred, 0.0, 5.0)

    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    out = pd.DataFrame({"id": test_df["id"], "score": pred})
    out.to_csv(SUB, index=False)
    print(f"Saved {SUB} with {len(out)} rows", file=sys.stderr)


if __name__ == "__main__":
    main()

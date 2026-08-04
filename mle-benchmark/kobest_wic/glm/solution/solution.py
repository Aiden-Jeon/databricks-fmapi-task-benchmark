"""Final WiC solution: word-conditional kNN sense-vote features + Logistic
Regression.

Pipeline:
1) Build context vectors (TF-IDF char n-grams) of a window of ~10 chars on each
   side of the target word (target removed) - this captures the "semantic
   field" of each context.
2) For each row (cross-validated for train rows), find other training pairs of
   the SAME word. Compute pair-level similarity accounting for orientation
   (c1-c1/c2-c2 vs c1-c2/c2-c1). Aggregate similarities weighted by the
   partner's label -> kNN sense vote features.
3) Train a tuned LogisticRegression on the dense kNN features and predict.
"""
import os
import re
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics.pairwise import linear_kernel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def clean(s):
    return re.sub(r"\[([^\]]+)\]", r"\1", str(s))


def target_span(word, ctx):
    s = str(ctx)
    m = re.search(r"\[" + re.escape(word) + r"\]", s)
    if m:
        return m.start(), m.end(), s
    m = re.search(r"(?<![가-힣A-Za-z0-9])" + re.escape(word) + r"(?![가-힣A-Za-z0-9])", s)
    if m:
        return m.start(), m.end(), s
    m = re.search(re.escape(word), s)
    if m:
        return m.start(), m.end(), s
    return -1, -1, s


def collocate(word, ctx, before=10, after=10):
    """Window of chars around the target word (target removed)."""
    st, en, s = target_span(word, ctx)
    if st < 0:
        return ""
    return s[max(0, st - before): st] + " " + s[en: en + after]


def diag_sim(A, B):
    return np.asarray(A.multiply(B).sum(axis=1)).ravel()


N_KNN_FEAT = 14


def _fill_row(F, i, s1, s2, labs, within, cand):
    sim = np.maximum(s1, s2)
    pm = labs == 1
    nm = labs == 0
    F[i, 0] = (sim * pm).sum()
    F[i, 1] = (sim * nm).sum()
    F[i, 2] = pm.sum()
    F[i, 3] = nm.sum()
    F[i, 4] = sim[pm].max() if pm.any() else 0
    F[i, 5] = sim[nm].max() if nm.any() else 0
    F[i, 6] = F[i, 4] - F[i, 5]
    F[i, 7] = sim[pm].mean() if pm.any() else 0
    F[i, 8] = sim[nm].mean() if nm.any() else 0
    F[i, 9] = ((sim ** 2) * pm).sum() - ((sim ** 2) * nm).sum()
    F[i, 10] = within
    F[i, 11] = s1.max() - s2.max()
    pos_star = int(np.argmax(sim))
    F[i, 12] = labs[pos_star]
    top3_idx = np.argsort(sim)[-3:]
    F[i, 13] = labs[top3_idx].mean()


def compute_knn_oof(V1c1, V1c2, y_train, train_words, within_tr, n_folds=5, seed=0):
    """Cross-validated kNN features for train rows (avoid leakage)."""
    n_tr = V1c1.shape[0]
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    F = np.zeros((n_tr, N_KNN_FEAT))
    for tr, va in skf.split(np.zeros(n_tr), y_train):
        tr = np.array(tr)
        va = np.array(va)
        by_word = defaultdict(list)
        for j in tr:
            by_word[train_words[j]].append(j)
        tr_idx_to_pos = {j: pos for pos, j in enumerate(tr)}
        s_c1 = linear_kernel(V1c1[va], V1c1[tr])
        s_c2 = linear_kernel(V1c2[va], V1c2[tr])
        s_c1c2 = linear_kernel(V1c1[va], V1c2[tr])
        s_c2c1 = linear_kernel(V1c2[va], V1c1[tr])
        for k, i in enumerate(va):
            w = train_words[i]
            cand = by_word.get(w, None)
            if not cand:
                cand = list(tr)
            pos = np.array([tr_idx_to_pos[j] for j in cand])
            s1 = s_c1[k, pos] + s_c2[k, pos]
            s2 = s_c1c2[k, pos] + s_c2c1[k, pos]
            labs = y_train[cand]
            _fill_row(F, i, s1, s2, labs, within_tr[i], cand)
    return F


def compute_knn_for_test(V1c1_tr, V1c2_tr, V1c1_te, V1c2_te, y_train, train_words, test_words, within_te):
    by_word = defaultdict(list)
    for i, w in enumerate(train_words):
        by_word[w].append(i)
    n_tr = V1c1_tr.shape[0]
    n_te = len(test_words)
    s_c1 = linear_kernel(V1c1_te, V1c1_tr)
    s_c2 = linear_kernel(V1c2_te, V1c2_tr)
    s_c1c2 = linear_kernel(V1c1_te, V1c2_tr)
    s_c2c1 = linear_kernel(V1c2_te, V1c1_tr)
    F = np.zeros((n_te, N_KNN_FEAT))
    for i in range(n_te):
        w = test_words[i]
        cand = by_word.get(w, None)
        if not cand:
            cand = list(range(n_tr))
        s1 = s_c1[i, cand] + s_c2[i, cand]
        s2 = s_c1c2[i, cand] + s_c2c1[i, cand]
        labs = y_train[cand]
        _fill_row(F, i, s1, s2, labs, within_te[i], cand)
    return F


def main():
    train = pd.read_csv(os.path.join(ROOT, "train.csv"))
    test = pd.read_csv(os.path.join(ROOT, "test.csv"))
    train["c1"] = train["context_1"].map(clean)
    train["c2"] = train["context_2"].map(clean)
    test["c1"] = test["context_1"].map(clean)
    test["c2"] = test["context_2"].map(clean)
    y = train["label"].values
    n_tr, n_te = len(train), len(test)

    train["k1"] = train.apply(lambda r: collocate(r["word"], r["c1"]), axis=1)
    train["k2"] = train.apply(lambda r: collocate(r["word"], r["c2"]), axis=1)
    test["k1"] = test.apply(lambda r: collocate(r["word"], r["c1"]), axis=1)
    test["k2"] = test.apply(lambda r: collocate(r["word"], r["c2"]), axis=1)

    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), min_df=1, sublinear_tf=True)
    Vc1 = v.fit_transform(list(train["k1"]) + list(test["k1"]))
    Vc2 = v.transform(list(train["k2"]) + list(test["k2"]))
    V1c1 = Vc1[:n_tr]
    V1c1t = Vc1[n_tr:]
    V1c2 = Vc2[:n_tr]
    V1c2t = Vc2[n_tr:]

    within_tr = diag_sim(V1c1, V1c2)
    within_te = diag_sim(V1c1t, V1c2t)

    train_words = train["word"].values
    test_words = test["word"].values

    Ftr = compute_knn_oof(V1c1, V1c2, y, train_words, within_tr, n_folds=5, seed=0)
    Fte = compute_knn_for_test(V1c1, V1c2, V1c1t, V1c2t, y, train_words, test_words, within_te)

    # CV of kNN-only to pick C (select among a robust range)
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    best_knn_score = -1
    best_knn_C = 1000.0
    for C in [100.0, 1000.0, 10000.0]:
        clf = LogisticRegression(C=C, max_iter=3000, random_state=0)
        scores = cross_val_score(clf, Ftr, y, cv=skf, scoring="accuracy")
        print(f"KNN-only LogReg C={C} CV={scores.mean():.4f} std={scores.std():.4f}")
        if scores.mean() > best_knn_score:
            best_knn_score = scores.mean()
            best_knn_C = C

    print(f"Using KNN-only model C={best_knn_C} (CV={best_knn_score:.4f})")
    clf = LogisticRegression(C=best_knn_C, max_iter=3000, random_state=0)
    clf.fit(Ftr, y)
    pred = clf.predict(Fte)

    out = pd.DataFrame({"id": test["id"], "label": pred})
    out.to_csv(os.path.join(ROOT, "outputs", "submission.csv"), index=False)
    print("Saved submission. dist:", out["label"].value_counts().to_dict())


if __name__ == "__main__":
    main()

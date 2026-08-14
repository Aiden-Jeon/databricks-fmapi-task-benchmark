"""Train PAWS-X ko paraphrase classifier and produce predictions.

Approach:
  1. Engineered similarity features (token/char overlap, order/movement, LCS, digits).
  2. Text-based stack features: OOF probabilities from
       - TF-IDF (word 1-2gram) diff/product features -> LogisticRegression
       - shared/only-token indicators -> LogisticRegression
  3. Final model: multi-seed HistGradientBoosting ensemble on [engineered + stack] features.
  4. Threshold at 0.5 (also searched on OOF).
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import build_features

RANDOM_STATE = 42
N_SPLITS = 5


def load_data(base):
    tr = pd.read_csv(os.path.join(base, "train.csv")).fillna("")
    te = pd.read_csv(os.path.join(base, "test.csv")).fillna("")
    return tr, te


def text_stack_features(tr_s1, tr_s2, te_s1, te_s2, y, skf):
    """Return OOF and test probabilities from two text models."""
    n_tr, n_te = len(tr_s1), len(te_s1)
    s1 = tr_s1.tolist() + te_s1.tolist()
    s2 = tr_s2.tolist() + te_s2.tolist()

    # --- Model T1: TF-IDF word diff/product
    tv = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60000,
                         sublinear_tf=True)
    T = tv.fit_transform(s1 + s2)
    T1, T2 = T[:len(s1)], T[len(s1):]
    Xt = hstack([abs(T1 - T2), T1.multiply(T2)]).tocsr()
    Xt_tr, Xt_te = Xt[:n_tr], Xt[n_tr:]

    # --- Model T2: shared/only-token indicators
    cvv = CountVectorizer(ngram_range=(1, 1), min_df=3, max_features=30000, binary=True)
    vocab = cvv.fit([a + " " + b for a, b in zip(s1, s2)]).vocabulary_
    c1 = CountVectorizer(vocabulary=vocab, binary=True).fit_transform(s1)
    c2 = CountVectorizer(vocabulary=vocab, binary=True).fit_transform(s2)
    S = c1.minimum(c2)
    Xs = hstack([S, -(c1 - S), -(c2 - S)]).tocsr()
    Xs_tr, Xs_te = Xs[:n_tr], Xs[n_tr:]

    oof_t1 = np.zeros(n_tr); te_t1 = np.zeros(n_te)
    oof_t2 = np.zeros(n_tr); te_t2 = np.zeros(n_te)
    for itr, iva in skf.split(Xt_tr, y):
        m = LogisticRegression(C=1.0, max_iter=2000, solver="liblinear")
        m.fit(Xt_tr[itr], y[itr])
        oof_t1[iva] = m.predict_proba(Xt_tr[iva])[:, 1]
        te_t1 += m.predict_proba(Xt_te)[:, 1] / N_SPLITS

        m2 = LogisticRegression(C=0.1, max_iter=3000, solver="liblinear")
        m2.fit(Xs_tr[itr], y[itr])
        oof_t2[iva] = m2.predict_proba(Xs_tr[iva])[:, 1]
        te_t2 += m2.predict_proba(Xs_te)[:, 1] / N_SPLITS

    print("  T1 (tfidf diff/prod) OOF acc:", accuracy_score(y, (oof_t1 >= 0.5).astype(int)))
    print("  T2 (shared/only)     OOF acc:", accuracy_score(y, (oof_t2 >= 0.5).astype(int)))
    return (oof_t1, te_t1), (oof_t2, te_t2)


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tr, te = load_data(base)
    y = tr["label"].values.astype(int)

    print("Building engineered features...")
    Ftr, s1tr, s2tr = build_features(tr)
    Fte, s1te, s2te = build_features(te)
    fcols = Ftr.columns.tolist()
    Xf_tr = Ftr[fcols].values
    Xf_te = Fte[fcols].values
    print("  engineered:", Xf_tr.shape)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    print("Building text stack features...")
    (oof_t1, te_t1), (oof_t2, te_t2) = text_stack_features(s1tr, s2tr, s1te, s2te, y, skf)

    X_tr = np.hstack([Xf_tr, oof_t1[:, None], oof_t2[:, None]])
    X_te = np.hstack([Xf_te, te_t1[:, None], te_t2[:, None]])

    # --- Final: multi-seed HistGB ensemble
    seeds = [0, 1, 2, 7, 13]
    oof = np.zeros(len(tr))
    te_p = np.zeros(len(te))
    for seed in seeds:
        o = np.zeros(len(tr))
        t = np.zeros(len(te))
        skf2 = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for itr, iva in skf2.split(X_tr, y):
            m = HistGradientBoostingClassifier(
                max_iter=600, learning_rate=0.05, max_leaf_nodes=63,
                l2_regularization=1.0, random_state=seed)
            m.fit(X_tr[itr], y[itr])
            o[iva] = m.predict_proba(X_tr[iva])[:, 1]
            t += m.predict_proba(X_te)[:, 1] / N_SPLITS
            acc = accuracy_score(y, (o >= 0.5).astype(int))
        print(f"  seed{seed} OOF acc={acc:.4f}")
        oof += o / len(seeds)
        te_p += t / len(seeds)

    acc05 = accuracy_score(y, (oof >= 0.5).astype(int))
    print("Ensemble OOF acc @0.5:", acc05)

    # threshold search on OOF
    best_t, best_acc = 0.5, acc05
    for t in np.arange(0.40, 0.60, 0.005):
        a = accuracy_score(y, (oof >= t).astype(int))
        if a > best_acc:
            best_acc, best_t = a, t
    print(f"Best OOF threshold {best_t:.3f} acc {best_acc:.4f}")

    pred = (te_p >= best_t).astype(int)
    sub = pd.DataFrame({"id": te["id"], "label": pred})
    out = os.path.join(base, "outputs", "submission.csv")
    sub.to_csv(out, index=False)
    print("Saved", out, sub.shape)
    print(sub["label"].value_counts())

    np.save(os.path.join(base, "outputs", "oof.npy"), oof)
    np.save(os.path.join(base, "outputs", "te_p.npy"), te_p)


if __name__ == "__main__":
    main()

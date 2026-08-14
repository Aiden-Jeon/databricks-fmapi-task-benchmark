"""KLUE-STS: TF-IDF (char + word) pair features + ensemble regression.

Self-contained: trains only on train.csv, predicts test.csv.
Run:  python solution/solution.py
"""
import os
import re
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.stats import pearsonr

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(BASE, "train.csv")
TEST = os.path.join(BASE, "test.csv")
OUT = os.path.join(BASE, "outputs", "submission.csv")

RANDOM_STATE = 42


def norm(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def char_ngrams(s: str, n: int = 3):
    s = norm(s).replace(" ", "")
    if len(s) < n:
        return {s}
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def word_set(s: str):
    return set(norm(s).split())


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def lcs_len(a: str, b: str) -> int:
    a, b = norm(a).split(), norm(b).split()
    if len(a) > 80 or len(b) > 80:
        return 0
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            tmp = dp[j]
            dp[j] = prev + 1 if x == y else max(dp[j], dp[j - 1])
            prev = tmp
    return dp[-1]


def cos_rows(A, B):
    num = np.asarray(A.multiply(B).sum(axis=1)).ravel()
    na = np.sqrt(np.asarray(A.multiply(A).sum(axis=1)).ravel())
    nb = np.sqrt(np.asarray(B.multiply(B).sum(axis=1)).ravel())
    return num / (na * nb + 1e-9)


def build_features(df, vec_c, vec_w, fit=False):
    """Returns (sparse_pair_matrix, dense_features)."""
    s1 = df["sentence1"].map(norm).tolist()
    s2 = df["sentence2"].map(norm).tolist()
    both = s1 + s2
    if fit:
        Xc = vec_c.fit_transform(both)
        Xw = vec_w.fit_transform(both)
    else:
        Xc = vec_c.transform(both)
        Xw = vec_w.transform(both)
    n = len(df)
    Xc1, Xc2 = Xc[:n], Xc[n:]
    Xw1, Xw2 = Xw[:n], Xw[n:]

    cos_c = cos_rows(Xc1, Xc2)
    cos_w = cos_rows(Xw1, Xw2)

    feats = []
    for a, b, cc, cw in zip(df["sentence1"], df["sentence2"], cos_c, cos_w):
        na_, nb_ = norm(a), norm(b)
        ca, cb = char_ngrams(a, 3), char_ngrams(b, 3)
        ca2, cb2 = char_ngrams(a, 2), char_ngrams(b, 2)
        wa, wb = word_set(a), word_set(b)
        la, lb = len(na_), len(nb_)
        wa_l, wb_l = na_.split(), nb_.split()
        l = lcs_len(a, b)
        f1 = 2 * l / (len(wa_l) + len(wb_l) + 1e-9)
        inter = len(wa & wb)
        feats.append([
            cc, cw, abs(cc - cw), cc * cw,
            jaccard(ca, cb), jaccard(ca2, cb2), jaccard(wa, wb),
            la, lb, abs(la - lb), min(la, lb) / (max(la, lb) + 1e-9),
            len(wa_l), len(wb_l),
            inter, inter / (len(wa) + 1e-9), inter / (len(wb) + 1e-9),
            f1, la / (lb + 1e-9),
        ])
    dense = np.array(feats, dtype=np.float64)
    # Add per-row TF-IDF norm statistics as extra dense features
    n1 = np.sqrt(np.asarray(Xc1.multiply(Xc1).sum(axis=1)).ravel())
    n2 = np.sqrt(np.asarray(Xc2.multiply(Xc2).sum(axis=1)).ravel())
    extra = np.column_stack([n1, n2, np.abs(n1 - n2)])
    dense = np.hstack([dense, extra])
    X = hstack([Xc1, Xc2, Xw1, Xw2, csr_matrix(dense)]).tocsr()
    return X, dense


def main():
    tr = pd.read_csv(TRAIN)
    te = pd.read_csv(TEST)
    y = tr["score"].values.astype(float)

    vec_c = TfidfVectorizer(analyzer="char", ngram_range=(2, 5),
                            min_df=2, max_features=60000, sublinear_tf=True)
    vec_w = TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                            min_df=2, max_features=40000, sublinear_tf=True,
                            token_pattern=r"(?u)\b\w+\b")

    Xtr, dense_tr = build_features(tr, vec_c, vec_w, fit=True)
    Xte, dense_te = build_features(te, vec_c, vec_w, fit=False)

    ridge = Ridge(alpha=3.0, random_state=RANDOM_STATE)
    gbm = HistGradientBoostingRegressor(
        max_iter=600, learning_rate=0.05,
        max_leaf_nodes=31, l2_regularization=1.0,
        random_state=RANDOM_STATE)

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    models = {"ridge": (ridge, "sparse"), "gbm": (gbm, "dense")}
    oof = {k: np.zeros(len(tr)) for k in models}
    pred = {k: np.zeros(len(te)) for k in models}

    for fold, (i_tr, i_va) in enumerate(kf.split(Xtr)):
        for name, (m, kind) in models.items():
            A = Xtr if kind == "sparse" else dense_tr
            B = Xte if kind == "sparse" else dense_te
            m.fit(A[i_tr], y[i_tr])
            oof[name][i_va] = m.predict(A[i_va])
            pred[name] += m.predict(B) / kf.n_splits
        msg = "  ".join(f"{k}={pearsonr(y[i_va], oof[k][i_va])[0]:.4f}"
                         for k in models)
        print(f"fold {fold}: {msg}", flush=True)

    # Blend weight tuned on OOF pearson
    keys = list(models)
    best_w, best_p = None, -1
    for w0 in np.arange(0, 1.01, 0.02):
        ws = [w0, 1 - w0]
        o = sum(w * oof[k] for w, k in zip(ws, keys))
        p = pearsonr(y, o)[0]
        if p > best_p:
            best_p, best_w = p, ws
    print(f"OOF blend: {dict(zip(keys, np.round(best_w, 3)))} pearson={best_p:.4f}")

    final = sum(w * pred[k] for w, k in zip(best_w, keys))
    final = np.clip(final, 0.0, 5.0)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    sub = pd.DataFrame({"id": te["id"], "score": final})
    assert sub["id"].is_unique and len(sub) == len(te)
    sub.to_csv(OUT, index=False)
    print(f"saved {OUT} ({len(sub)} rows)")


if __name__ == "__main__":
    main()

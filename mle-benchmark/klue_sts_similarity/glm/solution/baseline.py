"""Baseline KLUE-STS solution.

Approach (no external data / no pretrained weights):
  - TF-IDF (word + char n-grams) over the concatenation of sentence1 + sentence2.
  - For each pair, build features: cosine similarity between the two sentences'
    TF-IDF vectors (word-level and char-level), length-difference / ratio features,
    and Jaccard overlap of token sets.
  - Train a Ridge regression to predict the 0~5 score, then clip to [0,5].
"""
import re
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from scipy.spatial.distance import cosine
from sklearn.model_selection import KFold
from scipy.stats import pearsonr

DATA = "/tmp/kmle/M3_t7_klue_sts_full_20260804_033555/task"
OUT = "/tmp/kmle/M3_t7_klue_sts_full_20260804_033555/task/outputs"
RNG = 42


def normalize(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize(s: str):
    return re.findall(r"[가-힣a-zA-Z0-9]+", s)


def jaccard(a, b):
    sa, sb = set(tokenize(a)), set(tokenize(b))
    if not sa and not sb:
        return 1.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def char_ngram_overlap(a, b, n=3):
    def grams(s):
        s = re.sub(r"\s+", "", s)
        return set(s[i:i+n] for i in range(len(s)-n+1)) if len(s) >= n else {s}
    ga, gb = grams(a), grams(b)
    if not ga and not gb:
        return 1.0
    inter = len(ga & gb)
    union = len(ga | gb)
    return inter / union if union else 0.0


def hand_features(df):
    feats = []
    for s1, s2 in zip(df["sentence1"], df["sentence2"]):
        n1 = normalize(s1)
        n2 = normalize(s2)
        t1, t2 = tokenize(n1), tokenize(n2)
        len1, len2 = len(n1), len(n2)
        wc1, wc2 = len(t1), len(t2)
        feats.append([
            len1, len2,
            abs(len1 - len2),
            len1 / (len2 + 1e-9),
            wc1, wc2,
            abs(wc1 - wc2),
            wc1 / (wc2 + 1e-9),
            jaccard(n1, n2),
            char_ngram_overlap(n1, n2, 2),
            char_ngram_overlap(n1, n2, 3),
            char_ngram_overlap(n1, n2, 4),
            len(set(t1) & set(t2)),
            len(set(t1) | set(t2)),
        ])
    return np.array(feats, dtype=np.float32)


def cosine_sim(matrix, idx_a, idx_b):
    """Cosine similarity between rows idx_a and idx_b of a sparse matrix."""
    a = matrix[idx_a]
    b = matrix[idx_b]
    dot = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    na = np.sqrt(np.asarray(a.multiply(a).sum(axis=1)).ravel())
    nb = np.sqrt(np.asarray(b.multiply(b).sum(axis=1)).ravel())
    denom = na * nb
    out = np.zeros_like(dot)
    nz = denom > 0
    out[nz] = dot[nz] / denom[nz]
    return out


def build_tfidf(train_df, test_df, analyzer, ngram_range, max_features):
    texts = list(train_df["sentence1"]) + list(train_df["sentence2"]) + \
           list(test_df["sentence1"]) + list(test_df["sentence2"])
    texts = [normalize(t) for t in texts]
    vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range,
                          max_features=max_features, sublinear_tf=True,
                          min_df=2)
    M = vec.fit_transform(texts)
    n_tr = len(train_df)
    n_te = len(test_df)
    # indices: tr1 [0:n_tr], tr2 [n_tr:2n_tr], te1 [2n_tr:2n_tr+n_te], te2 [2n_tr+n_te:]
    return vec, M, n_tr, n_te


def similarity_features(M, n_tr, n_te):
    tr_a = slice(0, n_tr)
    tr_b = slice(n_tr, 2 * n_tr)
    te_a = slice(2 * n_tr, 2 * n_tr + n_te)
    te_b = slice(2 * n_tr + n_te, 2 * n_tr + 2 * n_te)
    tr_sim = cosine_sim(M, tr_a, tr_b)
    te_sim = cosine_sim(M, te_a, te_b)
    return tr_sim.reshape(-1, 1), te_sim.reshape(-1, 1)


def main():
    train_df = pd.read_csv(f"{DATA}/train.csv")
    test_df = pd.read_csv(f"{DATA}/test.csv")

    # Hand features
    hf_tr = hand_features(train_df)
    hf_te = hand_features(test_df)

    # Word TF-IDF cosine sim
    _, Mw, n_tr, n_te = build_tfidf(train_df, test_df, "word", (1, 2), 30000)
    wsim_tr, wsim_te = similarity_features(Mw, n_tr, n_te)

    # Char TF-IDF cosine sim
    _, Mc, _, _ = build_tfidf(train_df, test_df, "char_wb", (3, 5), 30000)
    csim_tr, csim_te = similarity_features(Mc, n_tr, n_te)

    X_tr = np.hstack([hf_tr, wsim_tr, csim_tr])
    X_te = np.hstack([hf_te, wsim_te, csim_te])

    y = train_df["score"].values.astype(np.float32)

    # CV for sanity
    kf = KFold(n_splits=5, shuffle=True, random_state=RNG)
    preds_oof = np.zeros(len(y))
    for tr_idx, va_idx in kf.split(X_tr):
        m = Ridge(alpha=10.0)
        m.fit(X_tr[tr_idx], y[tr_idx])
        preds_oof[va_idx] = m.predict(X_tr[va_idx])
    r, _ = pearsonr(preds_oof, y)
    print(f"CV Pearson: {r:.4f}")

    # Fit on all and predict
    model = Ridge(alpha=10.0)
    model.fit(X_tr, y)
    pred = model.predict(X_te)
    pred = np.clip(pred, 0.0, 5.0)

    sub = pd.DataFrame({"id": test_df["id"], "score": pred})
    sub.to_csv(f"{OUT}/submission.csv", index=False)
    print("Saved submission. rows:", len(sub))
    print(sub.head())


if __name__ == "__main__":
    main()

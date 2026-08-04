"""Improved KLUE-STS solution exploration.

Strategy:
 - Build a shared TF-IDF vocabulary (word + char ngrams) from training sentences.
 - For each pair, compute tfidf(s1), tfidf(s2).
 - Features = [tfidf(s1), tfidf(s2), |tfidf(s1)-tfidf(s2)|, hand features, cosine sims].
 - Train Ridge with CV; report Pearson.
"""
import re
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from scipy.stats import pearsonr

DATA = "/tmp/kmle/M3_t7_klue_sts_full_20260804_033555/task"
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
            len1, len2, abs(len1 - len2), len1 / (len2 + 1e-9),
            wc1, wc2, abs(wc1 - wc2), wc1 / (wc2 + 1e-9),
            jaccard(n1, n2),
            char_ngram_overlap(n1, n2, 2),
            char_ngram_overlap(n1, n2, 3),
            char_ngram_overlap(n1, n2, 4),
            len(set(t1) & set(t2)), len(set(t1) | set(t2)),
        ])
    return np.array(feats, dtype=np.float32)


def cosine_from_sparse(M):
    norm = np.sqrt(np.asarray(M.multiply(M).sum(axis=1)).ravel())
    norm[norm == 0] = 1.0
    return M.multiply(1.0 / norm[:, None])


def main():
    train_df = pd.read_csv(f"{DATA}/train.csv")
    test_df = pd.read_csv(f"{DATA}/test.csv")

    hf_tr = hand_features(train_df)
    hf_te = hand_features(test_df)

    # Word TF-IDF
    all_text = [normalize(t) for t in list(train_df["sentence1"]) + list(train_df["sentence2"])]
    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=20000,
                                sublinear_tf=True, min_df=2, token_pattern=r"[가-힣a-zA-Z0-9]+")
    word_vec.fit(all_text)

    tr1_w = word_vec.transform([normalize(s) for s in train_df["sentence1"]])
    tr2_w = word_vec.transform([normalize(s) for s in train_df["sentence2"]])
    te1_w = word_vec.transform([normalize(s) for s in test_df["sentence1"]])
    te2_w = word_vec.transform([normalize(s) for s in test_df["sentence2"]])

    # Char TF-IDF
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=20000,
                                sublinear_tf=True, min_df=2)
    char_vec.fit(all_text)
    tr1_c = char_vec.transform([normalize(s) for s in train_df["sentence1"]])
    tr2_c = char_vec.transform([normalize(s) for s in train_df["sentence2"]])
    te1_c = char_vec.transform([normalize(s) for s in test_df["sentence1"]])
    te2_c = char_vec.transform([normalize(s) for s in test_df["sentence2"]])

    # Cosine sims
    def cosine_sim(a, b):
        na = np.sqrt(np.asarray(a.multiply(a).sum(axis=1)).ravel())
        nb = np.sqrt(np.asarray(b.multiply(b).sum(axis=1)).ravel())
        dot = np.asarray(a.multiply(b).sum(axis=1)).ravel()
        denom = na * nb
        out = np.zeros_like(dot)
        nz = denom > 0
        out[nz] = dot[nz] / denom[nz]
        return out.reshape(-1, 1)

    wsim_tr = cosine_sim(tr1_w, tr2_w)
    wsim_te = cosine_sim(te1_w, te2_w)
    csim_tr = cosine_sim(tr1_c, tr2_c)
    csim_te = cosine_sim(te1_c, te2_c)

    # Build full feature set
    def build(a, b, hf, wsim, csim):
        diff = a - b  # sparse
        return hstack([a, b, diff, csr_matrix(hf), csr_matrix(wsim), csr_matrix(csim)]).tocsr()

    X_tr = build(tr1_w, tr2_w, hf_tr, wsim_tr, csim_tr)
    X_te = build(te1_w, te2_w, hf_te, wsim_te, csim_te)

    y = train_df["score"].values.astype(np.float32)

    kf = KFold(n_splits=5, shuffle=True, random_state=RNG)
    oof = np.zeros(len(y))
    for tr_idx, va_idx in kf.split(X_tr):
        m = Ridge(alpha=5.0)
        m.fit(X_tr[tr_idx], y[tr_idx])
        oof[va_idx] = m.predict(X_tr[va_idx])
    r, _ = pearsonr(oof, y)
    print(f"Word-only Ridge CV Pearson: {r:.4f}")

    X_tr_c = build(tr1_c, tr2_c, hf_tr, wsim_tr, csim_tr)
    X_te_c = build(te1_c, te2_c, hf_te, wsim_te, csim_te)
    oof2 = np.zeros(len(y))
    for tr_idx, va_idx in kf.split(X_tr_c):
        m = Ridge(alpha=5.0)
        m.fit(X_tr_c[tr_idx], y[tr_idx])
        oof2[va_idx] = m.predict(X_tr_c[va_idx])
    r2, _ = pearsonr(oof2, y)
    print(f"Char-only Ridge CV Pearson: {r2:.4f}")

    # Ensemble
    oof3 = 0.5 * oof + 0.5 * oof2
    r3, _ = pearsonr(oof3, y)
    print(f"Ensemble CV Pearson: {r3:.4f}")

    # Try different alphas on word
    for alpha in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
        oof_a = np.zeros(len(y))
        for tr_idx, va_idx in kf.split(X_tr):
            m = Ridge(alpha=alpha)
            m.fit(X_tr[tr_idx], y[tr_idx])
            oof_a[va_idx] = m.predict(X_tr[va_idx])
        ra, _ = pearsonr(oof_a, y)
        print(f"  word alpha={alpha}: {ra:.4f}")


if __name__ == "__main__":
    main()

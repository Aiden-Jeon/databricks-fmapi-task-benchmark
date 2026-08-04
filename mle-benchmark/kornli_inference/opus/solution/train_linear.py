"""Sparse feature-based linear model for KorNLI.

Features (all learned from train.csv):
  * hypothesis word 1/2-grams and stem unigrams  (hypothesis-only bias)
  * premise word unigrams
  * "unmatched" hypothesis words (present in s2 but not in s1)  -> strong cue
  * hashed cross word-pairs (premise word x hypothesis word)
  * char n-gram tf-idf of both sentences and their element-wise product
  * dense lexical-overlap / negation features

Writes work/lin_val_<tag>.npy, work/lin_test_<tag>.npy, work/lin_full_test.npy
"""
import argparse, os, sys, time
import numpy as np
import scipy.sparse as sp
from zlib import crc32

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import L2I, tokens, norm, stem, pair_features, load

CROSS_BITS = 20
CROSS_DIM = 1 << CROSS_BITS


def h(s):
    return crc32(s.encode("utf8")) & (CROSS_DIM - 1)


def cross_matrix(s1, s2, max_p=24, max_h=16):
    indptr = [0]; indices = []
    for a, b in zip(s1, s2):
        ta = [stem(w) for w in tokens(a)[:max_p]]
        tb = [stem(w) for w in tokens(b)[:max_h]]
        sa = set(ta)
        row = set()
        for wb in tb:
            unm = wb not in sa
            for wa in ta:
                row.add(h(wa + "|" + wb))
            row.add(h(("U#" if unm else "M#") + wb))
        indices.extend(row)
        indptr.append(len(indices))
    data = np.ones(len(indices), dtype=np.float32)
    return sp.csr_matrix((data, np.asarray(indices), np.asarray(indptr)),
                         shape=(len(s1), CROSS_DIM))


def unmatched_text(s1, s2):
    """words of the hypothesis that do not appear in the premise"""
    out = []
    for a, b in zip(s1, s2):
        sa = set(stem(w) for w in tokens(a))
        out.append(" ".join(w for w in tokens(b) if stem(w) not in sa))
    return np.asarray(out, dtype=object)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="a")
    ap.add_argument("--C", type=float, default=0.7)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    os.environ["OMP_NUM_THREADS"] = str(a.threads)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    t0 = time.time()
    tr, te = load()
    y = tr.label.map(L2I).values
    s1, s2 = tr.sentence1.values, tr.sentence2.values
    t1, t2 = te.sentence1.values, te.sentence2.values

    n1 = np.array([norm(x) for x in s1], dtype=object)
    n2 = np.array([norm(x) for x in s2], dtype=object)
    m1 = np.array([norm(x) for x in t1], dtype=object)
    m2 = np.array([norm(x) for x in t2], dtype=object)
    u_tr, u_te = unmatched_text(s1, s2), unmatched_text(t1, t2)

    v2 = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(n2)
    v1 = TfidfVectorizer(ngram_range=(1, 1), min_df=3, sublinear_tf=True).fit(n1)
    vu = TfidfVectorizer(ngram_range=(1, 1), min_df=2, sublinear_tf=True).fit(u_tr)
    vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=5,
                         sublinear_tf=True, max_features=200000).fit(
        np.concatenate([n1, n2]))

    def make(a1, a2, ut, raw1, raw2):
        C1, C2 = vc.transform(a1), vc.transform(a2)
        D = pair_features(raw1, raw2)
        return sp.hstack([
            v1.transform(a1), v2.transform(a2), vu.transform(ut),
            cross_matrix(raw1, raw2),
            C1.multiply(C2), abs(C1 - C2),
            sp.csr_matrix(D), sp.csr_matrix(D ** 2),
        ], format="csr", dtype=np.float32)

    X = make(n1, n2, u_tr, s1, s2)
    T = make(m1, m2, u_te, t1, t2)
    print("X", X.shape, X.nnz / X.shape[0], f"{time.time()-t0:.0f}s", flush=True)

    skf = StratifiedKFold(10, shuffle=True, random_state=7)
    tri, vai = next(iter(skf.split(y, y)))
    clf = LogisticRegression(C=a.C, max_iter=800, n_jobs=a.threads)
    clf.fit(X[tri], y[tri])
    pv = clf.predict_proba(X[vai])
    print("val acc", (pv.argmax(1) == y[vai]).mean(), f"{time.time()-t0:.0f}s", flush=True)
    np.save(f"work/lin_val_{a.tag}.npy", pv)
    np.save(f"work/lin_test_{a.tag}.npy", clf.predict_proba(T))
    np.save("work/lin_valid_idx.npy", vai)

    clf.fit(X, y)
    np.save(f"work/lin_fulltest_{a.tag}.npy", clf.predict_proba(T))
    print("full done", f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

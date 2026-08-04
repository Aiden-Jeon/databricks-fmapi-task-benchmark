"""Feature engineering for Korean STS (KLUE-STS) using only sklearn/scipy/numpy.

All features are symmetric w.r.t. (s1, s2) swap so the model does not need
order augmentation.
"""
import re
import unicodedata
from difflib import SequenceMatcher

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# ----------------------------------------------------------------------------- text utils
_PUNCT = re.compile(r"[^0-9A-Za-z\uac00-\ud7a3\u3131-\u318e ]+")
_WS = re.compile(r"\s+")

# frequent Korean particles / endings to strip for a crude stemmer
_SUFFIXES = [
    "에서는", "에서도", "으로는", "이라고", "라고", "에서", "으로", "부터", "까지", "에게",
    "한테", "처럼", "보다", "만큼", "이나", "든지", "이다", "입니다", "습니다", "ㅂ니다",
    "예요", "이에요", "에요", "해요", "네요", "겠다", "는데", "지만", "면서",
    "은", "는", "이", "가", "을", "를", "에", "의", "와", "과", "도", "만", "로", "고", "다",
]


def clean(s):
    s = unicodedata.normalize("NFKC", str(s))
    s = s.lower()
    return _WS.sub(" ", s).strip()


def depunct(s):
    return _WS.sub(" ", _PUNCT.sub(" ", s)).strip()


def tokens(s):
    return depunct(s).split()


def stem(t):
    for suf in _SUFFIXES:
        if len(t) > len(suf) + 1 and t.endswith(suf):
            return t[: -len(suf)]
    return t


def stems(s):
    return [stem(t) for t in tokens(s)]


def ngrams(s, n):
    s = depunct(s).replace(" ", "")
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}


def jac(a, b):
    if not a and not b:
        return 1.0, 1.0, 1.0
    inter = len(a & b)
    uni = len(a | b)
    mn = min(len(a), len(b)) or 1
    mx = max(len(a), len(b)) or 1
    return inter / uni if uni else 0.0, inter / mn, inter / mx


def lcs_len(a, b):
    """Length of longest common subsequence (char level), O(len(a)*len(b))."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0]
        for j, cb in enumerate(b):
            cur.append(prev[j] + 1 if ca == cb else max(cur[j], prev[j + 1]))
        prev = cur
    return prev[-1]


NUM = re.compile(r"\d+")


# ----------------------------------------------------------------------------- vectorizer bank
VEC_SPECS = [
    ("cwb2", dict(analyzer="char_wb", ngram_range=(2, 2), min_df=2, sublinear_tf=True)),
    ("cwb23", dict(analyzer="char_wb", ngram_range=(2, 3), min_df=2, sublinear_tf=True)),
    ("cwb24", dict(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)),
    ("cwb35", dict(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
    ("c15", dict(analyzer="char", ngram_range=(1, 5), min_df=2, sublinear_tf=True)),
    ("w1", dict(analyzer="word", ngram_range=(1, 1), min_df=1, sublinear_tf=True)),
    ("w12", dict(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
    ("w1b", dict(analyzer="word", ngram_range=(1, 1), min_df=1, binary=True, use_idf=False, norm="l2")),
]


class STSFeaturizer:
    """Unsupervised (label-free) featurizer; fit on the full sentence corpus."""

    def __init__(self, svd_dim=256, svd_out=96, random_state=0):
        self.svd_dim = svd_dim
        self.svd_out = svd_out
        self.random_state = random_state

    def fit(self, sentences):
        raw = [clean(s) for s in sentences]
        stemmed = [" ".join(stems(s)) for s in raw]
        self.vecs = {}
        for name, kw in VEC_SPECS:
            v = TfidfVectorizer(**kw)
            src = stemmed if name.startswith("w") else raw
            v.fit(src)
            self.vecs[name] = v
        # LSA spaces
        self.svds = {}
        for name in ("cwb24", "w12"):
            src = stemmed if name.startswith("w") else raw
            X = normalize(self.vecs[name].transform(src))
            s = TruncatedSVD(self.svd_dim, random_state=self.random_state)
            s.fit(X)
            self.svds[name] = s
        # token-level char-ngram space for greedy alignment + idf lookup
        self.tokvec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
        self.tokvec.fit(raw)
        wv = self.vecs["w1"]
        self.idf = dict(zip(wv.get_feature_names_out(), wv.idf_))
        self.max_idf = float(wv.idf_.max())
        return self

    # -------------------------------------------------------------- helpers
    def _tf(self, name, texts):
        return normalize(self.vecs[name].transform(texts))

    def _align(self, t1, t2):
        """IDF-weighted greedy soft alignment on char-ngram token vectors."""
        if not t1 or not t2:
            return 0.0, 0.0, 0.0
        A = normalize(self.tokvec.transform(t1))
        B = normalize(self.tokvec.transform(t2))
        S = (A @ B.T).toarray()
        w1 = np.array([self.idf.get(stem(t), self.max_idf) for t in t1])
        w2 = np.array([self.idf.get(stem(t), self.max_idf) for t in t2])
        m1 = S.max(axis=1)
        m2 = S.max(axis=0)
        a1 = float(np.dot(m1, w1) / w1.sum())
        a2 = float(np.dot(m2, w2) / w2.sum())
        hard = (m1 > 0.999).sum() + (m2 > 0.999).sum()
        return min(a1, a2), (a1 + a2) / 2, hard / (len(t1) + len(t2))

    # -------------------------------------------------------------- main
    def transform(self, s1_list, s2_list, verbose=False):
        S1 = [clean(s) for s in s1_list]
        S2 = [clean(s) for s in s2_list]
        St1 = [" ".join(stems(s)) for s in S1]
        St2 = [" ".join(stems(s)) for s in S2]
        n = len(S1)
        cols, names = [], []

        def add(vals, nm):
            cols.append(np.asarray(vals, dtype=np.float32).reshape(n, -1))
            k = cols[-1].shape[1]
            names.extend([nm] if k == 1 else [f"{nm}_{i}" for i in range(k)])

        # --- sparse cosine similarities
        for name, _ in VEC_SPECS:
            a = St1 if name.startswith("w") else S1
            b = St2 if name.startswith("w") else S2
            A, B = self._tf(name, a), self._tf(name, b)
            add(np.asarray(A.multiply(B).sum(1)).ravel(), f"cos_{name}")
            # l1 / dice style on the same space
            inter = A.minimum(B)
            add(np.asarray(inter.sum(1)).ravel(), f"min_{name}")
            na = np.asarray(A.sum(1)).ravel(); nb = np.asarray(B.sum(1)).ravel()
            add(2 * np.asarray(inter.sum(1)).ravel() / np.maximum(na + nb, 1e-9), f"dice_{name}")

        # --- LSA spaces: cosine, distances, and elementwise interactions
        for name in ("cwb24", "w12"):
            a = St1 if name.startswith("w") else S1
            b = St2 if name.startswith("w") else S2
            U = self.svds[name].transform(self._tf(name, a))
            V = self.svds[name].transform(self._tf(name, b))
            Un, Vn = normalize(U), normalize(V)
            add((Un * Vn).sum(1), f"lsacos_{name}")
            add(np.linalg.norm(U - V, axis=1), f"lsal2_{name}")
            add(np.abs(U - V).sum(1), f"lsal1_{name}")
            k = self.svd_out
            add(np.abs(Un[:, :k] - Vn[:, :k]), f"lsad_{name}")
            add(Un[:, :k] * Vn[:, :k], f"lsap_{name}")
            # cosine restricted to leading dims (denoised)
            for k2 in (8, 16, 32, 64, 128):
                add((normalize(U[:, :k2]) * normalize(V[:, :k2])).sum(1), f"lsacos{k2}_{name}")

        # --- discrete overlap + string features (python loop)
        feat_rows = []
        for i in range(n):
            a, b = S1[i], S2[i]
            da, db = depunct(a), depunct(b)
            ta, tb = tokens(a), tokens(b)
            sa, sb = set(stems(a)), set(stems(b))
            row = []
            for k in (1, 2, 3, 4, 5):
                j, cmin, cmax = jac(ngrams(a, k), ngrams(b, k))
                row += [j, cmin, cmax]
            j, cmin, cmax = jac(set(ta), set(tb))
            row += [j, cmin, cmax]
            j, cmin, cmax = jac(sa, sb)
            row += [j, cmin, cmax]
            # content-token (idf-weighted) overlap
            wsum = sum(self.idf.get(t, self.max_idf) for t in sa | sb) or 1.0
            row.append(sum(self.idf.get(t, self.max_idf) for t in sa & sb) / wsum)
            # lengths
            la, lb = len(da), len(db)
            row += [min(la, lb), max(la, lb), abs(la - lb),
                    min(la, lb) / max(max(la, lb), 1),
                    min(len(ta), len(tb)), max(len(ta), len(tb)),
                    abs(len(ta) - len(tb)),
                    min(len(ta), len(tb)) / max(max(len(ta), len(tb)), 1)]
            # string similarity
            sm = SequenceMatcher(None, da, db)
            row.append(sm.ratio())
            row.append(sm.quick_ratio())
            longest = max((bl.size for bl in sm.get_matching_blocks()), default=0)
            row.append(longest / max(min(la, lb), 1))
            L = lcs_len(da[:160], db[:160])
            row += [L / max(min(la, lb), 1), L / max(max(la, lb), 1),
                    2 * L / max(la + lb, 1)]
            # numbers
            na_, nb_ = set(NUM.findall(a)), set(NUM.findall(b))
            row += [float(len(na_) > 0 or len(nb_) > 0),
                    (len(na_ & nb_) / len(na_ | nb_)) if (na_ | nb_) else 1.0,
                    float(na_ == nb_), abs(len(na_) - len(nb_))]
            # question / negation / punctuation cues
            qa, qb = ("?" in a), ("?" in b)
            row += [float(qa == qb), float(qa or qb)]
            neg = ("안 ", "안돼", "않", "없", "못", "아니")
            fa = sum(x in a for x in neg); fb = sum(x in b for x in neg)
            row += [float((fa > 0) == (fb > 0)), abs(fa - fb)]
            # first/last token match
            row += [float(bool(ta) and bool(tb) and ta[0] == tb[0]),
                    float(bool(ta) and bool(tb) and ta[-1] == tb[-1])]
            # greedy alignment
            row += list(self._align(ta, tb))
            feat_rows.append(row)
            if verbose and i % 2000 == 0:
                print("  discrete feats", i, "/", n, flush=True)
        F = np.asarray(feat_rows, dtype=np.float32)
        add(F, "disc")

        X = np.hstack(cols).astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        self.feature_names_ = names
        return X

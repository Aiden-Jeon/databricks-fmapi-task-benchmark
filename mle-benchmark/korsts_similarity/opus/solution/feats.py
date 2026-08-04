"""Feature engineering for KorSTS sentence similarity (no pretrained models)."""
import re, unicodedata, difflib
from collections import Counter
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

# ---------------- text normalization ----------------
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")

def norm(s):
    s = unicodedata.normalize("NFKC", str(s)).lower().strip()
    s = _WS.sub(" ", s)
    return s

def nopunct(s):
    return _WS.sub(" ", _PUNCT.sub(" ", s)).strip()

def jamo(s):
    return unicodedata.normalize("NFD", s)

# common Korean particles / endings -> crude stemming by trimming suffixes
_SUF = ["으로써", "에서는", "에게서", "이라고", "라고는", "으로는", "에서도", "까지도",
        "이라는", "라는", "으로", "에서", "에게", "부터", "까지", "처럼", "보다", "만큼",
        "하고", "이나", "든지", "이며", "하며", "라도", "이다", "한다", "했다", "이는",
        "은", "는", "이", "가", "을", "를", "에", "의", "도", "만", "과", "와", "로", "야", "여"]

def stem_tok(t):
    if len(t) <= 2:
        return t
    for s in _SUF:
        if len(t) - len(s) >= 2 and t.endswith(s):
            return t[: len(t) - len(s)]
    return t

def toks(s):
    return nopunct(s).split()

def stoks(s):
    return [stem_tok(t) for t in toks(s)]


# ---------------- basic set similarity helpers ----------------
def jacc(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0

def dice(a, b):
    a, b = set(a), set(b)
    if not a or not b:
        return 0.0
    return 2 * len(a & b) / (len(a) + len(b))

def contain(a, b):
    a, b = set(a), set(b)
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))

def ngrams(s, n):
    s = s.replace(" ", "")
    return [s[i:i + n] for i in range(max(0, len(s) - n + 1))]

def lcs_len(a, b):
    # length of longest common subsequence (chars)
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0]
        for j, cb in enumerate(b):
            cur.append(prev[j] + 1 if ca == cb else max(cur[j], prev[j + 1]))
        prev = cur
    return prev[-1]

def lcsubstr(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for ca in a:
        cur = [0] * (len(b) + 1)
        for j, cb in enumerate(b):
            if ca == cb:
                cur[j + 1] = prev[j] + 1
                if cur[j + 1] > best:
                    best = cur[j + 1]
        prev = cur
    return best

_NUM = re.compile(r"\d+(?:[.,]\d+)?")
_LAT = re.compile(r"[a-z]{2,}")


def idf_weighted_overlap(a, b, idf, default):
    sa, sb = set(a), set(b)
    inter = sa & sb
    wi = sum(idf.get(t, default) for t in inter)
    wu = sum(idf.get(t, default) for t in (sa | sb))
    wmin = min(sum(idf.get(t, default) for t in sa), sum(idf.get(t, default) for t in sb))
    return (wi / wu if wu else 0.0), (wi / wmin if wmin else 0.0)


def char_sim(x, y):
    """cheap char-bigram dice between two tokens"""
    if x == y:
        return 1.0
    if len(x) < 2 or len(y) < 2:
        return 1.0 if x == y else 0.0
    A = Counter(x[i:i + 2] for i in range(len(x) - 1))
    B = Counter(y[i:i + 2] for i in range(len(y) - 1))
    inter = sum((A & B).values())
    return 2 * inter / (sum(A.values()) + sum(B.values()))


def soft_align(a, b, idf, default, thr=0.0):
    """IDF-weighted soft alignment: for each token in a, best fuzzy match in b."""
    if not a or not b:
        return 0.0, 0.0
    tot = 0.0
    wsum = 0.0
    for x in a:
        w = idf.get(x, default)
        best = max(char_sim(x, y) for y in b)
        if best < thr:
            best = 0.0
        tot += w * best
        wsum += w
    return (tot / wsum if wsum else 0.0), 0.0


class FeatureBuilder:
    """Fits unsupervised components on training sentences only."""

    def __init__(self, svd_dim=180, seed=0):
        self.svd_dim = svd_dim
        self.seed = seed

    def fit(self, s1, s2):
        corpus = [norm(x) for x in s1] + [norm(x) for x in s2]
        self.corpus_ = corpus
        cj = [jamo(nopunct(c)) for c in corpus]
        cw = [" ".join(stoks(c)) for c in corpus]

        self.vecs_ = {}
        self.vecs_["char23"] = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3),
                                               sublinear_tf=True, min_df=1).fit(corpus)
        self.vecs_["char45"] = TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 5),
                                               sublinear_tf=True, min_df=2).fit(corpus)
        self.vecs_["jamo34"] = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                               sublinear_tf=True, min_df=2).fit(cj)
        self.vecs_["word"] = TfidfVectorizer(analyzer="word", ngram_range=(1, 1),
                                             sublinear_tf=True, min_df=1,
                                             token_pattern=r"\S+").fit(cw)
        self.vecs_["word12"] = TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                               sublinear_tf=True, min_df=1,
                                               token_pattern=r"\S+").fit(cw)

        # idf dicts for tokens
        wv = self.vecs_["word"]
        self.idf_ = dict(zip(wv.get_feature_names_out(), wv.idf_))
        self.idf_default_ = float(np.max(wv.idf_))

        # LSA spaces
        self.svds_ = {}
        for key in ["char23", "jamo34", "word12"]:
            src = corpus if key == "char23" else (cj if key == "jamo34" else cw)
            X = self.vecs_[key].transform(src)
            sv = TruncatedSVD(n_components=self.svd_dim, random_state=self.seed)
            sv.fit(X)
            self.svds_[key] = sv
        return self

    # ---- vector part (dense embeddings for interaction features) ----
    def embed(self, s1, s2, key):
        a = [norm(x) for x in s1]
        b = [norm(x) for x in s2]
        if key == "jamo34":
            a = [jamo(nopunct(x)) for x in a]
            b = [jamo(nopunct(x)) for x in b]
        elif key == "word12":
            a = [" ".join(stoks(x)) for x in a]
            b = [" ".join(stoks(x)) for x in b]
        U = normalize(self.svds_[key].transform(self.vecs_[key].transform(a)))
        V = normalize(self.svds_[key].transform(self.vecs_[key].transform(b)))
        return U, V

    def transform(self, s1, s2):
        n = len(s1)
        A = [norm(x) for x in s1]
        B = [norm(x) for x in s2]
        Aj = [jamo(nopunct(x)) for x in A]
        Bj = [jamo(nopunct(x)) for x in B]
        Aw = [" ".join(stoks(x)) for x in A]
        Bw = [" ".join(stoks(x)) for x in B]

        cols = {}
        # tfidf cosines
        for key, (sa, sb) in {
            "char23": (A, B), "char45": (A, B), "jamo34": (Aj, Bj),
            "word": (Aw, Bw), "word12": (Aw, Bw)
        }.items():
            X = normalize(self.vecs_[key].transform(sa))
            Y = normalize(self.vecs_[key].transform(sb))
            cols["cos_" + key] = np.asarray(X.multiply(Y).sum(1)).ravel()

        # LSA cosines + euclidean
        for key in self.svds_:
            U, V = self.embed(s1, s2, key)
            cols["lsacos_" + key] = (U * V).sum(1)
            cols["lsal2_" + key] = np.linalg.norm(U - V, axis=1)

        # token/char set features
        feats = []
        for i in range(n):
            ta, tb = toks(A[i]), toks(B[i])
            sa_, sb_ = stoks(A[i]), stoks(B[i])
            ca, cb = list(A[i].replace(" ", "")), list(B[i].replace(" ", ""))
            g2a, g2b = ngrams(nopunct(A[i]), 2), ngrams(nopunct(B[i]), 2)
            g3a, g3b = ngrams(nopunct(A[i]), 3), ngrams(nopunct(B[i]), 3)
            ja, jb = list(Aj[i].replace(" ", "")), list(Bj[i].replace(" ", ""))

            na, nb = _NUM.findall(A[i]), _NUM.findall(B[i])
            la, lb = _LAT.findall(A[i]), _LAT.findall(B[i])

            la_, lb_ = len(ta), len(tb)
            cla, clb = len(A[i]), len(B[i])
            lcs = lcs_len(ca[:120], cb[:120])
            lss = lcsubstr(ca[:120], cb[:120])
            wo1, wo2 = idf_weighted_overlap(sa_, sb_, self.idf_, self.idf_default_)
            sa1, _ = soft_align(sa_, sb_, self.idf_, self.idf_default_)
            sa2, _ = soft_align(sb_, sa_, self.idf_, self.idf_default_)

            row = [
                jacc(ta, tb), dice(ta, tb), contain(ta, tb),
                jacc(sa_, sb_), dice(sa_, sb_), contain(sa_, sb_),
                jacc(ca, cb), dice(ca, cb), contain(ca, cb),
                jacc(g2a, g2b), dice(g2a, g2b), contain(g2a, g2b),
                jacc(g3a, g3b), dice(g3a, g3b), contain(g3a, g3b),
                jacc(ja, jb), dice(ja, jb),
                wo1, wo2, sa1, sa2, (sa1 + sa2) / 2, min(sa1, sa2),
                difflib.SequenceMatcher(None, A[i], B[i]).ratio(),
                difflib.SequenceMatcher(None, Aw[i], Bw[i]).ratio(),
                lcs / max(1, min(len(ca), len(cb))),
                lcs / max(1, max(len(ca), len(cb))),
                lss / max(1, min(len(ca), len(cb))),
                la_, lb_, abs(la_ - lb_), min(la_, lb_) / max(1, max(la_, lb_)),
                cla, clb, abs(cla - clb), min(cla, clb) / max(1, max(cla, clb)),
                len(na), len(nb), jacc(na, nb) if (na or nb) else 1.0,
                1.0 if set(na) == set(nb) else 0.0,
                len(la), len(lb), jacc(la, lb) if (la or lb) else 1.0,
                1.0 if A[i] == B[i] else 0.0,
                len(set(ta) & set(tb)), len(set(sa_) & set(sb_)),
                len(set(ta) - set(tb)), len(set(tb) - set(ta)),
            ]
            feats.append(row)
        F = np.asarray(feats, dtype=np.float64)
        names = ["tok_jacc", "tok_dice", "tok_cont", "stok_jacc", "stok_dice", "stok_cont",
                 "ch_jacc", "ch_dice", "ch_cont", "g2_jacc", "g2_dice", "g2_cont",
                 "g3_jacc", "g3_dice", "g3_cont", "jamo_jacc", "jamo_dice",
                 "idfov_u", "idfov_min", "salign_ab", "salign_ba", "salign_mean", "salign_min",
                 "sm_char", "sm_word", "lcs_min", "lcs_max", "lss_min",
                 "n_tok_a", "n_tok_b", "n_tok_diff", "n_tok_ratio",
                 "clen_a", "clen_b", "clen_diff", "clen_ratio",
                 "num_a", "num_b", "num_jacc", "num_eq",
                 "lat_a", "lat_b", "lat_jacc", "exact_eq",
                 "ntok_inter", "nstok_inter", "ntok_a_only", "ntok_b_only"]
        out = pd.DataFrame(F, columns=names)
        for k, v in cols.items():
            out[k] = v
        return out

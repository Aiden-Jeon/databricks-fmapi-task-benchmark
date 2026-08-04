"""Data loading / tokenisation utilities for KorNLI (t10).

Everything is learned from ``train.csv`` only: no external corpora,
no pretrained embeddings, no internet.

Korean is agglutinative, so a plain whitespace vocabulary is very sparse.
We therefore represent each *eojeol* (whitespace token) as a bag of
sub-word units (the word itself + its character n-grams, fastText style).
Word embeddings are the mean of the embeddings of those units, which lets
the model share statistical strength between inflected forms.
"""
import re
import unicodedata
from zlib import crc32
import numpy as np
import pandas as pd

LABELS = ["entailment", "neutral", "contradiction"]
L2I = {l: i for i, l in enumerate(LABELS)}

MAX_L1 = 32          # premise words
MAX_L2 = 20          # hypothesis words
NGRAM_MIN, NGRAM_MAX = 3, 4
N_HASH = 120_000     # hashed char-ngram buckets
K_UNITS = 10         # max sub-word units kept per word

_punct = re.compile(r"[^\w\s]", flags=re.UNICODE)
_ws = re.compile(r"\s+")


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = _punct.sub(" ", s)
    return _ws.sub(" ", s).strip()


def tokens(s: str):
    return norm(s).split()


def _ngrams(w: str):
    t = "<" + w + ">"
    out = []
    for n in range(NGRAM_MIN, NGRAM_MAX + 1):
        if len(t) < n:
            break
        for i in range(len(t) - n + 1):
            out.append(t[i:i + n])
    return out


class WordTable:
    """Maps every distinct word seen in the data to a list of unit ids.

    unit ids ``[0, n_vocab)``   -> whole-word ids (frequent words only)
    unit ids ``[n_vocab, ...)`` -> hashed character n-grams
    id 0 is PAD, id 1 is UNK.
    """

    def __init__(self, min_count=3):
        self.min_count = min_count

    def fit(self, texts):
        from collections import Counter
        c = Counter()
        for t in texts:
            c.update(tokens(t))
        vocab = [w for w, n in c.most_common() if n >= self.min_count]
        self.word2id = {w: i + 2 for i, w in enumerate(vocab)}
        self.n_vocab = len(vocab) + 2
        self.n_units = self.n_vocab + N_HASH
        self.counter = c
        self._cache = {}
        return self

    def units(self, w: str):
        u = self._cache.get(w)
        if u is None:
            ids = [self.word2id.get(w, 1)]
            for g in _ngrams(w):
                ids.append(self.n_vocab + (crc32(g.encode('utf8')) % N_HASH))
            u = ids[:K_UNITS]
            self._cache[w] = u
        return u


class WordIndex:
    """Assigns a dense index to each distinct word and stores its units."""

    def __init__(self, table: WordTable):
        self.table = table
        self.w2i = {"": 0}
        self.units = [[0] * K_UNITS]
        self.nunits = [1]

    def add(self, w):
        i = self.w2i.get(w)
        if i is None:
            i = len(self.w2i)
            self.w2i[w] = i
            u = self.table.units(w)
            self.units.append(u + [0] * (K_UNITS - len(u)))
            self.nunits.append(len(u))
        return i

    def finalize(self):
        self.U = np.asarray(self.units, dtype=np.int64)
        self.N = np.asarray(self.nunits, dtype=np.float32)
        return self


def encode(texts, widx: WordIndex, maxlen):
    n = len(texts)
    X = np.zeros((n, maxlen), dtype=np.int64)
    L = np.zeros(n, dtype=np.int64)
    for r, t in enumerate(texts):
        tk = tokens(t)[:maxlen]
        for j, w in enumerate(tk):
            X[r, j] = widx.add(w)
        L[r] = max(len(tk), 1)
    return X, L


# --------------------------------------------------------------------------
# hand crafted lexical-overlap features (cheap, complementary to the net)
# --------------------------------------------------------------------------
NEG = ["안", "않", "없", "못", "아니", "결코", "전혀", "누구도", "아무도", "절대"]


def stem(w):
    return w[:-1] if len(w) > 3 else w


def pair_features(s1, s2):
    F = []
    for a, b in zip(s1, s2):
        ta, tb = tokens(a), tokens(b)
        sa, sb = set(ta), set(tb)
        ga = set(stem(w) for w in ta)
        gb = set(stem(w) for w in tb)
        ca = set(norm(a).replace(" ", ""))
        cb = set(norm(b).replace(" ", ""))
        inter = len(sa & sb)
        gi = len(ga & gb)
        ci = len(ca & cb)
        la, lb = max(len(sa), 1), max(len(sb), 1)
        na = sum(any(k in w for k in NEG) for w in ta)
        nb = sum(any(k in w for k in NEG) for w in tb)
        F.append([
            inter / lb, inter / la, inter / max(len(sa | sb), 1),
            gi / lb, gi / la, gi / max(len(ga | gb), 1),
            ci / max(len(cb), 1), ci / max(len(ca | cb), 1),
            len(ta) / 30.0, len(tb) / 15.0, (len(tb) - len(ta)) / 20.0,
            np.log1p(len(ta)) - np.log1p(len(tb)),
            min(na, 3) / 3.0, min(nb, 3) / 3.0, float(nb > na), float(na > nb),
            float(len(sb - sa) == 0), len(sb - sa) / lb, len(gb - ga) / lb,
        ])
    return np.asarray(F, dtype=np.float32)


def load():
    tr = pd.read_csv("train.csv")
    te = pd.read_csv("test.csv")
    for d in (tr, te):
        d["sentence1"] = d["sentence1"].astype(str)
        d["sentence2"] = d["sentence2"].astype(str)
    return tr, te

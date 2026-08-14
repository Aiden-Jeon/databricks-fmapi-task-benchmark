"""Feature extraction for Korean STS (KorSTS) sentence-pair similarity.

All features are computed purely from the sentence pair itself
(no external data, no pretrained weights).
"""
import re
from difflib import SequenceMatcher
from collections import Counter

import numpy as np

_PUNCT_RE = re.compile(r"[^\w\s가-힣]", re.UNICODE)


def normalize(text):
    text = str(text).lower()
    text = _PUNCT_RE.sub(" ", text)
    return text.split()


def char_ngrams(text, n_min=2, n_max=4):
    text = re.sub(r"\s+", "", str(text).lower())
    out = []
    L = len(text)
    for n in range(n_min, n_max + 1):
        if L >= n:
            out.extend(text[i:i + n] for i in range(L - n + 1))
    return out


def _jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / (len(a) + len(b) - inter)


def _dice(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return 2.0 * len(a & b) / (len(a) + len(b))


def _containment(a, b):
    """fraction of grams of the smaller set covered by the larger"""
    if not a or not b:
        return 0.0
    small, large = (a, b) if len(a) <= len(b) else (b, a)
    return len(small & large) / len(small)


def _counter_overlap(ca, cb):
    """cosine-like overlap between multiset counters"""
    if not ca or not cb:
        return 0.0, 0.0
    keys = set(ca) | set(cb)
    dot = sum(ca.get(k, 0) * cb.get(k, 0) for k in keys)
    na = sum(v * v for v in ca.values()) ** 0.5
    nb = sum(v * v for v in cb.values()) ** 0.5
    cos = dot / (na * nb) if na and nb else 0.0
    inter = sum(min(ca.get(k, 0), cb.get(k, 0)) for k in keys)
    denom = min(sum(ca.values()), sum(cb.values()))
    cont = inter / denom if denom else 0.0
    return cos, cont


def pair_features(s1, s2):
    t1, t2 = normalize(s1), normalize(s2)
    w1, w2 = set(t1), set(t2)
    c1 = Counter(char_ngrams(s1, 2, 3))
    c2 = Counter(char_ngrams(s2, 2, 3))
    g1 = set(c1)
    g2 = set(c2)

    wcos, wcont = _counter_overlap(Counter(t1), Counter(t2))
    ccos, ccont = _counter_overlap(c1, c2)

    raw1 = re.sub(r"\s+", "", str(s1).lower())
    raw2 = re.sub(r"\s+", "", str(s2).lower())
    sm_ratio = SequenceMatcher(None, raw1, raw2).ratio()
    sm_tok = SequenceMatcher(None, t1, t2).ratio()

    l1, l2 = len(t1), len(t2)
    cl1, cl2 = len(raw1), len(raw2)

    # number overlap (handles "4" vs "네" mismatch partially via digit tokens)
    n1 = re.findall(r"\d+", str(s1))
    n2 = re.findall(r"\d+", str(s2))
    num_jac = _jaccard(set(n1), set(n2)) if (n1 or n2) else 0.5

    feats = [
        _jaccard(w1, w2),
        _dice(w1, w2),
        _containment(w1, w2),
        wcos,
        wcont,
        _jaccard(g1, g2),
        _dice(g1, g2),
        _containment(g1, g2),
        ccos,
        ccont,
        sm_ratio,
        sm_tok,
        abs(l1 - l2),
        l1,
        l2,
        min(l1, l2) / max(l1, l2) if max(l1, l2) else 1.0,
        abs(cl1 - cl2),
        min(cl1, cl2) / max(cl1, cl2) if max(cl1, cl2) else 1.0,
        l1 * l2,
        num_jac,
        len(w1 & w2),
        len(g1 & g2),
        1.0 if raw1 == raw2 else 0.0,
    ]
    return np.asarray(feats, dtype=np.float32)


def build_matrix(df):
    feats = [pair_features(a, b) for a, b in zip(df["sentence1"], df["sentence2"])]
    return np.vstack(feats)

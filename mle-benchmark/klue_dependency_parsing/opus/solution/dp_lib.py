"""
KLUE-DP (Korean dependency parsing) — feature-based structured perceptron parser.

Key structural facts verified on train.csv (see explore.py):
  * every arc is head-final: head index > dependent index (100%)
  * every tree is projective (100%)
  * exactly one root per sentence and it is always the LAST word (structurally forced)

Therefore the search space is exactly "projective trees over 1..n where every head is
to the right of its dependent".  Reversing the token order turns these into
head-initial projective trees rooted at position 0, for which there is a very simple
exact O(n^3) CKY-style DP:

    T[i][i] = 0
    T[i][k] = max_{d in (i, k]}  T[i][d-1] + T[d][k] + score(dep=d, head=i)

T[i][k] = best complete subtree rooted at i covering the (reversed) span [i, k];
the recursion splits off the *last* child d of i, whose subtree covers [d, k].

Scoring is a first-order (arc-factored) linear model trained with an averaged
structured perceptron.  Labels (deprel) are predicted by a separate multiclass
averaged perceptron conditioned on the predicted arc.
"""

import json
import numpy as np

# --------------------------------------------------------------------------- #
# token attributes
# --------------------------------------------------------------------------- #

_PUNCT = '.,!?"\'\u2019\u201d\u201c\u2018)(][}{<>\u300d\u300f\u300c\u300e\u2026\u00b7:;~/`'
_BOS = "<s>"
_EOS = "</s>"


def _digitize(w):
    out = []
    for ch in w:
        if ch.isdigit():
            out.append("0")
        else:
            out.append(ch)
    return "".join(out)


# Words seen at least MIN_WFREQ times get their own lexical feature; rarer words are
# backed off to a suffix-based pseudo-word.  Set via set_vocab() before building Toks.
VOCAB = None
CLUSTER = None


def set_vocab(v):
    global VOCAB
    VOCAB = v


def set_cluster(c):
    global CLUSTER
    CLUSTER = c


class Tok:
    """Pre-computed string attributes of one eojeol (whitespace-delimited word)."""

    __slots__ = ("w", "s1", "s2", "s3", "s4", "p1", "p2", "sh", "cl")

    def __init__(self, raw):
        core = raw.rstrip(_PUNCT)
        if not core:
            core = raw
        w = _digitize(core)
        self.s1 = w[-1:]
        self.s2 = w[-2:]
        self.s3 = w[-3:]
        self.s4 = w[-4:]
        self.p1 = w[:1]
        self.p2 = w[:2]
        trail = raw[len(raw.rstrip(_PUNCT)):]
        has_dig = any(c.isdigit() for c in raw)
        is_lat = all((ord(c) < 128) for c in core) and core.isascii()
        self.sh = "%s|%d|%d|%d" % (trail[:2], min(len(core), 6), has_dig, is_lat)
        if VOCAB is None or w in VOCAB:
            self.w = w
        else:
            self.w = "\x02" + w[-3:]
        self.cl = CLUSTER.get(w, "?") if CLUSTER else "?"


def _sentinel(tag):
    t = Tok("x")
    t.w = t.s1 = t.s2 = t.s3 = t.s4 = t.p1 = t.p2 = t.sh = t.cl = tag
    return t


_SENT_B = _sentinel(_BOS)
_SENT_E = _sentinel(_EOS)


def toks_of(words):
    return [Tok(w) for w in words]


def _at(tl, i, n):
    if i < 0:
        return _SENT_B
    if i >= n:
        return _SENT_E
    return tl[i]


def _dbucket(dist):
    if dist <= 5:
        return str(dist)
    if dist <= 7:
        return "6-7"
    if dist <= 10:
        return "8-10"
    if dist <= 15:
        return "b15"
    return "big"


# --------------------------------------------------------------------------- #
# arc features (for head selection)
# --------------------------------------------------------------------------- #

def arc_feats(tl, d, h, n):
    """String features of candidate arc dep=d -> head=h (0-based, h > d)."""
    D = tl[d]
    H = tl[h]
    dm = _at(tl, d - 1, n)
    dp = _at(tl, d + 1, n)
    hm = _at(tl, h - 1, n)
    hp = _at(tl, h + 1, n)
    dist = h - d
    B = _dbucket(dist)
    hlast = "1" if h == n - 1 else "0"
    adj = "1" if dist == 1 else "0"
    dpos = _dbucket(d + 1)
    hend = _dbucket(n - h)
    dend = _dbucket(n - d)

    f = [
        # ---- unigram: dependent
        "a1=" + D.w,
        "a2=" + D.s1,
        "a3=" + D.s2,
        "a4=" + D.s3,
        "a5=" + D.s4,
        "a6=" + D.p2,
        "a7=" + D.sh,
        # ---- unigram: head
        "b1=" + H.w,
        "b2=" + H.s1,
        "b3=" + H.s2,
        "b4=" + H.s3,
        "b5=" + H.s4,
        "b6=" + H.p2,
        "b7=" + H.sh,
        # ---- bigram dep x head
        "c1=" + D.w + "\x01" + H.w,
        "c2=" + D.s1 + "\x01" + H.s1,
        "c3=" + D.s2 + "\x01" + H.s2,
        "c4=" + D.s3 + "\x01" + H.s3,
        "c5=" + D.w + "\x01" + H.s2,
        "c6=" + D.s2 + "\x01" + H.w,
        "c7=" + D.s3 + "\x01" + H.s2,
        "c8=" + D.s2 + "\x01" + H.s3,
        "c9=" + D.sh + "\x01" + H.sh,
        # ---- distance
        "d0=" + B,
        "d1=" + D.s2 + "\x01" + B,
        "d2=" + H.s2 + "\x01" + B,
        "d3=" + D.s2 + "\x01" + H.s2 + "\x01" + B,
        "d4=" + D.s1 + "\x01" + H.s1 + "\x01" + B,
        "d5=" + D.w + "\x01" + B,
        "d6=" + H.w + "\x01" + B,
        "d7=" + D.s3 + "\x01" + H.s3 + "\x01" + B,
        "d8=" + D.s3 + "\x01" + B,
        # ---- adjacency / root flags
        "e0=" + adj + D.s2 + "\x01" + H.s2,
        "e1=" + hlast + "\x01" + D.s2,
        "e2=" + hlast + "\x01" + D.s3,
        "e3=" + hlast + "\x01" + D.s2 + "\x01" + B,
        # ---- positional
        "f1=" + D.s2 + "\x01" + dpos,
        "f2=" + H.s2 + "\x01" + hend,
        "f3=" + D.s2 + "\x01" + dend,
        "f4=" + D.s2 + "\x01" + H.s2 + "\x01" + hend,
        # ---- surrounding context
        "g1=" + D.s2 + "\x01" + dp.s2,
        "g2=" + dm.s2 + "\x01" + D.s2,
        "g3=" + hm.s2 + "\x01" + H.s2,
        "g4=" + H.s2 + "\x01" + hp.s2,
        "g5=" + D.s2 + "\x01" + dp.s2 + "\x01" + H.s2,
        "g6=" + dm.s2 + "\x01" + D.s2 + "\x01" + H.s2,
        "g7=" + D.s2 + "\x01" + hm.s2 + "\x01" + H.s2,
        "g8=" + D.s2 + "\x01" + H.s2 + "\x01" + hp.s2,
        "g9=" + D.s1 + "\x01" + dp.s1 + "\x01" + hm.s1 + "\x01" + H.s1,
        "ga=" + D.s2 + "\x01" + hm.s2,
        "gb=" + dp.s2 + "\x01" + H.s2,
        "gc=" + dp.s1 + "\x01" + H.s1 + "\x01" + B,
    ]
    # ---- "in between" features: which kinds of words sit between dep and head
    if dist > 1:
        seen = set()
        for b in range(d + 1, h):
            k = tl[b].s1
            if k in seen:
                continue
            seen.add(k)
            f.append("h1=" + D.s1 + "\x01" + k + "\x01" + H.s1)
            f.append("h2=" + k + "\x01" + H.s2)
        f.append("h3=" + D.s2 + "\x01" + str(min(len(seen), 6)) + "\x01" + H.s2)
    return f


# --------------------------------------------------------------------------- #
# second-order features
# --------------------------------------------------------------------------- #

def sib_feats(tl, h, c1, c2, n):
    """Adjacent-sibling features.  Original indices with c2 < c1 < h:
    h is the head, c1 the nearer child, c2 the farther (next) child."""
    H = tl[h]
    A = tl[c1]
    B = tl[c2]
    g = _dbucket(c1 - c2)
    adj = "1" if c1 - c2 == 1 else "0"
    return [
        "S0=" + H.s2 + "\x01" + A.s2 + "\x01" + B.s2,
        "S1=" + H.s2 + "\x01" + B.s2,
        "S2=" + A.s2 + "\x01" + B.s2,
        "S3=" + H.s1 + "\x01" + A.s1 + "\x01" + B.s1,
        "S4=" + H.s2 + "\x01" + A.s2 + "\x01" + B.s2 + "\x01" + g,
        "S5=" + H.w + "\x01" + A.s2 + "\x01" + B.s2,
        "S6=" + A.s3 + "\x01" + B.s3,
        "S7=" + H.s2 + "\x01" + A.s2 + "\x01" + B.s2 + "\x01" + adj,
        "S8=" + H.s3 + "\x01" + B.s2,
        "S9=" + A.s2 + "\x01" + B.s2 + "\x01" + g,
        "SA=" + H.s2 + "\x01" + A.s1 + "\x01" + B.s2 + "\x01" + _dbucket(h - c2),
    ]


def fc_feats(tl, h, n):
    """Score applied once when head h has at least one dependent (its nearest
    dependent is always h-1 in this head-final projective setting)."""
    H = tl[h]
    A = _at(tl, h - 1, n)
    return [
        "T0=" + H.s2,
        "T1=" + H.s3,
        "T2=" + H.w,
        "T3=" + H.s2 + "\x01" + A.s2,
        "T4=" + H.s2 + "\x01" + _dbucket(n - h),
    ]


# --------------------------------------------------------------------------- #
# label features
# --------------------------------------------------------------------------- #

def lab_feats(tl, d, h, n):
    """String features for predicting deprel of dep=d given head=h (h=-1 => root)."""
    D = tl[d]
    dm = _at(tl, d - 1, n)
    dp = _at(tl, d + 1, n)
    if h < 0:
        H = _SENT_E
        hm = _at(tl, d - 1, n)
        dist = 0
        B = "ROOT"
        hlast = "R"
    else:
        H = tl[h]
        hm = _at(tl, h - 1, n)
        dist = h - d
        B = _dbucket(dist)
        hlast = "1" if h == n - 1 else "0"
    f = [
        "*",
        "A1=" + D.w,
        "A2=" + D.s1,
        "A3=" + D.s2,
        "A4=" + D.s3,
        "A5=" + D.s4,
        "A6=" + D.p2,
        "A7=" + D.sh,
        "B1=" + H.w,
        "B2=" + H.s1,
        "B3=" + H.s2,
        "B4=" + H.s3,
        "B5=" + H.s4,
        "B7=" + H.sh,
        "C1=" + D.s1 + "\x01" + H.s1,
        "C2=" + D.s2 + "\x01" + H.s2,
        "C3=" + D.s3 + "\x01" + H.s3,
        "C4=" + D.w + "\x01" + H.s2,
        "C5=" + D.s2 + "\x01" + H.w,
        "C6=" + D.s3 + "\x01" + H.s2,
        "C7=" + D.s4 + "\x01" + H.s3,
        "C8=" + D.sh + "\x01" + H.sh,
        "D0=" + B,
        "D1=" + D.s2 + "\x01" + B,
        "D2=" + D.s3 + "\x01" + B,
        "D3=" + H.s2 + "\x01" + B,
        "D4=" + D.s2 + "\x01" + H.s2 + "\x01" + B,
        "E1=" + hlast,
        "E2=" + hlast + "\x01" + D.s2,
        "E3=" + hlast + "\x01" + D.s3,
        "F1=" + D.s2 + "\x01" + dp.s2,
        "F2=" + dm.s2 + "\x01" + D.s2,
        "F3=" + hm.s2 + "\x01" + D.s2,
        "F4=" + D.s3 + "\x01" + dp.s2,
        "F5=" + D.s2 + "\x01" + _dbucket(n - d),
        "F6=" + D.s3 + "\x01" + _dbucket(d + 1),
    ]
    return f


# --------------------------------------------------------------------------- #
# exact DP over head-final projective trees
# --------------------------------------------------------------------------- #

def decode(sc, n):
    """sc: (n,n) float array, sc[d,h] = score of arc dep=d -> head=h (only h>d used).
    Returns heads: list length n, 0-based head index, -1 for root (== n-1)."""
    if n == 1:
        return [-1]
    # reverse orientation: srev[i,j] = sc[n-1-i, n-1-j]; arc dep=i(rev) head=j(rev), j<i
    srev = sc[::-1, ::-1]
    NEG = -1e18
    T = np.zeros((n, n), dtype=np.float64)
    BP = np.zeros((n, n), dtype=np.int32)
    for L in range(1, n):
        for i in range(0, n - L):
            k = i + L
            cand = T[i, i:k] + T[i + 1:k + 1, k] + srev[i + 1:k + 1, i]
            j = int(np.argmax(cand))
            T[i, k] = cand[j]
            BP[i, k] = i + 1 + j
    heads = [-1] * n
    stack = [(0, n - 1)]
    while stack:
        i, k = stack.pop()
        if i >= k:
            continue
        d = int(BP[i, k])
        # reversed arc dep=d head=i  ->  original
        heads[n - 1 - d] = n - 1 - i
        if d - 1 > i:
            stack.append((i, d - 1))
        if k > d:
            stack.append((d, k))
    heads[n - 1] = -1
    return heads


def decode2(sc, sibt, fcs, n):
    """Second-order (adjacent-sibling) exact DP.

    All arrays are in ORIGINAL orientation:
      sc[d, h]        arc score, h > d
      sibt[h, c1, c2] sibling score, c2 < c1 < h
      fcs[h]          "h has >=1 dependent" score
    Returns 0-based heads, -1 for root (last token).
    """
    if n == 1:
        return [-1]
    # ---- reverse orientation -> head-initial, root at 0
    srev = sc[::-1, ::-1]                    # srev[d, i] = arc dep=n-1-d head=n-1-i
    sibr = sibt[::-1, ::-1, ::-1]            # sibr[i, dp, d]
    fcr = fcs[::-1]
    NEG = -1e17
    C = np.zeros((n, n), dtype=np.float64)   # C[i][k] complete subtree rooted at i over [i,k]
    X = np.full((n, n), NEG, dtype=np.float64)  # X[i][d] = i + its children up to d (excl. d's subtree)
    BX = np.zeros((n, n), dtype=np.int32)
    BC = np.zeros((n, n), dtype=np.int32)
    for L in range(1, n):
        for i in range(0, n - L):
            d = i + L
            if L == 1:
                X[i, d] = fcr[i]
            else:
                cand = (X[i, i + 1:d] + C[i + 1:d, d - 1]
                        + srev[i + 1:d, i] + sibr[i, i + 1:d, d])
                j = int(np.argmax(cand))
                X[i, d] = cand[j]
                BX[i, d] = i + 1 + j
            cand2 = X[i, i + 1:d + 1] + C[i + 1:d + 1, d] + srev[i + 1:d + 1, i]
            j2 = int(np.argmax(cand2))
            C[i, d] = cand2[j2]
            BC[i, d] = i + 1 + j2
    heads = [-1] * n
    stack = [(0, n - 1)]
    while stack:
        i, k = stack.pop()
        if i >= k:
            continue
        d = int(BC[i, k])
        heads[n - 1 - d] = n - 1 - i
        if k > d:
            stack.append((d, k))
        cur = d
        while cur > i + 1:
            dp = int(BX[i, cur])
            heads[n - 1 - dp] = n - 1 - i
            if cur - 1 > dp:
                stack.append((dp, cur - 1))
            cur = dp
    heads[n - 1] = -1
    return heads


# --------------------------------------------------------------------------- #
# data loading
# --------------------------------------------------------------------------- #

def word_counts(dfs):
    """Raw-word -> count over the given dataframes (uses the digit-normalised core form)."""
    import collections
    c = collections.Counter()
    for df in dfs:
        for t in df["tokens"]:
            for raw in json.loads(t):
                core = raw.rstrip(_PUNCT) or raw
                c[_digitize(core)] += 1
    return c


def load(df, with_gold=True):
    out = []
    for r in df.itertuples(index=False):
        words = json.loads(r.tokens)
        n = len(words)
        item = {"id": r.id, "words": words, "n": n, "tl": toks_of(words)}
        if with_gold:
            items = r.parse.split("|")
            heads = []
            rels = []
            for x in items:
                a, b = x.split(":")
                a = int(a)
                heads.append(a - 1 if a > 0 else -1)
                rels.append(b)
            assert len(heads) == n
            item["gold_heads"] = heads
            item["gold_rels"] = rels
        out.append(item)
    return out

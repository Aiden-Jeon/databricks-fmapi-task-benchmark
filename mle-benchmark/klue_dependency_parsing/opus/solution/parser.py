"""KLUE-DP dependency parser.

Environment has no torch / GPU, so this is a classical graph-based parser:

  * Structural facts induced from train.csv (verified):
      - every tree is projective
      - exactly one root, always the LAST eojeol
      - every non-root head is to the RIGHT of its dependent (99.98% of arcs)
    => exact O(n^3) DP decoding over right-headed projective trees.

  * Scoring: first-order arc factored model
        score(dep i -> head j, label l) = w_u . f(i,j) + w_l[l] . g(i,j)
    trained with an averaged structured perceptron (labeled, so it directly
    optimises the LAS objective).

  * Features: hand-built Korean eojeol features (surface form, punctuation
    stripped suffixes of 1..3 chars, hangul jamo decomposition of the final
    syllable, data-driven "headness" bucket of a suffix, positional /
    distance / in-between-tag templates).

Usage:
    python parser.py --mode dev      # hold-out evaluation
    python parser.py --mode submit   # train on all data, write submission
"""
import argparse
import json
import os
import time
import zlib
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

U_BITS = int(os.environ.get("U_BITS", 24))   # hash space for unlabeled arc features
L_BITS = int(os.environ.get("L_BITS", 21))   # hash space for label features
U_SIZE = 1 << U_BITS
L_SIZE = 1 << L_BITS
U_MASK = U_SIZE - 1
L_MASK = L_SIZE - 1

# ---------------------------------------------------------------- hangul utils
CHO = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
JUNG = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
JONG = ["_"] + list("ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")


def jamo(ch):
    """(onset, nucleus, coda) of a hangul syllable, else ('#','#','#')."""
    o = ord(ch)
    if 0xAC00 <= o <= 0xD7A3:
        k = o - 0xAC00
        return CHO[k // 588], JUNG[(k % 588) // 28], JONG[k % 28]
    return "#", "#", "#"


PUNCT = set(".,!?\"'()[]{}<>«»“”‘’·…~-—:;/%")
DIGITS = set("0123456789")


def strip_punct(w):
    i = len(w)
    while i > 0 and w[i - 1] in PUNCT:
        i -= 1
    return w[:i], w[i:]


# ------------------------------------------------------------------ token atoms
class Lexicon:
    """Vocabularies / statistics estimated on the training portion only."""

    def __init__(self, sents, parses=None, min_word=3, min_suf=15):
        wc = Counter()
        s2c = Counter()
        s3c = Counter()
        for toks in sents:
            for w in toks:
                b, _ = strip_punct(w)
                wc[w] += 1
                s2c[b[-2:]] += 1
                s3c[b[-3:]] += 1
        self.wc = wc
        self.s2ok = {s for s, c in s2c.items() if c >= min_suf}
        self.s3ok = {s for s, c in s3c.items() if c >= min_suf}
        self.min_word = min_word
        # data driven "headness": P(token with this suffix is a head) -> bucket
        self.headness = {}
        if parses is not None:
            tot = Counter()
            ish = Counter()
            for toks, parse in zip(sents, parses):
                heads = [int(x.split(":")[0]) for x in parse.split("|")]
                hset = set(heads)
                for k, w in enumerate(toks):
                    b, _ = strip_punct(w)
                    key = b[-2:]
                    tot[key] += 1
                    if (k + 1) in hset:
                        ish[key] += 1
            for key, c in tot.items():
                if c >= 5:
                    p = ish[key] / c
                    self.headness[key] = min(4, int(p * 5))

    def atoms(self, w):
        """Return the atom dict/tuple for one eojeol."""
        b, pc = strip_punct(w)
        if not b:
            b = w
        s1, s2, s3 = b[-1:], b[-2:], b[-3:]
        if any(c in DIGITS for c in w):
            form = "<NUM>" if all((c in DIGITS or c in PUNCT) for c in w) else "<D>" + s1
        elif self.wc.get(w, 0) >= self.min_word:
            form = w
        else:
            form = "<U>" + s2
        tag = s2 if s2 in self.s2ok else s1
        fine = s3 if s3 in self.s3ok else tag
        o, n_, c_ = jamo(b[-1]) if b else ("#", "#", "#")
        stem = b[:-1] if len(b) > 1 else b
        if self.wc.get(stem, 0) < self.min_word:
            stem = "<u>"
        hn = str(self.headness.get(s2, 2))
        # tag enriched with punctuation and coda
        tag = tag + pc[:1]
        return (form, tag, fine, s1, s2, c_ + n_, hn, stem, pc[:1], str(min(len(b), 6)))


F_FORM, F_TAG, F_FINE, F_S1, F_S2, F_JM, F_HN, F_STEM, F_PC, F_LEN = range(10)

ROOT_ATOMS = ("<R>", "<R>", "<R>", "<R>", "<R>", "<R>", "4", "<R>", "", "0")
BOS_ATOMS = ("<B>", "<B>", "<B>", "<B>", "<B>", "<B>", "0", "<B>", "", "0")
EOS_ATOMS = ("<E>", "<E>", "<E>", "<E>", "<E>", "<E>", "0", "<E>", "", "0")


def dbucket(d):
    if d <= 5:
        return str(d)
    if d <= 7:
        return "6"
    if d <= 10:
        return "8"
    if d <= 15:
        return "11"
    return "16"


crc = zlib.crc32


# ------------------------------------------------------------ feature extraction
def sent_features(atoms, n):
    """atoms: list length n+2 (index 0 = virtual root, 1..n tokens, n+1 = EOS).

    Returns (uids, uoff, lids, loff, rows, cols, narcs) where arc order is
    [(i,j) for j in 2..n for i in 1..j-1] + [(n,0)].
    """
    A = atoms
    uids = []
    uoff = []
    lids = []
    loff = []
    rows = []
    cols = []

    def ctx(k):
        if k < 0:
            return BOS_ATOMS
        if k > n:
            return EOS_ATOMS
        return A[k]

    arcs = []
    for j in range(2, n + 1):
        for i in range(1, j):
            arcs.append((i, j))
    arcs.append((n, 0))

    nb = str(min(n // 5, 5))
    for (i, j) in arcs:
        rows.append(i)
        cols.append(j)
        d = A[i]
        h = A[j] if j > 0 else ROOT_ATOMS
        isroot = j == 0
        dist = dbucket(abs(j - i)) if not isroot else "R"
        uoff.append(len(uids))
        loff.append(len(lids))
        dw, dt, df, ds1, ds2, djm, dhn, dst, dpc, dln = d
        hw, ht, hf, hs1, hs2, hjm, hhn, hst, hpc, hln = h
        u = uids.append
        l = lids.append
        # ---- unlabeled arc templates
        u(crc(b"1" + hw.encode()) & U_MASK)
        u(crc(b"2" + ht.encode()) & U_MASK)
        u(crc(b"3" + hw.encode() + b"|" + ht.encode()) & U_MASK)
        u(crc(b"4" + dw.encode()) & U_MASK)
        u(crc(b"5" + dt.encode()) & U_MASK)
        u(crc(b"6" + dw.encode() + b"|" + dt.encode()) & U_MASK)
        hw_b, ht_b, dw_b, dt_b = hw.encode(), ht.encode(), dw.encode(), dt.encode()
        u(crc(b"7" + hw_b + b"|" + ht_b + b"|" + dw_b + b"|" + dt_b) & U_MASK)
        u(crc(b"8" + ht_b + b"|" + dw_b + b"|" + dt_b) & U_MASK)
        u(crc(b"9" + hw_b + b"|" + dw_b + b"|" + dt_b) & U_MASK)
        u(crc(b"10" + hw_b + b"|" + ht_b + b"|" + dt_b) & U_MASK)
        u(crc(b"11" + hw_b + b"|" + ht_b + b"|" + dw_b) & U_MASK)
        u(crc(b"12" + hw_b + b"|" + dw_b) & U_MASK)
        u(crc(b"13" + ht_b + b"|" + dt_b) & U_MASK)
        u(crc(b"14" + hf.encode() + b"|" + df.encode()) & U_MASK)
        u(crc(b"15" + hst.encode() + b"|" + dt_b) & U_MASK)
        u(crc(b"16" + ht_b + b"|" + dst.encode()) & U_MASK)
        u(crc(b"17" + hjm.encode() + b"|" + djm.encode()) & U_MASK)
        db = dist.encode()
        u(crc(b"20" + ht_b + b"|" + dt_b + b"|" + db) & U_MASK)
        u(crc(b"21" + dt_b + b"|" + db) & U_MASK)
        u(crc(b"22" + ht_b + b"|" + db) & U_MASK)
        u(crc(b"23" + hw_b + b"|" + dt_b + b"|" + db) & U_MASK)
        u(crc(b"24" + ht_b + b"|" + dw_b + b"|" + db) & U_MASK)
        u(crc(b"25" + hf.encode() + b"|" + df.encode() + b"|" + db) & U_MASK)
        # context templates
        hp1, hn1 = ctx(j - 1)[F_TAG].encode(), ctx(j + 1)[F_TAG].encode()
        dp1, dn1 = ctx(i - 1)[F_TAG].encode(), ctx(i + 1)[F_TAG].encode()
        u(crc(b"30" + ht_b + b"|" + hn1 + b"|" + dp1 + b"|" + dt_b) & U_MASK)
        u(crc(b"31" + hp1 + b"|" + ht_b + b"|" + dp1 + b"|" + dt_b) & U_MASK)
        u(crc(b"32" + ht_b + b"|" + hn1 + b"|" + dt_b + b"|" + dn1) & U_MASK)
        u(crc(b"33" + hp1 + b"|" + ht_b + b"|" + dt_b + b"|" + dn1) & U_MASK)
        u(crc(b"34" + ht_b + b"|" + dt_b + b"|" + dn1) & U_MASK)
        u(crc(b"35" + ht_b + b"|" + hp1 + b"|" + dt_b) & U_MASK)
        # positional
        u(crc(b"40" + dt_b + b"|" + (b"1" if isroot or j == n else b"0")) & U_MASK)
        u(crc(b"41" + dt_b + b"|" + (b"1" if i == 1 else b"0") + b"|" + ht_b) & U_MASK)
        u(crc(b"42" + dt_b + b"|" + ht_b + b"|" + nb.encode()) & U_MASK)
        u(crc(b"43" + dt_b + b"|" + dbucket(n - i).encode()) & U_MASK)
        u(crc(b"44" + ht_b + b"|" + dbucket(n - j).encode() if not isroot else b"44R") & U_MASK)
        # in-between
        if not isroot:
            seen = set()
            same_tag = 0
            hi_head = 0
            comma = 0
            for k in range(i + 1, j):
                tk = A[k][F_TAG]
                if tk not in seen:
                    seen.add(tk)
                    u(crc(b"50" + ht_b + b"|" + tk.encode() + b"|" + dt_b) & U_MASK)
                if tk == ht:
                    same_tag += 1
                if A[k][F_HN] >= "3":
                    hi_head += 1
                if A[k][F_PC] == ",":
                    comma += 1
            u(crc(b"51" + ht_b + b"|" + dt_b + b"|" + dbucket(same_tag).encode()) & U_MASK)
            u(crc(b"52" + ht_b + b"|" + dt_b + b"|" + dbucket(hi_head).encode()) & U_MASK)
            u(crc(b"53" + dt_b + b"|" + dbucket(hi_head).encode()) & U_MASK)
            u(crc(b"54" + ht_b + b"|" + dt_b + b"|" + dbucket(comma).encode()) & U_MASK)
            u(crc(b"55" + hhn.encode() + b"|" + dt_b + b"|" + dbucket(hi_head).encode()) & U_MASK)
            u(crc(b"56" + hhn.encode() + b"|" + dhn.encode() + b"|" + db) & U_MASK)
        else:
            u(crc(b"57root") & U_MASK)
        # ---- label templates (own hash space)
        l(crc(b"a" + dw_b) & L_MASK)
        l(crc(b"b" + dt_b) & L_MASK)
        l(crc(b"c" + df.encode()) & L_MASK)
        l(crc(b"d" + ds1.encode()) & L_MASK)
        l(crc(b"e" + ds2.encode()) & L_MASK)
        l(crc(b"f" + djm.encode()) & L_MASK)
        l(crc(b"g" + dst.encode()) & L_MASK)
        l(crc(b"h" + dpc.encode()) & L_MASK)
        l(crc(b"i" + ht_b) & L_MASK)
        l(crc(b"j" + hw_b) & L_MASK)
        l(crc(b"k" + hf.encode()) & L_MASK)
        l(crc(b"l" + hjm.encode()) & L_MASK)
        l(crc(b"m" + dt_b + b"|" + ht_b) & L_MASK)
        l(crc(b"n" + df.encode() + b"|" + hf.encode()) & L_MASK)
        l(crc(b"o" + dw_b + b"|" + ht_b) & L_MASK)
        l(crc(b"p" + dt_b + b"|" + hw_b) & L_MASK)
        l(crc(b"q" + dt_b + b"|" + db) & L_MASK)
        l(crc(b"r" + dt_b + b"|" + ht_b + b"|" + db) & L_MASK)
        l(crc(b"s" + dt_b + b"|" + (b"1" if isroot else (b"2" if j == n else b"0"))) & L_MASK)
        l(crc(b"t" + dt_b + b"|" + dn1) & L_MASK)
        l(crc(b"u" + dp1 + b"|" + dt_b) & L_MASK)
        l(crc(b"v" + dt_b + b"|" + (b"1" if i == 1 else b"0")) & L_MASK)
        l(crc(b"w" + df.encode() + b"|" + dbucket(n - i).encode()) & L_MASK)
        l(crc(b"x" + dhn.encode() + b"|" + hhn.encode()) & L_MASK)
        l(crc(b"y" + dt_b + b"|" + dln.encode()) & L_MASK)
        l(crc(b"z" + dpc.encode() + b"|" + ht_b) & L_MASK)
    uoff.append(len(uids))
    loff.append(len(lids))
    return (
        np.asarray(uids, dtype=np.int32),
        np.asarray(uoff, dtype=np.int64),
        np.asarray(lids, dtype=np.int32),
        np.asarray(loff, dtype=np.int64),
        np.asarray(rows, dtype=np.int64),
        np.asarray(cols, dtype=np.int64),
        len(arcs),
    )


# ---------------------------------------------------------------------- decoder
NEG = -1e18


def decode(arc, n):
    """arc: (n+1, n+1) matrix, arc[i, j] = score of arc dep i -> head j.

    Returns heads[1..n] for the best right-headed projective tree rooted at n.
    """
    if n == 1:
        return [0]
    A = np.zeros((n + 2, n + 2), dtype=np.float64)
    bp = np.zeros((n + 2, n + 2), dtype=np.int32)
    for j in range(2, n + 1):
        col = arc[:, j]
        for i in range(j - 1, 0, -1):
            vals = col[i:j] + A[i, i:j] + A[i + 1 : j + 1, j]
            k = int(np.argmax(vals))
            A[i, j] = vals[k]
            bp[i, j] = i + k
    heads = [0] * (n + 1)
    stack = [(1, n)]
    while stack:
        i, j = stack.pop()
        if i >= j:
            continue
        k = int(bp[i, j])
        heads[k] = j
        stack.append((i, k))
        stack.append((k + 1, j))
    heads[n] = 0
    return heads[1:]


# ------------------------------------------------------------------- model core
class Model:
    def __init__(self, nlab):
        self.nlab = nlab
        self.Wu = np.zeros(U_SIZE, dtype=np.float64)
        self.Au = np.zeros(U_SIZE, dtype=np.float64)
        self.Tu = np.zeros(U_SIZE, dtype=np.int32)
        self.Wl = np.zeros((L_SIZE, nlab), dtype=np.float64)
        self.Al = np.zeros((L_SIZE, nlab), dtype=np.float64)
        self.Tl = np.zeros(L_SIZE, dtype=np.int32)
        self.t = 1

    # lazy averaging ------------------------------------------------------
    def _touch_u(self, idx):
        dt = self.t - self.Tu[idx]
        self.Au[idx] += self.Wu[idx] * dt
        self.Tu[idx] = self.t

    def _touch_l(self, idx):
        dt = (self.t - self.Tl[idx]).astype(np.float64)
        self.Al[idx] += self.Wl[idx] * dt[:, None]
        self.Tl[idx] = self.t

    def update(self, uids, lids, lab, sign):
        uu = np.unique(uids)
        self._touch_u(uu)
        np.add.at(self.Wu, uids, sign)
        lu = np.unique(lids)
        self._touch_l(lu)
        np.add.at(self.Wl[:, lab], lids, sign)

    def finalize(self):
        allu = np.arange(U_SIZE)
        self.Au += self.Wu * (self.t - self.Tu)
        self.Tu[:] = self.t
        self.Al += self.Wl * (self.t - self.Tl)[:, None]
        self.Tl[:] = self.t
        del allu
        self.avg_u = (self.Au / self.t).astype(np.float32)
        self.avg_l = (self.Al / self.t).astype(np.float32)

    def raw(self, feats, use_avg=False):
        uids, uoff, lids, loff, rows, cols, narcs = feats
        Wu = self.avg_u if use_avg else self.Wu
        Wl = self.avg_l if use_avg else self.Wl
        us = np.add.reduceat(Wu[uids], uoff[:-1])
        ls = np.add.reduceat(Wl[lids], loff[:-1], axis=0)
        return ls + us[:, None]

    def score(self, feats, use_avg=False):
        tot = self.raw(feats, use_avg)
        best = np.argmax(tot, axis=1)
        bs = tot[np.arange(tot.shape[0]), best]
        return bs, best


def build_arc_matrix(bs, rows, cols, n):
    arc = np.full((n + 1, n + 1), NEG)
    arc[rows[:-1], cols[:-1]] = bs[:-1]
    return arc


# --------------------------------------------------------------------- data prep
def load(path):
    df = pd.read_csv(path)
    df["toks"] = df.tokens.map(json.loads)
    return df


def gold_of(parse):
    heads = []
    rels = []
    for it in parse.split("|"):
        a, b = it.split(":")
        heads.append(int(a))
        rels.append(b)
    return heads, rels


def prepare(df, lex, with_gold):
    """Return list of per-sentence dicts with cached features."""
    out = []
    for r in df.itertuples():
        toks = r.toks
        n = len(toks)
        atoms = [ROOT_ATOMS] + [lex.atoms(w) for w in toks] + [EOS_ATOMS]
        feats = sent_features(atoms, n)
        item = {"id": r.id, "n": n, "feats": feats}
        if with_gold:
            heads, rels = gold_of(r.parse)
            item["heads"] = heads
            item["rels"] = rels
            ok = heads[n - 1] == 0 and all(h > k + 1 for k, h in enumerate(heads[: n - 1]))
            if ok:
                # projectivity
                arcs = [(k + 1, h) for k, h in enumerate(heads) if h != 0]
                for a in range(len(arcs)):
                    x1, y1 = arcs[a]
                    for b in range(a + 1, len(arcs)):
                        x2, y2 = arcs[b]
                        if x1 < x2 < y1 < y2 or x2 < x1 < y2 < y1:
                            ok = False
                            break
                    if not ok:
                        break
            item["trainable"] = ok
        out.append(item)
    return out


def arc_index(n):
    """map (i,j) -> position in the arc list"""
    idx = {}
    c = 0
    for j in range(2, n + 1):
        for i in range(1, j):
            idx[(i, j)] = c
            c += 1
    idx[(n, 0)] = c
    return idx


# ------------------------------------------------------------------- train / run
def train(model, data, labmap, epochs, seed=0, log=True, dev=None, labels=None, margin=1.0):
    rng = np.random.RandomState(seed)
    order = np.arange(len(data))
    aidx_cache = {}
    for ep in range(epochs):
        rng.shuffle(order)
        nerr = 0
        ntok = 0
        t0 = time.time()
        for si in order:
            s = data[si]
            if not s["trainable"]:
                continue
            n = s["n"]
            feats = s["feats"]
            uids, uoff, lids, loff, rows, cols, narcs = feats
            if n not in aidx_cache:
                aidx_cache[n] = arc_index(n)
            aidx = aidx_cache[n]
            gh = s["heads"]
            gr = s["rels"]
            tot = model.raw(feats)
            if margin:
                # Hamming (LAS) cost augmented decoding: cost 0 only for the
                # gold (head,label) pair of each dependent, 1 elsewhere.
                for k in range(n):
                    tot[aidx[(k + 1, gh[k])], labmap[gr[k]]] -= margin
            best = np.argmax(tot, axis=1)
            bs = tot[np.arange(tot.shape[0]), best]
            if n == 1:
                pheads = [0]
            else:
                arc = build_arc_matrix(bs, rows, cols, n)
                pheads = decode(arc, n)
            model.t += 1
            upd = False
            for k in range(n):
                dep = k + 1
                ph = pheads[k]
                pa = aidx[(dep, ph)]
                pl = int(best[pa])
                gl = labmap[gr[k]]
                ntok += 1
                if ph == gh[k] and pl == gl:
                    continue
                nerr += 1
                upd = True
                ga = aidx[(dep, gh[k])]
                model.update(uids[uoff[ga] : uoff[ga + 1]], lids[loff[ga] : loff[ga + 1]], gl, 1.0)
                model.update(uids[uoff[pa] : uoff[pa + 1]], lids[loff[pa] : loff[pa + 1]], pl, -1.0)
            if not upd:
                pass
        if log:
            msg = "epoch %d  train-LAS %.4f  %.1fs" % (ep + 1, 1 - nerr / max(ntok, 1), time.time() - t0)
            if dev is not None:
                model.finalize()
                las, uas = evaluate(model, dev, labels, labmap)
                msg += "   dev LAS %.4f UAS %.4f" % (las, uas)
            print(msg, flush=True)


def predict(model, data, labels):
    res = []
    for s in data:
        n = s["n"]
        feats = s["feats"]
        bs, best = model.score(feats, use_avg=True)
        if n == 1:
            pheads = [0]
        else:
            arc = build_arc_matrix(bs, feats[4], feats[5], n)
            pheads = decode(arc, n)
        aidx = arc_index(n)
        parts = []
        for k in range(n):
            dep = k + 1
            h = pheads[k]
            lab = labels[int(best[aidx[(dep, h)]])]
            parts.append("%d:%s" % (h, lab))
        res.append((s["id"], "|".join(parts)))
    return res


def decode_from_scores(tot, n):
    """tot: (narcs, nlab) score matrix for one sentence -> (heads, label ids)"""
    best = np.argmax(tot, axis=1)
    bs = tot[np.arange(tot.shape[0]), best]
    if n == 1:
        pheads = [0]
    else:
        rows, cols = ARC_RC(n)
        arc = build_arc_matrix(bs, rows, cols, n)
        pheads = decode(arc, n)
    aidx = arc_index(n)
    labs = [int(best[aidx[(k + 1, pheads[k])]]) for k in range(n)]
    return pheads, labs


_RC = {}


def ARC_RC(n):
    if n not in _RC:
        rows = []
        cols = []
        for j in range(2, n + 1):
            for i in range(1, j):
                rows.append(i)
                cols.append(j)
        rows.append(n)
        cols.append(0)
        _RC[n] = (np.asarray(rows), np.asarray(cols))
    return _RC[n]


def ensemble_scores(tn, labels, labmap, targets, n_models, epochs, margin=1.0, seed0=0,
                    dev=None, verbose=True):
    """Train n_models perceptrons (different shuffles) and accumulate arc/label
    scores on every dataset in `targets` (list of prepared data lists)."""
    accs = [[np.zeros((len(s["feats"][4]), len(labels))) for s in tgt] for tgt in targets]
    for m in range(n_models):
        t0 = time.time()
        model = Model(len(labels))
        train(model, tn, labmap, epochs, seed=seed0 + m, log=verbose, margin=margin)
        model.finalize()
        for ti, tgt in enumerate(targets):
            for k, s in enumerate(tgt):
                accs[ti][k] += model.raw(s["feats"], use_avg=True)
        if verbose:
            print("  model %d done %.1fs" % (m + 1, time.time() - t0), flush=True)
            if dev is not None:
                di = dev
                las, uas = score_eval(accs[di], targets[di])
                print("  ensemble(%d) dev LAS %.4f UAS %.4f" % (m + 1, las, uas), flush=True)
        del model
    return accs


def score_eval(acc, data):
    ok = okh = tot = 0
    for a, s in zip(acc, data):
        n = s["n"]
        pheads, labs = decode_from_scores(a, n)
        for k in range(n):
            tot += 1
            if pheads[k] == s["heads"][k]:
                okh += 1
                if LABELS_G[labs[k]] == s["rels"][k]:
                    ok += 1
    return ok / tot, okh / tot


LABELS_G = None


def evaluate(model, data, labels, labmap):
    ok = 0
    okh = 0
    tot = 0
    for s in data:
        n = s["n"]
        feats = s["feats"]
        bs, best = model.score(feats, use_avg=True)
        if n == 1:
            pheads = [0]
        else:
            arc = build_arc_matrix(bs, feats[4], feats[5], n)
            pheads = decode(arc, n)
        aidx = arc_index(n)
        for k in range(n):
            tot += 1
            h = pheads[k]
            lab = labels[int(best[aidx[(k + 1, h)]])]
            if h == s["heads"][k]:
                okh += 1
                if lab == s["rels"][k]:
                    ok += 1
    return ok / tot, okh / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="dev")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--dev-size", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ensemble", type=int, default=0)
    ap.add_argument("--margin", type=float, default=1.0)
    args = ap.parse_args()

    tr = load(os.path.join(ROOT, "train.csv"))
    if args.limit:
        tr = tr.iloc[: args.limit].reset_index(drop=True)

    if args.mode == "dev":
        rng = np.random.RandomState(12345)
        perm = rng.permutation(len(tr))
        dv = tr.iloc[perm[: args.dev_size]].reset_index(drop=True)
        tn = tr.iloc[perm[args.dev_size :]].reset_index(drop=True)
    else:
        tn = tr
        dv = None

    labels = sorted({r for p in tn.parse for r in [x.split(":")[1] for x in p.split("|")]})
    labmap = {l: i for i, l in enumerate(labels)}
    print("labels", len(labels))

    t0 = time.time()
    lex = Lexicon(list(tn.toks), list(tn.parse))
    data_tn = prepare(tn, lex, True)
    print("prep train %.1fs" % (time.time() - t0), flush=True)
    data_dv = prepare(dv, lex, True) if dv is not None else None

    global LABELS_G
    LABELS_G = labels

    if args.ensemble:
        if dv is not None:
            accs = ensemble_scores(data_tn, labels, labmap, [data_dv], args.ensemble,
                                   args.epochs, margin=args.margin, seed0=args.seed, dev=0)
            las, uas = score_eval(accs[0], data_dv)
            print("FINAL ensemble dev LAS %.4f UAS %.4f" % (las, uas))
        else:
            te = load(os.path.join(ROOT, "test.csv"))
            data_te = prepare(te, lex, False)
            accs = ensemble_scores(data_tn, labels, labmap, [data_te], args.ensemble,
                                   args.epochs, margin=args.margin, seed0=args.seed)
            rows = []
            for a, s in zip(accs[0], data_te):
                pheads, labs = decode_from_scores(a, s["n"])
                rows.append((s["id"], "|".join("%d:%s" % (h, labels[l])
                                               for h, l in zip(pheads, labs))))
            out = pd.DataFrame(rows, columns=["id", "parse"])
            path = args.out or os.path.join(ROOT, "outputs", "submission.csv")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            out.to_csv(path, index=False)
            print("wrote", path, out.shape)
        return

    model = Model(len(labels))
    train(model, data_tn, labmap, args.epochs, seed=args.seed, dev=data_dv, labels=labels)
    model.finalize()

    if dv is not None:
        las, uas = evaluate(model, data_dv, labels, labmap)
        print("FINAL dev LAS %.4f UAS %.4f" % (las, uas))
    else:
        te = load(os.path.join(ROOT, "test.csv"))
        data_te = prepare(te, lex, False)
        res = predict(model, data_te, labels)
        out = pd.DataFrame(res, columns=["id", "parse"])
        path = args.out or os.path.join(ROOT, "outputs", "submission.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        out.to_csv(path, index=False)
        print("wrote", path, out.shape)


if __name__ == "__main__":
    main()

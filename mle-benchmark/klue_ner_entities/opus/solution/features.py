"""Character-level feature extraction for Korean NER (hashed features)."""
import sys
from collections import Counter, defaultdict
from zlib import crc32

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from data import LABELS  # noqa

HASH_BITS = 22
HASH_MASK = (1 << HASH_BITS) - 1
NFEAT = 1 << HASH_BITS
MAX_F = 48          # fixed feature slots per char (padded with bucket 0)
MAX_GAZ_LEN = 12


def h(s):
    # bucket 0 reserved as padding (weights stay zero)
    return (crc32(s.encode("utf8")) & HASH_MASK) or 1


def ctype(ch):
    if ch == " ":
        return "S"
    o = ord(ch)
    if 0xAC00 <= o <= 0xD7A3:
        return "H"
    if 0x3130 <= o <= 0x318F:
        return "J"
    if ch.isdigit():
        return "D"
    if "A" <= ch <= "Z":
        return "U"
    if "a" <= ch <= "z":
        return "L"
    if 0x4E00 <= o <= 0x9FFF:
        return "C"
    if 0x3040 <= o <= 0x30FF:
        return "K"
    if ch.isspace():
        return "S"
    return "P"


def jamo(ch):
    o = ord(ch) - 0xAC00
    if 0 <= o <= 11171:
        return o // 588, (o % 588) // 28, o % 28
    return -1, -1, -1


# ---------------------------------------------------------------- gazetteer
def build_gaz(rows, min_prec=0.2, min_count=1):
    """surface -> (type, prec_bucket) from labelled rows."""
    cnt = defaultdict(Counter)
    for r in rows:
        for s, t in r["ents"]:
            if 1 <= len(s) <= MAX_GAZ_LEN:
                cnt[s][t] += 1
    # count occurrences of each surface in all sentences (denominator)
    occ = Counter()
    surfaces = list(cnt)
    by_len = defaultdict(set)
    for s in surfaces:
        by_len[len(s)].add(s)
    lens = sorted(by_len)
    for r in rows:
        sent = r["sentence"]
        n = len(sent)
        for L in lens:
            bl = by_len[L]
            for i in range(n - L + 1):
                sub = sent[i:i + L]
                if sub in bl:
                    occ[sub] += 1
    gaz = {}
    for s, c in cnt.items():
        typ, _ = c.most_common(1)[0]
        tot = sum(c.values())
        if tot < min_count:
            continue
        prec = tot / max(occ[s], 1)
        if prec < min_prec:
            continue
        pb = "h" if prec >= 0.8 else ("m" if prec >= 0.5 else "l")
        gaz[s] = (typ, pb)
    return gaz


def gaz_matches(sent, gaz, gaz_lens):
    """Return list per position of feature strings from dictionary matches."""
    n = len(sent)
    out = [[] for _ in range(n)]
    for L in gaz_lens:
        if L > n:
            continue
        for i in range(n - L + 1):
            v = gaz.get(sent[i:i + L])
            if v is None:
                continue
            typ, pb = v
            lb = "1" if L == 1 else ("2" if L <= 3 else "3")
            for k in range(i, i + L):
                if L == 1:
                    pos = "U"
                elif k == i:
                    pos = "B"
                elif k == i + L - 1:
                    pos = "E"
                else:
                    pos = "I"
                out[k].append("g%s%s%s%s" % (pos, typ, pb, lb))
                out[k].append("G%s%s" % (pos, typ))
    return out


# ---------------------------------------------------------------- features
def sent_features(sent, gaz=None, gaz_lens=None, gmatch=None):
    n = len(sent)
    pad = "\x02\x02" + sent + "\x03\x03"
    # word (whitespace token) info per char
    widx = [0] * n
    words = []
    cur = []
    start = 0
    for i, ch in enumerate(sent):
        if ch == " ":
            if cur:
                words.append((start, "".join(cur)))
                cur = []
            widx[i] = -1
        else:
            if not cur:
                start = i
            cur.append(ch)
            widx[i] = len(words)
    if cur:
        words.append((start, "".join(cur)))
    if gmatch is None:
        gmatch = gaz_matches(sent, gaz, gaz_lens) if gaz else [[] for _ in range(n)]

    ids = np.zeros((n, MAX_F), dtype=np.int32)
    for i in range(n):
        p = i + 2  # index into pad
        c_2, c_1, c0, c1, c2 = pad[p - 2], pad[p - 1], pad[p], pad[p + 1], pad[p + 2]
        f = [
            "a" + c_2, "b" + c_1, "c" + c0, "d" + c1, "e" + c2,
            "f" + c_2 + c_1, "g" + c_1 + c0, "h" + c0 + c1, "i" + c1 + c2,
            "j" + c_2 + c_1 + c0, "k" + c_1 + c0 + c1, "l" + c0 + c1 + c2,
            "m" + c_1 + c1,
        ]
        t_1, t0, t1 = ctype(c_1), ctype(c0), ctype(c1)
        f += ["n" + t_1, "o" + t0, "p" + t1, "q" + t_1 + t0 + t1,
              "r" + ctype(pad[p - 2]) + t_1 + t0 + t1 + ctype(pad[p + 2])]
        ch0, ju0, jo0 = jamo(c0)
        f += ["s%d" % ch0, "t%d" % ju0, "u%d" % jo0]
        _, _, jo_1 = jamo(c_1)
        f += ["v%d" % jo_1, "w%d_%s" % (jo_1, c0)]
        # word features
        wi = widx[i]
        if wi >= 0:
            wstart, w = words[wi]
            off = i - wstart
            wl = len(w)
            f += ["x" + w,
                  "y%d" % (0 if off == 0 else (1 if off == wl - 1 else 2)),
                  "z" + w[:1], "A" + w[:2], "B" + w[-1:], "C" + w[-2:],
                  "D%d" % min(wl, 8),
                  "E%s_%d" % (w[-1:], 0 if off == 0 else 1),
                  "F%s_%s" % (w[:1], c0),
                  ]
            pw = words[wi - 1][1] if wi > 0 else "^"
            nw = words[wi + 1][1] if wi + 1 < len(words) else "$"
            f += ["I" + pw, "J" + nw, "K" + pw[-1:], "L" + nw[:1],
                  "M" + pw[-2:], "N" + nw[:2]]
        else:
            f += ["SP"]
        if i == 0:
            f.append("BOS")
        if i == n - 1:
            f.append("EOS")
        f += gmatch[i]
        m = min(len(f), MAX_F)
        for k in range(m):
            ids[i, k] = h(f[k])
    return ids

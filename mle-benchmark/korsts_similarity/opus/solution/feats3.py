"""Round-3 features: BM25, residual-IDF, thresholded embedding alignment, content-word overlap."""
import numpy as np
import pandas as pd
from collections import Counter
from feats import norm, nopunct, stoks, toks, char_sim, ngrams, jacc, dice


def bm25_sim(a, b, idf, idf_def, avgdl, k1=1.5, bmb=0.75):
    cb = Counter(b)
    dl = len(b)
    s = 0.0
    for t in set(a):
        f = cb.get(t, 0)
        if f == 0:
            continue
        w = idf.get(t, idf_def)
        s += w * f * (k1 + 1) / (f + k1 * (1 - bmb + bmb * dl / max(avgdl, 1e-9)))
    norm_ = sum(idf.get(t, idf_def) for t in set(a)) or 1.0
    return s / (norm_ * (k1 + 1))


def emb_align(a, b, Ea, Eb, wa, thr):
    """thresholded embedding alignment, ignoring exact matches boost"""
    if not a or not b:
        return 0.0
    tot = 0.0
    for i, u in enumerate(a):
        best = 0.0
        for j, v in enumerate(b):
            if u == v:
                best = 1.0
                break
            if Ea[i] is not None and Eb[j] is not None:
                c = float(np.dot(Ea[i], Eb[j]))
                if c > thr:
                    c = (c - thr) / (1 - thr)
                else:
                    c = 0.0
                if c > best:
                    best = c
        tot += wa[i] * best
    return tot / wa.sum()


def extra3(s1, s2, wv, idf, idf_def, avgdl, stop=None):
    stop = stop or set()
    rows = []
    for x, yy in zip(s1, s2):
        A, B = norm(x), norm(yy)
        a, b = stoks(A), stoks(B)
        wa = np.array([idf.get(t, idf_def) for t in a]) if a else np.array([1.0])
        wb = np.array([idf.get(t, idf_def) for t in b]) if b else np.array([1.0])
        Ea = [wv.vec(t) for t in a]
        Eb = [wv.vec(t) for t in b]

        # residual idf weight (unmatched content)
        sa, sb = set(a), set(b)
        res_a = sum(idf.get(t, idf_def) for t in sa - sb)
        res_b = sum(idf.get(t, idf_def) for t in sb - sa)
        tot_a = sum(idf.get(t, idf_def) for t in sa) or 1.0
        tot_b = sum(idf.get(t, idf_def) for t in sb) or 1.0
        # content words only (high idf)
        ca = {t for t in sa if t not in stop}
        cb_ = {t for t in sb if t not in stop}

        b25ab = bm25_sim(a, b, idf, idf_def, avgdl)
        b25ba = bm25_sim(b, a, idf, idf_def, avgdl)

        e1 = emb_align(a, b, Ea, Eb, wa, 0.5)
        e2 = emb_align(b, a, Eb, Ea, wb, 0.5)
        e3 = emb_align(a, b, Ea, Eb, wa, 0.75)
        e4 = emb_align(b, a, Eb, Ea, wb, 0.75)

        g4a, g4b = ngrams(nopunct(A), 4), ngrams(nopunct(B), 4)
        g5a, g5b = ngrams(nopunct(A), 5), ngrams(nopunct(B), 5)

        # first/last token match
        ft = 1.0 if (a and b and a[0] == b[0]) else 0.0
        lt = 1.0 if (a and b and a[-1] == b[-1]) else 0.0
        fchar = 1.0 if (A[:4] == B[:4]) else 0.0

        rows.append([
            b25ab, b25ba, (b25ab + b25ba) / 2, min(b25ab, b25ba), max(b25ab, b25ba),
            res_a / tot_a, res_b / tot_b, max(res_a / tot_a, res_b / tot_b),
            (res_a + res_b) / (tot_a + tot_b),
            np.log1p(res_a + res_b), np.log1p(min(res_a, res_b)), np.log1p(max(res_a, res_b)),
            jacc(ca, cb_), dice(ca, cb_), len(ca & cb_), len(ca | cb_),
            len(ca - cb_), len(cb_ - ca),
            e1, e2, (e1 + e2) / 2, min(e1, e2), e3, e4, (e3 + e4) / 2, min(e3, e4),
            jacc(g4a, g4b), dice(g4a, g4b), jacc(g5a, g5b), dice(g5a, g5b),
            ft, lt, fchar,
        ])
    names = ["bm25_ab", "bm25_ba", "bm25_mean", "bm25_min", "bm25_max",
             "res_a", "res_b", "res_max", "res_tot",
             "lres_sum", "lres_min", "lres_max",
             "cw_jacc", "cw_dice", "cw_inter", "cw_union", "cw_aonly", "cw_bonly",
             "ea50_ab", "ea50_ba", "ea50_mean", "ea50_min",
             "ea75_ab", "ea75_ba", "ea75_mean", "ea75_min",
             "g4_jacc", "g4_dice", "g5_jacc", "g5_dice",
             "first_tok_eq", "last_tok_eq", "first_chars_eq"]
    return pd.DataFrame(np.asarray(rows), columns=names)

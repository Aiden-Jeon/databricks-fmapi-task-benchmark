"""Second-generation feature set (superset of feats.build_features).

Adds:
  * word-boundary clause prefixes/suffixes (Korean leading adnominal clause carries
    the reference to the previous event, so "first 2 words" is a sharper cue than a
    fixed character fraction)
  * predicate-stem matching with the last sentence of the context
  * LSA (SVD of char tf-idf) cosine similarities for soft/semantic matching
"""
import re
import numpy as np
from itertools import permutations
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from feats import (split_sents, stem_words, pref, suff, rowcos, ngrams,
                   END_COLS, Vecs)


def words(t):
    return re.findall(r'\S+', str(t))


def wpref(t, k):
    w = words(t)
    return ' '.join(w[:k]) if w else str(t)


def wsuff(t, k):
    w = words(t)
    return ' '.join(w[-k:]) if w else str(t)


def stems(t, m=2):
    out = []
    for w in re.findall(r'[\uac00-\ud7a3A-Za-z0-9]+', str(t)):
        out.append(w[:m] if len(w) > m else w)
    return out


class Vecs2(Vecs):
    def __init__(self, texts, n_svd=150):
        super().__init__(texts)
        M = self.char.transform(texts)
        self.svd = TruncatedSVD(n_components=n_svd, random_state=0).fit(M)
        Mw = self.word.transform([stem_words(t) for t in texts])
        self.svdw = TruncatedSVD(n_components=120, random_state=0).fit(Mw)

    def dense(self, kind, texts):
        if kind == 'svd':
            return normalize(self.svd.transform(self.char.transform(texts)))
        return normalize(self.svdw.transform(
            self.word.transform([stem_words(t) for t in texts])))


def dcos(A, B):
    return (A * B).sum(1)


def build_features2(df, vecs):
    n = len(df)
    ctx = df['context'].astype(str).tolist()
    ends = [df[c].astype(str).tolist() for c in END_COLS]
    sl = [split_sents(c) for c in ctx]
    last1 = [s[-1] for s in sl]
    last2 = [' '.join(s[-2:]) for s in sl]
    first1 = [s[0] for s in sl]
    nsent = np.array([len(s) for s in sl], float)

    ctx_parts = {
        'ctx': ctx, 'l1': last1, 'l2': last2, 'f1': first1,
        'l1s': [suff(x, 0.5) for x in last1],
        'l1p': [pref(x, 0.5) for x in last1],
        'ctxs': [suff(x, 0.3) for x in ctx],
        'l1w2': [wsuff(x, 2) for x in last1],
        'l1w3': [wsuff(x, 3) for x in last1],
        'l1wp2': [wpref(x, 2) for x in last1],
    }
    cand_parts = {
        'full': ends,
        'p50': [[pref(x, 0.5) for x in e] for e in ends],
        'p35': [[pref(x, 0.35) for x in e] for e in ends],
        's50': [[suff(x, 0.5) for x in e] for e in ends],
        's35': [[suff(x, 0.35) for x in e] for e in ends],
        'w1': [[wpref(x, 1) for x in e] for e in ends],
        'w2': [[wpref(x, 2) for x in e] for e in ends],
        'w3': [[wpref(x, 3) for x in e] for e in ends],
        'z1': [[wsuff(x, 1) for x in e] for e in ends],
        'z2': [[wsuff(x, 2) for x in e] for e in ends],
    }

    feats = {}
    K = {}
    for kind in ('char', 'char5', 'word'):
        for cn, ct in ctx_parts.items():
            K[(kind, 'c_' + cn)] = vecs.t(kind, ct)
        for pn, pt in cand_parts.items():
            K[(kind, 'e_' + pn)] = [vecs.t(kind, e) for e in pt]
    for kind in ('svd', 'svdw'):
        for cn, ct in ctx_parts.items():
            K[(kind, 'c_' + cn)] = vecs.dense(kind, ct)
        for pn, pt in cand_parts.items():
            K[(kind, 'e_' + pn)] = [vecs.dense(kind, e) for e in pt]

    pairs = [('full', 'ctx'), ('full', 'l1'), ('full', 'l2'), ('full', 'f1'),
             ('p50', 'ctx'), ('p50', 'l1'), ('p50', 'l1s'), ('p50', 'l2'),
             ('p35', 'l1'), ('p35', 'l1s'), ('p35', 'ctxs'),
             ('s50', 'l1'), ('s50', 'ctx'), ('s35', 'l1'),
             ('w1', 'l1'), ('w2', 'l1'), ('w3', 'l1'), ('w2', 'l1s'),
             ('w2', 'ctx'), ('w2', 'l2'), ('w3', 'l1s'), ('w2', 'l1w2'),
             ('w3', 'l1w3'), ('z1', 'l1'), ('z2', 'l1'), ('z2', 'ctx'),
             ('full', 'l1w2'), ('w1', 'l1w2'), ('w2', 'l1wp2')]
    for kind in ('char', 'char5', 'word', 'svd', 'svdw'):
        cf = dcos if kind.startswith('svd') else rowcos
        for pn, cn in pairs:
            if kind.startswith('svd') and pn in ('s35',):
                continue
            E = K[(kind, 'e_' + pn)]; C = K[(kind, 'c_' + cn)]
            feats['%s_%s_%s' % (kind, pn, cn)] = np.stack(
                [cf(E[i], C) for i in range(4)], 1)

    # ---- directional dependency ------------------------------------------
    dep_mats = {}
    combos = [('p50', 's50', 'd5'), ('p35', 's50', 'd35'), ('full', 'full', 'sym'),
              ('w2', 'z2', 'w2z2'), ('w2', 'full', 'w2f'), ('w3', 'full', 'w3f'),
              ('w1', 'full', 'w1f')]
    for kind in ('char', 'char5', 'word', 'svd'):
        cf = dcos if kind.startswith('svd') else rowcos
        for pn, sn, tag in combos:
            P = K[(kind, 'e_' + pn)]; S = K[(kind, 'e_' + sn)]
            M = np.zeros((n, 4, 4))
            for i in range(4):
                for j in range(4):
                    if i == j:
                        continue
                    M[:, i, j] = cf(P[i], S[j])
            dep_mats[(kind, tag)] = M
            feats['%s_%s_out' % (kind, tag)] = M.sum(2)
            feats['%s_%s_outmax' % (kind, tag)] = M.max(2)
            feats['%s_%s_in' % (kind, tag)] = M.sum(1)
            feats['%s_%s_inmax' % (kind, tag)] = M.max(1)
            feats['%s_%s_diff' % (kind, tag)] = M.sum(1) - M.sum(2)

    # ---- structured chain ordering ---------------------------------------
    for kind, tag, cpn, cn in (('char', 'd5', 'p50', 'l1'), ('char5', 'd5', 'p50', 'l1'),
                               ('char', 'w2f', 'w2', 'l1'), ('svd', 'd5', 'p50', 'l1')):
        M = dep_mats[(kind, tag)]
        cf = dcos if kind.startswith('svd') else rowcos
        c0 = np.stack([cf(K[(kind, 'e_' + cpn)][i], K[(kind, 'c_' + cn)])
                       for i in range(4)], 1)
        best = np.full((n, 4), -1e9)
        for p in permutations(range(4)):
            sc = c0[:, p[0]].copy()
            for a, b in zip(p[:-1], p[1:]):
                sc = sc + M[:, b, a]
            np.maximum(best[:, p[0]], sc, out=best[:, p[0]])
        feats['perm_%s_%s' % (kind, tag)] = best
        feats['permrel_%s_%s' % (kind, tag)] = best - best.max(1, keepdims=True)

    # ---- raw n-gram overlap / novelty ------------------------------------
    for N in (2, 3, 4):
        cg = [ngrams(c, N) for c in ctx]
        lg = [ngrams(x, N) for x in last1]
        a = np.zeros((n, 4)); b = np.zeros((n, 4)); nv = np.zeros((n, 4))
        pa = np.zeros((n, 4))
        for i in range(4):
            for r in range(n):
                eg = ngrams(ends[i][r], N)
                pg = ngrams(wpref(ends[i][r], 2), N)
                a[r, i] = len(eg & cg[r]) / max(1, len(eg))
                b[r, i] = len(eg & lg[r]) / max(1, len(eg))
                pa[r, i] = len(pg & lg[r]) / max(1, len(pg))
                nv[r, i] = len(eg - cg[r])
        feats['ov%d_ctx' % N] = a
        feats['ov%d_l1' % N] = b
        feats['ov%d_p2l1' % N] = pa
        feats['nov%d' % N] = nv / 30.0

    # candidate-candidate raw overlap
    g3 = [[ngrams(ends[i][r], 3) for i in range(4)] for r in range(n)]
    cc = np.zeros((n, 4))
    for r in range(n):
        for i in range(4):
            cc[r, i] = sum(len(g3[r][i] & g3[r][j]) / max(1, len(g3[r][i]))
                           for j in range(4) if j != i)
    feats['cc3'] = cc

    # ---- predicate / stem matching ---------------------------------------
    l1_stems = [set(stems(x)) for x in last1]
    ctx_stems = [set(stems(x)) for x in ctx]
    verb = [stems(wsuff(x, 1))[0] if stems(wsuff(x, 1)) else '' for x in last1]
    e_stems = [[set(stems(ends[i][r])) for i in range(4)] for r in range(n)]
    p_stems = [[set(stems(wpref(ends[i][r], 2))) for i in range(4)] for r in range(n)]
    sv = np.zeros((n, 4)); sp = np.zeros((n, 4)); sc = np.zeros((n, 4))
    sn2 = np.zeros((n, 4)); vin = np.zeros((n, 4))
    for r in range(n):
        for i in range(4):
            sv[r, i] = 1.0 if verb[r] and verb[r] in e_stems[r][i] else 0.0
            vin[r, i] = 1.0 if verb[r] and verb[r] in p_stems[r][i] else 0.0
            sp[r, i] = len(p_stems[r][i] & l1_stems[r]) / max(1, len(p_stems[r][i]))
            sc[r, i] = len(e_stems[r][i] & l1_stems[r]) / max(1, len(e_stems[r][i]))
            sn2[r, i] = len(e_stems[r][i] - ctx_stems[r]) / max(1, len(e_stems[r][i]))
    feats['stem_verb_in_e'] = sv
    feats['stem_verb_in_pref'] = vin
    feats['stem_pref_l1'] = sp
    feats['stem_e_l1'] = sc
    feats['stem_new_frac'] = sn2

    # candidate own last-verb referenced by other candidates' prefixes
    everb = [[stems(wsuff(ends[i][r], 1))[0] if stems(wsuff(ends[i][r], 1)) else ''
              for i in range(4)] for r in range(n)]
    vref = np.zeros((n, 4)); vrefo = np.zeros((n, 4))
    for r in range(n):
        for i in range(4):
            vref[r, i] = sum(1.0 for j in range(4)
                             if j != i and everb[r][i] and everb[r][i] in p_stems[r][j])
            vrefo[r, i] = sum(1.0 for j in range(4)
                              if j != i and everb[r][j] and everb[r][j] in p_stems[r][i])
    feats['verb_ref_in'] = vref
    feats['verb_ref_out'] = vrefo

    # ---- surface ----------------------------------------------------------
    L = np.array([[len(ends[i][r]) for i in range(4)] for r in range(n)], float)
    feats['len'] = L / 30.0
    feats['len_rel'] = L / (L.mean(1, keepdims=True) + 1e-9)
    feats['len_vs_last'] = L / (np.array([len(x) for x in last1], float)[:, None] + 1e-9)
    W = np.array([[len(words(ends[i][r])) for i in range(4)] for r in range(n)], float)
    feats['nwords'] = W / 8.0
    feats['pos'] = np.tile(np.arange(4, dtype=float), (n, 1))
    feats['nsent'] = np.tile(nsent[:, None], (1, 4)) / 5.0
    for pat, nm in ((r',', 'comma'), (r'습니다', 'formal'), (r'다\.$', 'da'),
                    (r'(그리고|그러나|그래서|이윽고|계속|다시|후에|뒤|마지막|모두|다시)', 'conn'),
                    (r'(하고|하며|한 뒤|한 후|고 나서)', 'seq')):
        feats['mk_' + nm] = np.array([[1.0 if re.search(pat, ends[i][r]) else 0.0
                                       for i in range(4)] for r in range(n)])

    names = sorted(feats)
    X = np.stack([feats[k] for k in names], axis=2)
    return np.nan_to_num(X), names

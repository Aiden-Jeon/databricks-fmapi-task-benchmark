"""Feature engineering for KoBEST HellaSwag (train.csv only, no external data).

Key insight from data inspection: the 4 candidate endings form a temporal chain of
consecutive narrative events; the correct label is the event that comes *immediately*
after the context, and the distractors are events further down the chain.  Korean
sentences in this dataset typically restate the previous event in a leading
adnominal clause ("잠이 깬 아빠가 안방에서 나온다"), so directional similarity
between the *prefix* of one sentence and the *suffix* of another reveals the chain
order.  Features therefore focus on:
  * similarity of a candidate (and its prefix) to the context / its last sentence
  * a 4x4 directional dependency matrix between candidates (in/out degree)
  * a structured 4!-permutation chain-ordering search
"""
import re
import numpy as np
from itertools import permutations
from sklearn.feature_extraction.text import TfidfVectorizer

END_COLS = ['ending_%d' % i for i in range(1, 5)]


def split_sents(c):
    s = [x.strip() for x in re.split(r'(?<=[.!?])\s+', str(c).strip()) if x.strip()]
    return s if s else [str(c)]


def stem_words(t):
    """Crude Korean stemming: keep the leading 2 chars of every whitespace token
    (drops most josa / verb endings)."""
    out = []
    for w in re.findall(r'[\uac00-\ud7a3A-Za-z0-9]+', str(t)):
        out.append(w[:2] if len(w) > 2 else w)
    return ' '.join(out)


def pref(x, f):
    x = str(x)
    return x[:max(5, int(round(len(x) * f)))]


def suff(x, f):
    x = str(x)
    return x[-max(5, int(round(len(x) * f))):]


def rowcos(A, B):
    num = np.asarray(A.multiply(B).sum(1)).ravel()
    na = np.sqrt(np.asarray(A.multiply(A).sum(1)).ravel())
    nb = np.sqrt(np.asarray(B.multiply(B).sum(1)).ravel())
    return num / (na * nb + 1e-9)


class Vecs:
    """Container of fitted vectorizers (fit on all provided raw text, unsupervised)."""

    def __init__(self, texts):
        self.char = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4),
                                    min_df=2, sublinear_tf=True).fit(texts)
        self.char5 = TfidfVectorizer(analyzer='char', ngram_range=(3, 5),
                                     min_df=3, sublinear_tf=True).fit(texts)
        st = [stem_words(t) for t in texts]
        self.word = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1,
                                    sublinear_tf=True, token_pattern=r'\S+').fit(st)

    def t(self, kind, texts):
        if kind == 'char':
            return self.char.transform(texts)
        if kind == 'char5':
            return self.char5.transform(texts)
        return self.word.transform([stem_words(t) for t in texts])


def ngrams(s, n=3):
    s = re.sub(r'\s+', ' ', str(s))
    return set(s[i:i + n] for i in range(max(1, len(s) - n + 1)))


def build_features(df, vecs):
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
    }
    cand_parts = {
        'full': ends,
        'p50': [[pref(x, 0.5) for x in e] for e in ends],
        'p35': [[pref(x, 0.35) for x in e] for e in ends],
        's50': [[suff(x, 0.5) for x in e] for e in ends],
        's35': [[suff(x, 0.35) for x in e] for e in ends],
    }

    feats = {}
    K = {}  # cached matrices  (kind, name) -> matrix / list of matrices
    for kind in ('char', 'char5', 'word'):
        for cn, ct in ctx_parts.items():
            K[(kind, 'c_' + cn)] = vecs.t(kind, ct)
        for pn, pt in cand_parts.items():
            K[(kind, 'e_' + pn)] = [vecs.t(kind, e) for e in pt]

    # ---- candidate <-> context similarities -------------------------------
    pairs = [('full', 'ctx'), ('full', 'l1'), ('full', 'l2'), ('full', 'f1'),
             ('p50', 'ctx'), ('p50', 'l1'), ('p50', 'l1s'), ('p50', 'l2'),
             ('p35', 'l1'), ('p35', 'l1s'), ('p35', 'ctxs'),
             ('s50', 'l1'), ('s50', 'ctx'), ('s35', 'l1')]
    for kind in ('char', 'char5', 'word'):
        for pn, cn in pairs:
            E = K[(kind, 'e_' + pn)]
            C = K[(kind, 'c_' + cn)]
            feats['%s_%s_%s' % (kind, pn, cn)] = np.stack(
                [rowcos(E[i], C) for i in range(4)], 1)

    # ---- directional dependency between candidates ------------------------
    dep_mats = {}
    for kind in ('char', 'char5', 'word'):
        for pn, sn, tag in (('p50', 's50', 'd5'), ('p35', 's50', 'd35'),
                            ('full', 'full', 'sym')):
            P = K[(kind, 'e_' + pn)]
            S = K[(kind, 'e_' + sn)]
            M = np.zeros((n, 4, 4))
            for i in range(4):
                for j in range(4):
                    if i == j:
                        continue
                    M[:, i, j] = rowcos(P[i], S[j])
            dep_mats[(kind, tag)] = M
            feats['%s_%s_out' % (kind, tag)] = M.sum(2)
            feats['%s_%s_outmax' % (kind, tag)] = M.max(2)
            feats['%s_%s_in' % (kind, tag)] = M.sum(1)
            feats['%s_%s_inmax' % (kind, tag)] = M.max(1)
            feats['%s_%s_diff' % (kind, tag)] = M.sum(1) - M.sum(2)

    # ---- structured chain ordering (permutation search) -------------------
    for kind in ('char', 'char5'):
        M = dep_mats[(kind, 'd5')]
        # transition ctx -> candidate i
        c0 = np.stack([rowcos(K[(kind, 'e_p50')][i], K[(kind, 'c_l1')])
                       for i in range(4)], 1)
        best_first = np.full((n, 4), -1e9)
        perms = list(permutations(range(4)))
        for p in perms:
            sc = c0[:, p[0]].copy()
            for a, b in zip(p[:-1], p[1:]):
                sc = sc + M[:, b, a]
            np.maximum(best_first[:, p[0]], sc, out=best_first[:, p[0]])
        feats['%s_perm' % kind] = best_first
        feats['%s_perm_rel' % kind] = best_first - best_first.max(1, keepdims=True)

    # ---- raw n-gram overlap / novelty ------------------------------------
    for N in (2, 3, 4):
        cg = [ngrams(c, N) for c in ctx]
        lg = [ngrams(x, N) for x in last1]
        cont_ctx = np.zeros((n, 4)); cont_l1 = np.zeros((n, 4)); nov = np.zeros((n, 4))
        for i in range(4):
            for r in range(n):
                eg = ngrams(ends[i][r], N)
                cont_ctx[r, i] = len(eg & cg[r]) / max(1, len(eg))
                cont_l1[r, i] = len(eg & lg[r]) / max(1, len(eg))
                nov[r, i] = len(eg - cg[r])
        feats['ov%d_ctx' % N] = cont_ctx
        feats['ov%d_l1' % N] = cont_l1
        feats['nov%d' % N] = nov / 30.0

    # candidate-candidate raw overlap (undirected, 3-gram)
    g = [[ngrams(ends[i][r], 3) for i in range(4)] for r in range(n)]
    cc = np.zeros((n, 4))
    for r in range(n):
        for i in range(4):
            v = [len(g[r][i] & g[r][j]) / max(1, len(g[r][i])) for j in range(4) if j != i]
            cc[r, i] = sum(v)
    feats['cc3'] = cc

    # ---- surface features -------------------------------------------------
    L = np.array([[len(ends[i][r]) for i in range(4)] for r in range(n)], float)
    feats['len'] = L / 30.0
    feats['len_rel'] = L / (L.mean(1, keepdims=True) + 1e-9)
    feats['len_vs_last'] = L / (np.array([len(x) for x in last1], float)[:, None] + 1e-9)
    W = np.array([[len(str(ends[i][r]).split()) for i in range(4)] for r in range(n)], float)
    feats['nwords'] = W / 8.0
    feats['pos'] = np.tile(np.arange(4, dtype=float), (n, 1))
    feats['nsent'] = np.tile(nsent[:, None], (1, 4)) / 5.0
    # comma / connective markers (later events often more complex)
    for pat, nm in ((r',', 'comma'), (r'다\.$', 'da'), (r'습니다', 'formal'),
                    (r'(그리고|그러나|그래서|이윽고|계속|다시|후에|뒤)', 'conn')):
        feats['mk_' + nm] = np.array([[1.0 if re.search(pat, ends[i][r]) else 0.0
                                       for i in range(4)] for r in range(n)])

    names = sorted(feats)
    X = np.stack([feats[k] for k in names], axis=2)  # (n, 4, d)
    return X, names


def add_group_stats(X):
    """Augment with within-group normalisations (rank, z-score, gap-to-max)."""
    mu = X.mean(1, keepdims=True)
    sd = X.std(1, keepdims=True) + 1e-6
    z = (X - mu) / sd
    order = X.argsort(1).argsort(1).astype(float)  # rank within group
    gap = X - X.max(1, keepdims=True)
    return np.concatenate([X, z, order, gap], axis=2)

"""Extra features: distributional word vectors learned from train corpus + pair structure.

No labels are used here (only the fact that sentences are paired), so no CV leakage.
"""
import re
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from feats import norm, nopunct, stoks, toks, char_sim

NEG = ["않", "없", "아니", "못하", "못 ", "말고", " 안 ", "무", "비"]
NEGW = set("않 않다 안 없다 없는 없이 아니다 아닌 아니라 못 못한다 무 no not never none".split())
Q = set("무엇 어떻게 왜 누가 누구 언제 어디 어느 얼마 몇 무슨 어떤".split())


class WordVecs:
    """PPMI + truncated SVD word vectors from within-sentence and cross-pair contexts."""

    def __init__(self, dim=150, seed=0, cross_w=3.0):
        self.dim = dim
        self.seed = seed
        self.cross_w = cross_w

    def fit(self, s1, s2):
        A = [stoks(norm(x)) for x in s1]
        B = [stoks(norm(x)) for x in s2]
        cnt = Counter()
        for t in A + B:
            cnt.update(t)
        vocab = {w: i for i, (w, c) in enumerate(cnt.most_common()) if c >= 2}
        self.vocab = vocab
        V = len(vocab)
        rows, cols, vals = [], [], []

        def add(u, v, w):
            iu, iv = vocab.get(u), vocab.get(v)
            if iu is None or iv is None or iu == iv:
                return
            rows.append(iu); cols.append(iv); vals.append(w)
            rows.append(iv); cols.append(iu); vals.append(w)

        for a, b in zip(A, B):
            # within-sentence context (window 4)
            for t in (a, b):
                for i, u in enumerate(t):
                    for j in range(i + 1, min(len(t), i + 5)):
                        add(u, t[j], 1.0 / (j - i))
            # cross-pair context (paraphrase-ish signal)
            for u in a:
                for v in b:
                    add(u, v, self.cross_w)
        C = sparse.coo_matrix((vals, (rows, cols)), shape=(V, V)).tocsr()
        C.sum_duplicates()
        # PPMI
        tot = C.sum()
        rs = np.asarray(C.sum(1)).ravel() + 1e-9
        Cc = C.tocoo()
        pmi = np.log(np.maximum(Cc.data * tot / (rs[Cc.row] * rs[Cc.col]), 1e-12))
        pmi = np.maximum(pmi, 0)
        M = sparse.coo_matrix((pmi, (Cc.row, Cc.col)), shape=(V, V)).tocsr()
        M.eliminate_zeros()
        k = min(self.dim, min(M.shape) - 1)
        U, S, Vt = svds(M, k=k, random_state=self.seed)
        E = U * np.sqrt(np.maximum(S, 0))
        self.E = normalize(E)
        return self

    def vec(self, w):
        i = self.vocab.get(w)
        return self.E[i] if i is not None else None


def _pairsim_matrix(a, b, wv, use_char=True):
    """similarity matrix between token lists via wordvec cosine (fallback char)"""
    na, nb = len(a), len(b)
    Ea = [wv.vec(x) for x in a]
    Eb = [wv.vec(x) for x in b]
    S = np.zeros((na, nb))
    for i in range(na):
        for j in range(nb):
            if a[i] == b[j]:
                S[i, j] = 1.0
                continue
            s = 0.0
            if Ea[i] is not None and Eb[j] is not None:
                s = float(np.dot(Ea[i], Eb[j]))
            if use_char:
                s = max(s, char_sim(a[i], b[j]))
            S[i, j] = s
    return S


def extra_features(s1, s2, wv, idf, idf_def):
    rows = []
    for x, yy in zip(s1, s2):
        A = norm(x); B = norm(yy)
        a, b = stoks(A), stoks(B)
        wa = np.array([idf.get(t, idf_def) for t in a]) if a else np.zeros(0)
        wb = np.array([idf.get(t, idf_def) for t in b]) if b else np.zeros(0)

        # idf-weighted mean embedding cosine
        def meanvec(t, w):
            vs, ws = [], []
            for tt, ww in zip(t, w):
                v = wv.vec(tt)
                if v is not None:
                    vs.append(v); ws.append(ww)
            if not vs:
                return None
            M = np.asarray(vs); ws = np.asarray(ws)
            m = (M * ws[:, None]).sum(0) / ws.sum()
            n = np.linalg.norm(m)
            return m / n if n > 0 else None

        ma, mb = meanvec(a, wa), meanvec(b, wb)
        emb_cos = float(np.dot(ma, mb)) if (ma is not None and mb is not None) else 0.0

        # unweighted
        ma2, mb2 = meanvec(a, np.ones(len(a))), meanvec(b, np.ones(len(b)))
        emb_cos_u = float(np.dot(ma2, mb2)) if (ma2 is not None and mb2 is not None) else 0.0

        if a and b:
            S = _pairsim_matrix(a, b, wv, True)
            Sp = _pairsim_matrix(a, b, wv, False)
            ra = S.max(1); rb = S.max(0)
            al_ab = float((ra * wa).sum() / wa.sum())
            al_ba = float((rb * wb).sum() / wb.sum())
            pa = Sp.max(1); pb = Sp.max(0)
            pal_ab = float((pa * wa).sum() / wa.sum())
            pal_ba = float((pb * wb).sum() / wb.sum())
            # greedy 1-1 matching score
            Sc = S.copy(); used_r, used_c = set(), set()
            tot = 0.0; wtot = 0.0
            order = np.dstack(np.unravel_index(np.argsort(-Sc, axis=None), Sc.shape))[0]
            for i, j in order:
                if i in used_r or j in used_c:
                    continue
                used_r.add(i); used_c.add(j)
                w = (wa[i] + wb[j]) / 2
                tot += w * Sc[i, j]; wtot += w
                if len(used_r) == min(len(a), len(b)):
                    break
            match = tot / wtot if wtot else 0.0
            # order/monotonicity of best matches
            mi = S.argmax(1)
            if len(mi) > 1:
                d = np.diff(mi.astype(float))
                mono = float((d > 0).mean())
            else:
                mono = 1.0
            smax = float(S.max()); smean = float(S.mean())
        else:
            al_ab = al_ba = pal_ab = pal_ba = match = mono = smax = smean = 0.0

        ta, tb = toks(A), toks(B)
        nega = sum(1 for t in ta if any(n in t for n in NEG[:4])) + len(set(ta) & NEGW)
        negb = sum(1 for t in tb if any(n in t for n in NEG[:4])) + len(set(tb) & NEGW)
        qa = 1 if ("?" in A or set(ta) & Q) else 0
        qb = 1 if ("?" in B or set(tb) & Q) else 0

        rows.append([
            emb_cos, emb_cos_u, al_ab, al_ba, (al_ab + al_ba) / 2, min(al_ab, al_ba),
            pal_ab, pal_ba, (pal_ab + pal_ba) / 2, min(pal_ab, pal_ba),
            match, mono, smax, smean,
            min(nega, negb), max(nega, negb), abs(nega - negb), 1.0 if (nega > 0) != (negb > 0) else 0.0,
            min(qa, qb), max(qa, qb), abs(qa - qb),
        ])
    names = ["emb_cos", "emb_cos_u", "wv_al_ab", "wv_al_ba", "wv_al_mean", "wv_al_min",
             "wvp_al_ab", "wvp_al_ba", "wvp_al_mean", "wvp_al_min",
             "wv_match", "wv_mono", "wv_smax", "wv_smean",
             "neg_min", "neg_max", "neg_diff", "neg_xor",
             "q_min", "q_max", "q_diff"]
    return pd.DataFrame(np.asarray(rows), columns=names)

"""Pairwise "which event comes first" model.

For every row we know the gold ending precedes each of the three distractors, which
yields 6 labelled ordered pairs per row (3 with label 1, 3 with label 0) instead of a
single 4-way label.  Features are antisymmetric by construction so that the learned
score s(i,j) approximately satisfies s(i,j) = -s(j,i); the final candidate score is
sum_j s(i,j) (a Borda count over "comes first" votes).
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from feats import Vecs, split_sents, pref, suff, rowcos, END_COLS, ngrams


def build_pair_features(df, vecs):
    n = len(df)
    ends = [df[c].astype(str).tolist() for c in END_COLS]
    ctx = df['context'].astype(str).tolist()
    last1 = [split_sents(c)[-1] for c in ctx]

    parts = {
        'full': ends,
        'p50': [[pref(x, 0.5) for x in e] for e in ends],
        'p35': [[pref(x, 0.35) for x in e] for e in ends],
        's50': [[suff(x, 0.5) for x in e] for e in ends],
    }
    M = {}
    for kind in ('char', 'char5', 'word'):
        for pn, pt in parts.items():
            M[(kind, pn)] = [vecs.t(kind, e) for e in pt]
        M[(kind, 'l1')] = vecs.t(kind, last1)

    cols = []
    # directional similarity matrices
    for kind in ('char', 'char5', 'word'):
        for pn, sn in (('p50', 's50'), ('p35', 's50'), ('p50', 'full'),
                       ('p35', 'full'), ('full', 'full')):
            D = np.zeros((n, 4, 4))
            for i in range(4):
                for j in range(4):
                    D[:, i, j] = rowcos(M[(kind, pn)][i], M[(kind, sn)][j])
            cols.append(D)                       # dep(i<-j): i follows j
            cols.append(D.transpose(0, 2, 1))    # dep(j<-i)
            cols.append(D - D.transpose(0, 2, 1))

    # raw n-gram directional containment: prefix of i inside j (and vice versa)
    for N in (3, 4):
        gp = [[ngrams(pref(ends[i][r], 0.5), N) for i in range(4)] for r in range(n)]
        gf = [[ngrams(ends[i][r], N) for i in range(4)] for r in range(n)]
        gl = [ngrams(last1[r], N) for r in range(n)]
        D = np.zeros((n, 4, 4))
        for r in range(n):
            for i in range(4):
                for j in range(4):
                    D[r, i, j] = len(gp[r][i] & gf[r][j]) / max(1, len(gp[r][i]))
        cols.append(D); cols.append(D.transpose(0, 2, 1)); cols.append(D - D.transpose(0, 2, 1))
        # "prefix of i is covered by context but not by j" style contrast
        E = np.zeros((n, 4, 4))
        for r in range(n):
            for i in range(4):
                a = len(gp[r][i] & gl[r]) / max(1, len(gp[r][i]))
                for j in range(4):
                    b = len(gp[r][j] & gl[r]) / max(1, len(gp[r][j]))
                    E[r, i, j] = a - b
        cols.append(E)

    # third-party structure: does another candidate k depend on i vs on j?
    kind = 'char'
    D = np.zeros((n, 4, 4))
    for i in range(4):
        for j in range(4):
            D[:, i, j] = rowcos(M[(kind, 'p50')][i], M[(kind, 's50')][j])
    for stat in ('sum', 'max'):
        T = np.zeros((n, 4, 4))
        for i in range(4):
            for j in range(4):
                oth = [k for k in range(4) if k not in (i, j)]
                di = D[:, oth, i]; dj = D[:, oth, j]
                T[:, i, j] = (di.sum(1) - dj.sum(1)) if stat == 'sum' \
                    else (di.max(1) - dj.max(1))
        cols.append(T)

    P = np.stack(cols, axis=3)
    return P


def pair_matrix(X, P):
    """Concatenate antisymmetric candidate-feature differences with pair features."""
    n, k, d = X.shape
    diff = X[:, :, None, :] - X[:, None, :, :]
    return np.concatenate([diff, P], axis=3)


class PairModel:
    def __init__(self, random_state=0, **kw):
        self.rs = random_state
        self.kw = kw

    def fit(self, XP, y, sw=None):
        n = XP.shape[0]
        rows, lab = [], []
        for r in range(n):
            g = y[r]
            for j in range(4):
                if j == g:
                    continue
                rows.append(XP[r, g, j]); lab.append(1)
                rows.append(XP[r, j, g]); lab.append(0)
        Xf = np.asarray(rows); yf = np.asarray(lab)
        p = dict(max_iter=350, learning_rate=0.06, max_leaf_nodes=15,
                 min_samples_leaf=30, l2_regularization=1.0,
                 random_state=self.rs, early_stopping=False)
        p.update(self.kw)
        self.m = HistGradientBoostingClassifier(**p).fit(Xf, yf)
        return self

    def decision(self, XP):
        n, k = XP.shape[0], XP.shape[1]
        idx = [(i, j) for i in range(k) for j in range(k) if i != j]
        flat = np.stack([XP[:, i, j] for i, j in idx], 1).reshape(n * len(idx), -1)
        s = self.m.predict_proba(flat)[:, 1].reshape(n, len(idx))
        lg = np.log(np.clip(s, 1e-9, 1)) - np.log(np.clip(1 - s, 1e-9, 1))
        out = np.zeros((n, k))
        for t, (i, j) in enumerate(idx):
            out[:, i] += lg[:, t]
            out[:, j] -= lg[:, t]
        return out / (2 * (k - 1))

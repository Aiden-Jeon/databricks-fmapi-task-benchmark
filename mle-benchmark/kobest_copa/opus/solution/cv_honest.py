"""Honest CV: vectorizers + scaler fitted on training folds only (docs cached for speed)."""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import features as F
from cv_fast import load, docs
from features import numeric_block

_D = {}


def cached_docs(df, fn, name, tag):
    key = (name, tag)
    if key not in _D:
        _D[key] = (docs(df, 'alternative_1', fn), docs(df, 'alternative_2', fn))
    return _D[key]


def fold_matrix(df_tr, df_te, spec, tr_idx, num_tr, num_te, tag_tr='tr', tag_te='te'):
    Pt, Pe = [], []
    for name, fn, min_df, w in spec:
        if name == 'NUM':
            sc = StandardScaler().fit(np.vstack([num_tr[0][tr_idx], num_tr[1][tr_idx]]))
            a = sc.transform(num_tr[0]) - sc.transform(num_tr[1])
            b = sc.transform(num_te[0]) - sc.transform(num_te[1])
            s = np.sqrt((a[tr_idx] ** 2).sum(1)).mean()
            Pt.append(sparse.csr_matrix(a / s * w)); Pe.append(sparse.csr_matrix(b / s * w))
            continue
        t1, t2 = cached_docs(df_tr, fn, name, tag_tr)
        e1, e2 = cached_docs(df_te, fn, name, tag_te)
        vec = TfidfVectorizer(analyzer=lambda x: x.split(), min_df=min_df, sublinear_tf=True)
        vec.fit([t1[i] for i in tr_idx] + [t2[i] for i in tr_idx])
        Pt.append((vec.transform(t1) - vec.transform(t2)) * w)
        Pe.append((vec.transform(e1) - vec.transform(e2)) * w)
    return sparse.hstack(Pt).tocsr(), sparse.hstack(Pe).tocsr()


def honest_cv(tr, spec, models, seeds=(0, 1, 2), n_splits=5):
    y = tr.label.values; t = 1 - y
    num = (numeric_block(tr, 'alternative_1'), numeric_block(tr, 'alternative_2'))
    accs, oof = [], np.zeros((len(seeds), len(y)))
    for si, seed in enumerate(seeds):
        skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
        for a, b in skf.split(np.zeros(len(y)), y):
            X, _ = fold_matrix(tr, tr, spec, a, num, num, 'tr', 'tr')
            ps = []
            for kind, C in models:
                if kind == 'lr':
                    clf = LogisticRegression(C=C, max_iter=4000, solver='liblinear').fit(X[a], t[a])
                    ps.append(clf.predict_proba(X[b])[:, 1])
                else:
                    clf = LinearSVC(C=C, max_iter=20000).fit(X[a], t[a])
                    ps.append(1 / (1 + np.exp(-clf.decision_function(X[b]))))
            oof[si, b] = np.mean(ps, axis=0)
        accs.append(((oof[si] < 0.5).astype(int) == y).mean())
    return float(np.mean(accs)), float(np.std(accs)), oof


if __name__ == '__main__':
    tr, te = load()
    A = [('alt', F.alt_doc, 2, 1.0), ('x', F.cross_doc, 2, 1.0), ('NUM', None, 0, 1.0)]
    B = A + [('xw', F.cross_word, 2, 1.0)]
    cands = [
        ('A lr8', A, [('lr', 8.0)]),
        ('A lr8+svc.4', A, [('lr', 8.0), ('svc', 0.4)]),
        ('A lr4', A, [('lr', 4.0)]),
        ('B lr8+svc.4', B, [('lr', 8.0), ('svc', 0.4)]),
        ('B lr4', B, [('lr', 4.0)]),
    ]
    for nm, sp, mo in cands:
        m, s, _ = honest_cv(tr, sp, mo)
        print(f'{nm:16s} honest-CV acc={m:.4f} +-{s:.4f}')

"""Fast CV: vectorizers fit once on train+test docs (unsupervised), matrices cached."""
import itertools, numpy as np, pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from features import numeric_block, alt_doc, cross_doc

DATA = '..'
_cache = {}


def load():
    tr = pd.read_csv(f'{DATA}/train.csv')
    te = pd.read_csv(f'{DATA}/test.csv')
    for d in (tr, te):
        d['question'] = d.question.astype(str).str.strip()
    return tr, te


def docs(df, alt_col, fn):
    return [' '.join(fn(p, q, a)) for p, q, a in zip(df.premise, df.question, df[alt_col])]


def get_docs(name, fn, tr, te):
    key = ('docs', name)
    if key not in _cache:
        _cache[key] = tuple(docs(d, c, fn) for d in (tr, te) for c in ('alternative_1', 'alternative_2'))
    return _cache[key]


def block(name, fn, tr, te, min_df=2, sublinear=True, **vkw):
    key = ('blk', name, min_df, sublinear, tuple(sorted(vkw.items())))
    if key in _cache:
        return _cache[key]
    t1, t2, e1, e2 = get_docs(name, fn, tr, te)
    kw = dict(analyzer=lambda x: x.split(), min_df=min_df, sublinear_tf=sublinear, norm='l2')
    kw.update(vkw)
    vec = TfidfVectorizer(**kw)
    vec.fit(t1 + t2 + e1 + e2)
    D_tr = vec.transform(t1) - vec.transform(t2)
    D_te = vec.transform(e1) - vec.transform(e2)
    _cache[key] = (D_tr.tocsr(), D_te.tocsr(), len(vec.vocabulary_))
    return _cache[key]


def num_block(tr, te, scale=1.0):
    key = ('num',)
    if key not in _cache:
        N1t, N2t = numeric_block(tr, 'alternative_1'), numeric_block(tr, 'alternative_2')
        N1e, N2e = numeric_block(te, 'alternative_1'), numeric_block(te, 'alternative_2')
        sc = StandardScaler().fit(np.vstack([N1t, N2t]))
        Dt = sc.transform(N1t) - sc.transform(N2t)
        De = sc.transform(N1e) - sc.transform(N2e)
        # normalise magnitude so it is comparable to l2-normalised tfidf blocks
        s = np.sqrt((Dt ** 2).sum(1)).mean()
        _cache[key] = (Dt / s, De / s)
    Dt, De = _cache[key]
    return sparse.csr_matrix(Dt * scale), sparse.csr_matrix(De * scale)


def assemble(tr, te, spec):
    """spec: list of (name, fn, min_df, weight[, vkw]) plus optional ('NUM', None, 0, w)"""
    Pt, Pe = [], []
    for item in spec:
        name, fn, min_df, w = item[:4]
        vkw = item[4] if len(item) > 4 else {}
        if name == 'NUM':
            a, b = num_block(tr, te, w)
        else:
            a, b, _ = block(name, fn, tr, te, min_df=min_df, **vkw)
            a, b = a * w, b * w
        Pt.append(a); Pe.append(b)
    return sparse.hstack(Pt).tocsr(), sparse.hstack(Pe).tocsr()


def cv_probs(X, y, C, seeds=(0, 1, 2, 3, 4), n_splits=5, model='lr'):
    t = 1 - y
    P = np.zeros((len(seeds), len(y)))
    for si, seed in enumerate(seeds):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr_i, va_i in skf.split(X, y):
            if model == 'lr':
                clf = LogisticRegression(C=C, max_iter=4000, solver='liblinear')
            elif model == 'svc':
                clf = LinearSVC(C=C, max_iter=20000, dual=True)
            elif model == 'ridge':
                from sklearn.linear_model import RidgeClassifier
                clf = RidgeClassifier(alpha=C)
            clf.fit(X[tr_i], t[tr_i])
            if model == 'lr':
                P[si, va_i] = clf.predict_proba(X[va_i])[:, 1]
            else:
                P[si, va_i] = 1 / (1 + np.exp(-clf.decision_function(X[va_i])))
    return P


def score(P, y):
    accs = [((p < 0.5).astype(int) == y).mean() for p in P]
    return float(np.mean(accs)), float(np.std(accs))

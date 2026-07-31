"""Experiment: augment training folds with reverse-direction (cause<->effect) examples.

For a 원인 item  (premise=effect, correct=cause, wrong=w) we can create
  premise'=cause, question'=결과, correct'=effect, wrong'=w
and vice versa.  This doubles the supervision available to the
premise x alternative association features without any external data.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import features as F
from cv_fast import load, docs
from features import numeric_block

FLIP = {'원인': '결과', '결과': '원인'}


def make_aug(tr, seed=0):
    rng = np.random.RandomState(seed)
    correct = np.where(tr.label.values == 0, tr.alternative_1, tr.alternative_2)
    wrong = np.where(tr.label.values == 0, tr.alternative_2, tr.alternative_1)
    put_first = rng.rand(len(tr)) < 0.5
    a1 = np.where(put_first, tr.premise, wrong)
    a2 = np.where(put_first, wrong, tr.premise)
    return pd.DataFrame({
        'premise': correct,
        'question': [FLIP[q] for q in tr.question],
        'alternative_1': a1,
        'alternative_2': a2,
        'label': np.where(put_first, 0, 1),
    })


_D = {}


def cdocs(df, fn, name, tag):
    key = (name, tag)
    if key not in _D:
        _D[key] = (docs(df, 'alternative_1', fn), docs(df, 'alternative_2', fn))
    return _D[key]


def build(spec, dfs, tags, fit_rows, nums):
    """dfs[0] is the fitting/eval frame; returns delta matrices for every df."""
    parts = [[] for _ in dfs]
    for name, fn, min_df, w in spec:
        if name == 'NUM':
            n0 = nums[0]
            sc = StandardScaler().fit(np.vstack([n0[0][fit_rows], n0[1][fit_rows]]))
            s = None
            for i, nm in enumerate(nums):
                d = sc.transform(nm[0]) - sc.transform(nm[1])
                if s is None:
                    s = np.sqrt((d[fit_rows] ** 2).sum(1)).mean()
                parts[i].append(sparse.csr_matrix(d / s * w))
            continue
        dd = [cdocs(df, fn, name, tg) for df, tg in zip(dfs, tags)]
        vec = TfidfVectorizer(analyzer=lambda x: x.split(), min_df=min_df, sublinear_tf=True)
        fit_docs = []
        for d1, d2 in dd:
            fit_docs += [d1[i] for i in fit_rows] + [d2[i] for i in fit_rows]
        vec.fit(fit_docs)
        for i, (d1, d2) in enumerate(dd):
            parts[i].append((vec.transform(d1) - vec.transform(d2)) * w)
    return [sparse.hstack(p).tocsr() for p in parts]


def cv(tr, spec, C, use_aug=True, aug_w=1.0, seeds=(0, 1, 2), n_splits=5):
    y = tr.label.values; t = 1 - y
    aug = make_aug(tr, seed=0)
    num_tr = (numeric_block(tr, 'alternative_1'), numeric_block(tr, 'alternative_2'))
    num_ag = (numeric_block(aug, 'alternative_1'), numeric_block(aug, 'alternative_2'))
    ta = 1 - aug.label.values
    accs = []
    for seed in seeds:
        skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
        oof = np.zeros(len(y))
        for a, b in skf.split(np.zeros(len(y)), y):
            Xt, Xa = build(spec, [tr, aug], ['tr', 'aug'], a, [num_tr, num_ag])
            if use_aug:
                X = sparse.vstack([Xt[a], Xa[a]]); yy = np.r_[t[a], ta[a]]
                sw = np.r_[np.ones(len(a)), np.full(len(a), aug_w)]
            else:
                X, yy, sw = Xt[a], t[a], None
            clf = LogisticRegression(C=C, max_iter=6000, solver='liblinear')
            clf.fit(X, yy, sample_weight=sw)
            oof[b] = clf.predict_proba(Xt[b])[:, 1]
        accs.append(((oof < 0.5).astype(int) == y).mean())
    return float(np.mean(accs)), float(np.std(accs))


if __name__ == '__main__':
    tr, te = load()
    A = [('alt', F.alt_doc, 2, 1.0), ('x', F.cross_doc, 2, 1.0), ('NUM', None, 0, 1.0)]
    print('no aug  C=4 :', cv(tr, A, 4.0, use_aug=False))
    for w in [0.25, 0.5, 1.0]:
        for C in [4.0, 8.0]:
            print(f'aug w={w} C={C}:', cv(tr, A, C, use_aug=True, aug_w=w))

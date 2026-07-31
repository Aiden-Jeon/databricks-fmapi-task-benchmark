"""CV harness: evaluate antisymmetric delta-feature logistic models."""
import sys, numpy as np, pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from features import numeric_block, alt_doc, cross_doc

DATA = '..'


def load():
    tr = pd.read_csv(f'{DATA}/train.csv')
    te = pd.read_csv(f'{DATA}/test.csv')
    for d in (tr, te):
        d['question'] = d.question.astype(str).str.strip()
    return tr, te


def docs(df, alt_col, fn):
    return [' '.join(fn(p, q, a)) for p, q, a in zip(df.premise, df.question, df[alt_col])]


def build_delta(tr_idx, all_df, blocks, min_df=2, sub=1.0):
    """Return list of (X1, X2) sparse/dense delta blocks; vectorizers fitted on tr_idx rows only."""
    mats = []
    for name, fn in blocks:
        d1 = docs(all_df, 'alternative_1', fn)
        d2 = docs(all_df, 'alternative_2', fn)
        fit_docs = [d1[i] for i in tr_idx] + [d2[i] for i in tr_idx]
        vec = TfidfVectorizer(analyzer=lambda x: x.split(), min_df=min_df,
                              sublinear_tf=True, norm='l2')
        vec.fit(fit_docs)
        X1, X2 = vec.transform(d1), vec.transform(d2)
        mats.append((X1 - X2))
    return sparse.hstack(mats).tocsr() if mats else None


def eval_config(tr, blocks, use_num, C, min_df=2, n_splits=5, seeds=(0, 1, 2), verbose=True):
    y = tr.label.values
    t = 1 - y  # 1 if alternative_1 is correct
    accs = []
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        preds = np.zeros(len(tr))
        for tr_i, va_i in skf.split(tr, y):
            parts = []
            if blocks:
                Xd = build_delta(tr_i, tr, blocks, min_df=min_df)
                parts.append(Xd)
            if use_num:
                N1 = numeric_block(tr, 'alternative_1')
                N2 = numeric_block(tr, 'alternative_2')
                sc = StandardScaler().fit(np.vstack([N1[tr_i], N2[tr_i]]))
                Nd = sc.transform(N1) - sc.transform(N2)
                parts.append(sparse.csr_matrix(Nd))
            X = sparse.hstack(parts).tocsr()
            clf = LogisticRegression(C=C, max_iter=3000)
            clf.fit(X[tr_i], t[tr_i])
            preds[va_i] = clf.predict_proba(X[va_i])[:, 1]
        acc = ((preds < 0.5).astype(int) == y).mean()
        accs.append(acc)
    if verbose:
        print(f'  acc={np.mean(accs):.4f} +-{np.std(accs):.4f} {[round(a,4) for a in accs]}')
    return float(np.mean(accs))


if __name__ == '__main__':
    tr, te = load()
    print('majority(all-0):', (tr.label == 0).mean())
    print('numeric only:')
    for C in [0.03, 0.1, 0.3, 1.0]:
        print(' C=', C, end='');  eval_config(tr, [], True, C)
    print('alt_doc only:')
    for C in [0.1, 0.3, 1.0, 3.0]:
        print(' C=', C, end=''); eval_config(tr, [('alt', alt_doc)], False, C)
    print('cross only:')
    for C in [0.3, 1.0, 3.0]:
        print(' C=', C, end=''); eval_config(tr, [('x', cross_doc)], False, C)
    print('alt+cross+num:')
    for C in [0.3, 1.0, 3.0]:
        print(' C=', C, end=''); eval_config(tr, [('alt', alt_doc), ('x', cross_doc)], True, C)

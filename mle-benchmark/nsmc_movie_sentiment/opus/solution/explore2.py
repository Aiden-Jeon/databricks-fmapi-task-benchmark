"""Round 2: finer n-grams, C sweep, transductive vectorizers, self-training check."""
import time
import numpy as np
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score

from common import load_data, decompose_jamo

t0 = time.time()
tr, te = load_data()
txt = tr.text.values
y = tr.label.values
itr, iva = train_test_split(np.arange(len(y)), test_size=0.12, random_state=42, stratify=y)
tr_txt, va_txt = txt[itr], txt[iva]
ytr, yva = y[itr], y[iva]

# transductive: fit vectorizers on train-part + val-part text (no labels used)
fit_txt = np.concatenate([tr_txt, va_txt])
jam_tr = [decompose_jamo(t) for t in tr_txt]
jam_va = [decompose_jamo(t) for t in va_txt]
jam_fit = jam_tr + jam_va


def block(vec, a_txt, b_txt, fit_all):
    vec.fit(fit_all)
    return vec.transform(a_txt), vec.transform(b_txt)


configs = {
    'char15': (dict(analyzer='char_wb', ngram_range=(1, 5), min_df=3, sublinear_tf=True), False),
    'char26': (dict(analyzer='char_wb', ngram_range=(2, 6), min_df=3, sublinear_tf=True, max_features=1200000), False),
    'word13': (dict(analyzer='word', ngram_range=(1, 3), min_df=2, sublinear_tf=True, token_pattern=r'(?u)\S+'), False),
    'jamo26': (dict(analyzer='char', ngram_range=(2, 6), min_df=3, sublinear_tf=True, max_features=1000000), True),
    'jamo37': (dict(analyzer='char', ngram_range=(3, 7), min_df=3, sublinear_tf=True, max_features=1200000), True),
}
B = {}
for name, (kw, use_jamo) in configs.items():
    t = time.time()
    if use_jamo:
        B[name] = block(TfidfVectorizer(**kw), jam_tr, jam_va, jam_fit)
    else:
        B[name] = block(TfidfVectorizer(**kw), tr_txt, va_txt, fit_txt)
    print(f'{name}: {B[name][0].shape[1]} feats ({time.time()-t:.0f}s)')

print(f'features done {time.time()-t0:.0f}s')


def ev(names, C, model='lr'):
    Xtr = sparse.hstack([B[n][0] for n in names]).tocsr()
    Xva = sparse.hstack([B[n][1] for n in names]).tocsr()
    t = time.time()
    if model == 'lr':
        m = LogisticRegression(C=C, max_iter=1000, solver='liblinear')
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xva)[:, 1]
    else:
        m = LinearSVC(C=C)
        m.fit(Xtr, ytr)
        p = 1 / (1 + np.exp(-m.decision_function(Xva)))
    acc = accuracy_score(yva, (p > 0.5).astype(int))
    print(f'{model} C={C} {"+".join(names)[:40]:42s} acc={acc:.4f} ({time.time()-t:.0f}s)')
    return acc, p


store = {}
for names in [('char15', 'word13', 'jamo26'), ('char26', 'word13', 'jamo37'),
              ('char15', 'word13', 'jamo26', 'jamo37')]:
    for C in [0.5, 1, 2]:
        a, p = ev(names, C)
        store[f'lr{C}_{"+".join(names)}'] = (a, p)

for C in [0.2, 0.5]:
    a, p = ev(('char15', 'word13', 'jamo26'), C, 'svc')
    store[f'svc{C}'] = (a, p)

best_key = max(store, key=lambda k: store[k][0])
print('best', best_key, store[best_key][0])

# ---- self-training on the "unlabeled" val block using best config
names = ('char15', 'word13', 'jamo26')
Xtr = sparse.hstack([B[n][0] for n in names]).tocsr()
Xva = sparse.hstack([B[n][1] for n in names]).tocsr()
m = LogisticRegression(C=1, max_iter=1000, solver='liblinear')
m.fit(Xtr, ytr)
p = m.predict_proba(Xva)[:, 1]
print(f'base acc {accuracy_score(yva,(p>0.5).astype(int)):.4f}')
for thr in [0.9, 0.95]:
    conf = (p > thr) | (p < 1 - thr)
    pl = (p[conf] > 0.5).astype(int)
    Xaug = sparse.vstack([Xtr, Xva[conf]]).tocsr()
    yaug = np.concatenate([ytr, pl])
    w = np.concatenate([np.ones(len(ytr)), np.full(conf.sum(), 1.0)])
    m2 = LogisticRegression(C=1, max_iter=1000, solver='liblinear')
    m2.fit(Xaug, yaug, sample_weight=w)
    p2 = m2.predict_proba(Xva)[:, 1]
    print(f'self-train thr={thr} n_pl={conf.sum()} acc={accuracy_score(yva,(p2>0.5).astype(int)):.4f}')

np.save('/tmp/p2_keys.npy', np.array(list(store.keys())))
np.save('/tmp/p2.npy', np.vstack([store[k][1] for k in store]))
print(f'total {time.time()-t0:.0f}s')

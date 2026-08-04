"""Round 3: non-linear models (MLP on selected TF-IDF) for diversity beyond linear plateau."""
import time
import numpy as np
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from common import load_data, decompose_jamo

t0 = time.time()
CACHE = '/tmp/feat_holdout.npz'
tr, _ = load_data()
txt = tr.text.values
y = tr.label.values
itr, iva = train_test_split(np.arange(len(y)), test_size=0.12, random_state=42, stratify=y)
ytr, yva = y[itr], y[iva]

if not __import__('os').path.exists(CACHE):
    tr_txt, va_txt = txt[itr], txt[iva]
    fit_txt = np.concatenate([tr_txt, va_txt])
    jam_tr = [decompose_jamo(t) for t in tr_txt]
    jam_va = [decompose_jamo(t) for t in va_txt]
    mats_tr, mats_va = [], []
    for kw, jam in [
        (dict(analyzer='char_wb', ngram_range=(1, 5), min_df=3, sublinear_tf=True), False),
        (dict(analyzer='word', ngram_range=(1, 3), min_df=2, sublinear_tf=True, token_pattern=r'(?u)\S+'), False),
        (dict(analyzer='char', ngram_range=(2, 6), min_df=3, sublinear_tf=True, max_features=1000000), True),
    ]:
        v = TfidfVectorizer(**kw)
        if jam:
            v.fit(jam_tr + jam_va)
            mats_tr.append(v.transform(jam_tr)); mats_va.append(v.transform(jam_va))
        else:
            v.fit(fit_txt)
            mats_tr.append(v.transform(tr_txt)); mats_va.append(v.transform(va_txt))
        print('block', mats_tr[-1].shape, f'{time.time()-t0:.0f}s')
    Xtr = sparse.hstack(mats_tr).tocsr(); Xva = sparse.hstack(mats_va).tocsr()
    sparse.save_npz('/tmp/f_tr.npz', Xtr); sparse.save_npz('/tmp/f_va.npz', Xva)
    open(CACHE, 'w').close()
else:
    Xtr = sparse.load_npz('/tmp/f_tr.npz'); Xva = sparse.load_npz('/tmp/f_va.npz')
print('X', Xtr.shape, Xtr.nnz, f'{time.time()-t0:.0f}s')

res = {}
for K in [100000, 300000]:
    sel = SelectKBest(chi2, k=K).fit(Xtr, ytr)
    A, Bv = sel.transform(Xtr), sel.transform(Xva)
    m = LogisticRegression(C=1, max_iter=1000, solver='liblinear').fit(A, ytr)
    p = m.predict_proba(Bv)[:, 1]
    print(f'LR on chi2-{K}: {accuracy_score(yva,(p>0.5).astype(int)):.4f} ({time.time()-t0:.0f}s)')
    for hid in [(256,), (512,)]:
        t = time.time()
        mlp = MLPClassifier(hidden_layer_sizes=hid, alpha=1e-5, batch_size=256,
                            learning_rate_init=3e-4, max_iter=12, early_stopping=True,
                            n_iter_no_change=2, validation_fraction=0.08, random_state=0)
        mlp.fit(A, ytr)
        p = mlp.predict_proba(Bv)[:, 1]
        acc = accuracy_score(yva, (p > 0.5).astype(int))
        res[f'mlp{hid}_k{K}'] = p
        print(f'MLP {hid} chi2-{K}: {acc:.4f} iters={mlp.n_iter_} ({time.time()-t:.0f}s)')
    del A, Bv

np.save('/tmp/p3_keys.npy', np.array(list(res.keys())))
if res:
    np.save('/tmp/p3.npy', np.vstack([res[k] for k in res]))
print(f'total {time.time()-t0:.0f}s')

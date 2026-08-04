"""Holdout evaluation of feature blocks / models to pick the final recipe."""
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score

from common import load_data, build_feature_blocks, hstack_blocks, nb_ratio, apply_nb

t0 = time.time()
tr, _ = load_data()
txt = tr.text.values
y = tr.label.values

itr, iva = train_test_split(np.arange(len(y)), test_size=0.12, random_state=42, stratify=y)
tr_txt, va_txt = txt[itr], txt[iva]
ytr, yva = y[itr], y[iva]
print('split', len(itr), len(iva))

blocks = build_feature_blocks(tr_txt, va_txt)
print(f'features built {time.time()-t0:.0f}s')

combos = [('char',), ('word',), ('jamo',), ('char', 'word'), ('char', 'jamo'),
          ('char', 'word', 'jamo')]

results = {}
probs = {}
for names in combos:
    Xtr, Xva = hstack_blocks(blocks, names)
    for C in ([1, 4, 8] if len(names) > 1 else [4]):
        t = time.time()
        m = LogisticRegression(C=C, max_iter=1000, solver='liblinear')
        m.fit(Xtr, ytr)
        pv = m.predict_proba(Xva)[:, 1]
        acc = accuracy_score(yva, (pv > 0.5).astype(int))
        key = f"LR C={C} {'+'.join(names)}"
        results[key] = acc
        probs[key] = pv
        print(f'{key:38s} acc={acc:.4f}  ({time.time()-t:.0f}s)')

# models on the full block union
Xtr, Xva = hstack_blocks(blocks, ('char', 'word', 'jamo'))

t = time.time()
m = LinearSVC(C=0.5)
m.fit(Xtr, ytr)
d = m.decision_function(Xva)
results['SVC C=0.5'] = accuracy_score(yva, (d > 0).astype(int))
probs['SVC C=0.5'] = 1 / (1 + np.exp(-d))
print(f"SVC C=0.5 acc={results['SVC C=0.5']:.4f} ({time.time()-t:.0f}s)")

t = time.time()
m = SGDClassifier(loss='modified_huber', alpha=1e-6, max_iter=30, tol=1e-4, random_state=0)
m.fit(Xtr, ytr)
pv = m.predict_proba(Xva)[:, 1]
results['SGD mh'] = accuracy_score(yva, (pv > 0.5).astype(int))
probs['SGD mh'] = pv
print(f"SGD mh acc={results['SGD mh']:.4f} ({time.time()-t:.0f}s)")

t = time.time()
m = ComplementNB(alpha=0.3)
m.fit(Xtr, ytr)
pv = m.predict_proba(Xva)[:, 1]
results['CNB'] = accuracy_score(yva, (pv > 0.5).astype(int))
probs['CNB'] = pv
print(f"CNB acc={results['CNB']:.4f} ({time.time()-t:.0f}s)")

# NBSVM on the union
t = time.time()
r = nb_ratio(Xtr, ytr)
Xtrn, Xvan = apply_nb(Xtr, r), apply_nb(Xva, r)
for C in [0.1, 0.5, 1.0]:
    m = LogisticRegression(C=C, max_iter=1000, solver='liblinear')
    m.fit(Xtrn, ytr)
    pv = m.predict_proba(Xvan)[:, 1]
    acc = accuracy_score(yva, (pv > 0.5).astype(int))
    results[f'NBSVM C={C}'] = acc
    probs[f'NBSVM C={C}'] = pv
    print(f'NBSVM C={C} acc={acc:.4f} ({time.time()-t:.0f}s)')

print('\n--- ranked ---')
for k, v in sorted(results.items(), key=lambda x: -x[1]):
    print(f'{k:38s} {v:.4f}')

np.save('/tmp/probs_keys.npy', np.array(list(probs.keys())))
np.save('/tmp/probs.npy', np.vstack([probs[k] for k in probs]))
np.save('/tmp/yva.npy', yva)
print(f'total {time.time()-t0:.0f}s')

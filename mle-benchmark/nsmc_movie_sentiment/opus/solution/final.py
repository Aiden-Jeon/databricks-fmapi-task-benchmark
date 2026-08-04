"""Final NSMC solution.

Recipe (selected on a stratified 12% holdout, see explore*.py):
  Features (TF-IDF, fit on train+test text -- transductive, no test labels used):
    - char_wb 1-5 grams          (min_df=3, sublinear tf)
    - whitespace word 1-3 grams  (min_df=2, sublinear tf)
    - jamo-decomposed char 2-6 grams (min_df=3, sublinear tf) -> robust to Korean typos
  Models (rank-averaged ensemble):
    - LogisticRegression C=1 on the feature union
    - LinearSVC C=0.2 on the union
    - ComplementNB alpha=0.3 on the union
    - NBSVM (LogisticRegression C=1 on NB log-count-ratio weighted union)
    - LogisticRegression C=4 on jamo block only
Holdout accuracy: best single 0.8731 -> ensemble 0.8747
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB

from common import ROOT, load_data, decompose_jamo, nb_ratio, apply_nb

t0 = time.time()


def log(msg):
    print(f'[{time.time()-t0:6.0f}s] {msg}', flush=True)


tr, te = load_data()
ytr = tr.label.values
tr_txt, te_txt = tr.text.values, te.text.values
log(f'data: train={len(tr_txt)} test={len(te_txt)}')

jam_tr = [decompose_jamo(t) for t in tr_txt]
jam_te = [decompose_jamo(t) for t in te_txt]

blocks = {}
specs = [
    ('char', dict(analyzer='char_wb', ngram_range=(1, 5), min_df=3, sublinear_tf=True), False),
    ('word', dict(analyzer='word', ngram_range=(1, 3), min_df=2, sublinear_tf=True,
                  token_pattern=r'(?u)\S+'), False),
    ('jamo', dict(analyzer='char', ngram_range=(2, 6), min_df=3, sublinear_tf=True,
                  max_features=1000000), True),
]
for name, kw, use_jamo in specs:
    v = TfidfVectorizer(**kw)
    a_tr, a_te = (jam_tr, jam_te) if use_jamo else (tr_txt, te_txt)
    v.fit(list(a_tr) + list(a_te))          # unsupervised fit on all text
    blocks[name] = (v.transform(a_tr), v.transform(a_te))
    log(f'block {name}: {blocks[name][0].shape[1]} features')

Xtr = sparse.hstack([blocks[n][0] for n in ('char', 'word', 'jamo')]).tocsr()
Xte = sparse.hstack([blocks[n][1] for n in ('char', 'word', 'jamo')]).tocsr()
log(f'union: {Xtr.shape} nnz={Xtr.nnz}')

preds = {}

m = LogisticRegression(C=1, max_iter=1000, solver='liblinear').fit(Xtr, ytr)
preds['lr_union'] = m.predict_proba(Xte)[:, 1]
log('LR union done')

m = LinearSVC(C=0.2).fit(Xtr, ytr)
preds['svc_union'] = m.decision_function(Xte)
log('LinearSVC union done')

m = ComplementNB(alpha=0.3).fit(Xtr, ytr)
preds['cnb_union'] = m.predict_proba(Xte)[:, 1]
log('ComplementNB done')

r = nb_ratio(Xtr, ytr)
m = LogisticRegression(C=1, max_iter=1000, solver='liblinear').fit(apply_nb(Xtr, r), ytr)
preds['nbsvm'] = m.predict_proba(apply_nb(Xte, r))[:, 1]
log('NBSVM done')

m = LogisticRegression(C=4, max_iter=1000, solver='liblinear').fit(blocks['jamo'][0], ytr)
preds['lr_jamo'] = m.predict_proba(blocks['jamo'][1])[:, 1]
log('LR jamo done')


def to_rank(p):
    return np.argsort(np.argsort(p)) / (len(p) - 1)


score = np.mean([to_rank(p) for p in preds.values()], axis=0)
label = (score > 0.5).astype(int)

for k, p in preds.items():
    ind = (to_rank(p) > 0.5).astype(int)
    log(f'{k}: pos_rate={ind.mean():.4f} agree_with_ensemble={(ind==label).mean():.4f}')

out_dir = os.path.join(ROOT, 'outputs')
os.makedirs(out_dir, exist_ok=True)
sub = pd.DataFrame({'id': te.id.values, 'label': label})
sub.to_csv(os.path.join(out_dir, 'submission.csv'), index=False)
log(f'submission written: {sub.shape}, pos_rate={label.mean():.4f}')

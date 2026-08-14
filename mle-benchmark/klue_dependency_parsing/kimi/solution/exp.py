"""Experiment harness: cache pair features once, try multiple model configs."""
import pickle
import time
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from common import load_train, las
from parser import (build_arc_data, build_label_data, pair_feats,
                    tok_label_extra, uas, predict_labels)


def split(rows, seed=42, ndev=480):
    import random
    random.seed(seed)
    idx = list(range(len(rows)))
    random.shuffle(idx)
    dev = set(idx[:ndev])
    return ([r for k, r in enumerate(rows) if k not in dev],
            [r for k, r in enumerate(rows) if k in dev])


def main():
    t0 = time.time()
    rows = load_train('../train.csv')
    tr, dv = split(rows)
    print('building arc data...', flush=True)
    Xa, ya = build_arc_data(tr)
    va = DictVectorizer(sparse=True)
    Xav = va.fit_transform(Xa)
    del Xa
    print('arc data', Xav.shape, f'{time.time()-t0:.0f}s', flush=True)

    print('building label data...', flush=True)
    Xl, yl = build_label_data(tr)
    vl = DictVectorizer(sparse=True)
    Xlv = vl.fit_transform(Xl)
    del Xl
    print('label data', Xlv.shape, f'{time.time()-t0:.0f}s', flush=True)

    # precompute dev pair-feat matrices for fast head scoring
    dev_arc = []   # per sentence: list of (i, cand_feat_matrix)
    for r in dv:
        toks, n = r['tokens'], len(r['tokens'])
        per_i = []
        for i in range(n - 1):
            cand = [pair_feats(toks, i, j) for j in range(i + 1, n)]
            per_i.append(va.transform(cand))
        dev_arc.append(per_i)
    print('dev feats done', f'{time.time()-t0:.0f}s', flush=True)

    with open('_cache.pkl', 'wb') as f:
        pickle.dump({'va': va, 'vl': vl, 'Xav': Xav, 'ya': ya,
                     'Xlv': Xlv, 'yl': yl, 'dev_arc': dev_arc}, f,
                    protocol=4)
    print('cached', f'{time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()

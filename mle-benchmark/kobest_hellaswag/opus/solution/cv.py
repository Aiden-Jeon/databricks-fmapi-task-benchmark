"""Cross-validation experiment driver."""
import sys, time, os
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feats import Vecs, build_features, add_group_stats, END_COLS
from model import CondLogit, FlatClf, _softmax

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE = os.path.join(HERE, 'cache.npz')


def load():
    tr = pd.read_csv(os.path.join(ROOT, 'train.csv'))
    te = pd.read_csv(os.path.join(ROOT, 'test.csv'))
    return tr, te


def get_X(force=False):
    tr, te = load()
    if os.path.exists(CACHE) and not force:
        z = np.load(CACHE, allow_pickle=True)
        return tr, te, z['Xtr'], z['Xte'], list(z['names'])
    texts = list(tr.context.astype(str)) + list(te.context.astype(str))
    for c in END_COLS:
        texts += list(tr[c].astype(str)) + list(te[c].astype(str))
    t0 = time.time()
    vecs = Vecs(texts)
    Xtr, names = build_features(tr, vecs)
    Xte, _ = build_features(te, vecs)
    print('features %s in %.1fs' % (str(Xtr.shape), time.time() - t0))
    np.savez_compressed(CACHE, Xtr=Xtr, Xte=Xte, names=np.array(names))
    return tr, te, Xtr, Xte, names


def standardize(Xtr, Xte):
    n, k, d = Xtr.shape
    mu = Xtr.reshape(-1, d).mean(0)
    sd = Xtr.reshape(-1, d).std(0) + 1e-8
    return (Xtr - mu) / sd, (Xte - mu) / sd


def cv_oof(X, y, make, nfold=5, seeds=(0,)):
    n = len(y)
    oof = np.zeros((n, 4))
    for seed in seeds:
        skf = StratifiedKFold(nfold, shuffle=True, random_state=seed)
        for trn, val in skf.split(np.zeros(n), y):
            m = make().fit(X[trn], y[trn])
            oof[val] += _softmax(m.decision(X[val]))
    oof /= len(seeds)
    return oof, (oof.argmax(1) == y).mean()


if __name__ == '__main__':
    tr, te, Xtr, Xte, names = get_X(force='--force' in sys.argv)
    y = tr.label.values
    Xa = add_group_stats(Xtr)
    Xtr_s, _ = standardize(Xtr, Xte)
    Xa_s, _ = standardize(Xa, add_group_stats(Xte))
    for C in (0.03, 0.1, 0.3, 1.0, 3.0):
        _, a = cv_oof(Xtr_s, y, lambda C=C: CondLogit(C=C))
        print('condlogit raw C=%-5s %.4f' % (C, a))
    for C in (0.03, 0.1, 0.3, 1.0):
        _, a = cv_oof(Xa_s, y, lambda C=C: CondLogit(C=C))
        print('condlogit aug C=%-5s %.4f' % (C, a))
    _, a = cv_oof(Xa, y, lambda: FlatClf('hgb'))
    print('hgb aug %.4f' % a)
    _, a = cv_oof(Xtr, y, lambda: FlatClf('hgb'))
    print('hgb raw %.4f' % a)

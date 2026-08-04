"""CV driver with chain augmentation."""
import sys, os, time
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feats import Vecs, build_features, add_group_stats, END_COLS
from model import CondLogit, FlatClf, _softmax
from aug import augment

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
C2 = os.path.join(HERE, 'cache_aug.npz')


def prepare(force=False, n_aug_rounds=2):
    tr = pd.read_csv(os.path.join(ROOT, 'train.csv'))
    te = pd.read_csv(os.path.join(ROOT, 'test.csv'))
    if os.path.exists(C2) and not force:
        z = np.load(C2, allow_pickle=True)
        return tr, te, z['Xtr'], z['Xte'], z['Xag'], z['yag'], z['src'], list(z['names'])
    texts = list(tr.context.astype(str)) + list(te.context.astype(str))
    for c in END_COLS:
        texts += list(tr[c].astype(str)) + list(te[c].astype(str))
    t0 = time.time()
    vecs = Vecs(texts)
    Xtr, names = build_features(tr, vecs)
    Xte, _ = build_features(te, vecs)
    ags, srcs = [], []
    rng = np.random.default_rng(7)
    for _ in range(n_aug_rounds):
        a, s = augment(tr, rng)
        ags.append(a); srcs.append(s)
    ag = pd.concat(ags, ignore_index=True)
    src = np.concatenate(srcs)
    dedup = ~ag.duplicated(subset=['context'] + END_COLS + ['label'])
    ag, src = ag[dedup].reset_index(drop=True), src[dedup.values]
    Xag, _ = build_features(ag, vecs)
    yag = ag.label.values
    print('feat %s aug %s in %.1fs' % (str(Xtr.shape), str(Xag.shape), time.time() - t0))
    np.savez_compressed(C2, Xtr=Xtr, Xte=Xte, Xag=Xag, yag=yag, src=src,
                        names=np.array(names))
    return tr, te, Xtr, Xte, Xag, yag, src, names


def cv_aug(Xtr, y, Xag, yag, src, make, nfold=5, seeds=(0,), w_aug=1.0, use_aug=True):
    n = len(y)
    oof = np.zeros((n, 4))
    for seed in seeds:
        for trn, val in StratifiedKFold(nfold, shuffle=True, random_state=seed).split(np.zeros(n), y):
            if use_aug:
                mask = np.isin(src, trn)
                Xf = np.concatenate([Xtr[trn], Xag[mask]])
                yf = np.concatenate([y[trn], yag[mask]])
                sw = np.concatenate([np.ones(len(trn)), np.full(mask.sum(), w_aug)])
            else:
                Xf, yf, sw = Xtr[trn], y[trn], None
            m = make()
            try:
                m.fit(Xf, yf, sw)
            except TypeError:
                m.fit(Xf, yf)
            oof[val] += _softmax(m.decision(Xtr[val]))
    oof /= len(seeds)
    return oof, (oof.argmax(1) == y).mean()


if __name__ == '__main__':
    tr, te, Xtr, Xte, Xag, yag, src, names = prepare(force='--force' in sys.argv)
    y = tr.label.values
    A = add_group_stats(Xtr); Aag = add_group_stats(Xag)
    d = Xtr.shape[2]
    mu = Xtr.reshape(-1, d).mean(0); sd = Xtr.reshape(-1, d).std(0) + 1e-8
    Xs, Xags = (Xtr - mu) / sd, (Xag - mu) / sd
    for use in (False, True):
        _, a = cv_aug(Xs, y, Xags, yag, src, lambda: CondLogit(C=0.05), use_aug=use)
        print('condlogit aug=%s %.4f' % (use, a))
    for use in (False, True):
        _, a = cv_aug(A, y, Aag, yag, src, lambda: FlatClf('hgb'), use_aug=use)
        print('hgb   aug=%s %.4f' % (use, a))

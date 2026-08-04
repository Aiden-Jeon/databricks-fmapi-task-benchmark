"""End-to-end reproducible pipeline for KoBEST HellaSwag.

    python solution/run.py [--force]

Trains on train.csv only (no external data / pretrained weights) and writes
outputs/submission.csv.

Model: hand-crafted narrative-chain features (see feats2.py) fed to
  * HistGradientBoosting binary "is-correct" classifiers, scores re-normalised inside
    each 4-candidate group (two feature-normalisation views x several seeds)
  * a conditional-logit (softmax over the 4 candidates) linear ranker
blended with weights chosen by repeated 5-fold CV (OOF accuracy ~0.700).
"""
import os, sys, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)

from feats import END_COLS
from feats2 import Vecs2, build_features2
from model import CondLogit, FlatClf, _softmax

CACHE = os.path.join(HERE, 'cache_f2.npz')
W_CL = 0.15                    # weight of the conditional-logit model
HGB = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=15,
           min_samples_leaf=25, l2_regularization=1.0, early_stopping=False)
VIEWS = (('rg', (0, 1, 2, 3, 4)), ('zrg', (0, 1, 2)))


def group_stats(X, mode):
    parts = [X]
    if 'z' in mode:
        parts.append((X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-6))
    if 'r' in mode:
        parts.append(X.argsort(1).argsort(1).astype(float))
    if 'g' in mode:
        parts.append(X - X.max(1, keepdims=True))
    return np.concatenate(parts, 2)


def get_features(force=False):
    tr = pd.read_csv(os.path.join(ROOT, 'train.csv'))
    te = pd.read_csv(os.path.join(ROOT, 'test.csv'))
    if os.path.exists(CACHE) and not force:
        z = np.load(CACHE, allow_pickle=True)
        return tr, te, z['Xtr'], z['Xte']
    texts = list(tr.context.astype(str)) + list(te.context.astype(str))
    for c in END_COLS:
        texts += list(tr[c].astype(str)) + list(te[c].astype(str))
    t0 = time.time()
    vecs = Vecs2(texts)                     # unsupervised tf-idf / LSA on given text
    Xtr, names = build_features2(tr, vecs)
    Xte, _ = build_features2(te, vecs)
    print('features %s in %.1fs' % (Xtr.shape, time.time() - t0))
    np.savez_compressed(CACHE, Xtr=Xtr, Xte=Xte, names=np.array(names))
    return tr, te, Xtr, Xte


def main():
    tr, te, Xtr, Xte = get_features(force='--force' in sys.argv)
    y = tr.label.values
    probs, wsum = np.zeros((len(te), 4)), 0.0
    for mode, seeds in VIEWS:
        A, Ate = group_stats(Xtr, mode), group_stats(Xte, mode)
        p = np.zeros((len(te), 4))
        for s in seeds:
            m = FlatClf('hgb', random_state=s, **HGB).fit(A, y)
            p += _softmax(m.decision(Ate))
            print('  hgb[%s] seed %d done' % (mode, s), flush=True)
        probs += (1 - W_CL) / len(VIEWS) * p / len(seeds)
        wsum += (1 - W_CL) / len(VIEWS)
    d = Xtr.shape[2]
    mu = Xtr.reshape(-1, d).mean(0); sd = Xtr.reshape(-1, d).std(0) + 1e-8
    cl = CondLogit(C=0.05).fit((Xtr - mu) / sd, y)
    probs += W_CL * _softmax(cl.decision((Xte - mu) / sd))
    wsum += W_CL
    probs /= wsum

    out = pd.DataFrame({'id': te.id, 'label': probs.argmax(1)})
    os.makedirs(os.path.join(ROOT, 'outputs'), exist_ok=True)
    out.to_csv(os.path.join(ROOT, 'outputs', 'submission.csv'), index=False)
    np.save(os.path.join(HERE, 'test_probs.npy'), probs)
    assert len(out) == len(te) and out.id.is_unique
    print('wrote submission', out.shape, out.label.value_counts().sort_index().to_dict())


if __name__ == '__main__':
    main()

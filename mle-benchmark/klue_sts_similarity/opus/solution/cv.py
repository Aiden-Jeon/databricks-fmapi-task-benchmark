"""CV harness: build features once, cache, then evaluate models."""
import os, sys, time
import numpy as np, pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feats import STSFeaturizer

CACHE = "solution/_cache.npz"


def build(seed=0):
    tr = pd.read_csv("train.csv"); te = pd.read_csv("test.csv")
    corpus = pd.concat([tr.sentence1, tr.sentence2, te.sentence1, te.sentence2]).tolist()
    t0 = time.time()
    fz = STSFeaturizer(random_state=seed).fit(corpus)
    print("fit vectorizers %.1fs" % (time.time() - t0), flush=True)
    Xtr = fz.transform(tr.sentence1, tr.sentence2, verbose=True)
    print("train feats %s %.1fs" % (Xtr.shape, time.time() - t0), flush=True)
    Xte = fz.transform(te.sentence1, te.sentence2, verbose=True)
    print("test feats %s %.1fs" % (Xte.shape, time.time() - t0), flush=True)
    np.savez_compressed(CACHE, Xtr=Xtr, Xte=Xte, y=tr.score.values,
                        names=np.array(fz.feature_names_))
    return Xtr, Xte, tr.score.values


def load():
    if not os.path.exists(CACHE):
        return build()
    d = np.load(CACHE, allow_pickle=True)
    return d["Xtr"], d["Xte"], d["y"]


if __name__ == "__main__":
    build()

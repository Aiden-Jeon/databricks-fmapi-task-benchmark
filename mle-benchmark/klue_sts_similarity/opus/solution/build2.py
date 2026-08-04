import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feats2 import ExtraFeaturizer

tr = pd.read_csv("train.csv"); te = pd.read_csv("test.csv")
corpus = pd.concat([tr.sentence1, tr.sentence2, te.sentence1, te.sentence2]).tolist()
t0 = time.time()
fz = ExtraFeaturizer().fit(corpus)
print("fit %.1fs  wv vocab=%d dim=%d" % (time.time() - t0, len(fz.wvocab), fz.W.shape[1]), flush=True)
X2tr = fz.transform(tr.sentence1, tr.sentence2, verbose=True)
print("train2", X2tr.shape, "%.1fs" % (time.time() - t0), flush=True)
X2te = fz.transform(te.sentence1, te.sentence2, verbose=True)
print("test2", X2te.shape, "%.1fs" % (time.time() - t0), flush=True)
np.savez_compressed("solution/_cache2.npz", Xtr=X2tr, Xte=X2te,
                    names=np.array(fz.feature_names_))

"""Evaluate the effect of neighbour features on the holdout split."""
import sys, time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score

from dev3 import A, ytr, B, yva, ALL, assemble
from neighbors import NeighborFeatures

tr = pd.read_csv("../train.csv")
a, b, ya, yb = train_test_split(
    tr, tr["label"].values, test_size=0.2, random_state=42,
    stratify=tr["label"].values)
assert (ya == ytr).all() and (yb == yva).all()

Xa, Xb = assemble(ALL)
print("base", Xa.shape, flush=True)

nf = NeighborFeatures().fit(a, ya)
Na = nf.transform(a)   # self excluded -> leak free
Nb = nf.transform(b)
print("neigh", Na.shape, "nnz/row %.1f" % (Na.nnz / Na.shape[0]), flush=True)


def run(Xa_, Xb_, C, tag):
    m = LinearSVC(C=C, max_iter=4000, dual=True)
    t = time.time()
    m.fit(Xa_, ytr)
    p = m.predict(Xb_)
    acc = accuracy_score(yva, p)
    print(f"{tag} C={C}: acc={acc:.4f} f1m={f1_score(yva,p,average='macro'):.4f} ({time.time()-t:.0f}s)", flush=True)
    return acc


run(Xa, Xb, 0.2, "base    ")
for w in [1.0, 3.0, 6.0]:
    run(sp.hstack([Xa, Na * w], format="csr"), sp.hstack([Xb, Nb * w], format="csr"),
        0.2, f"+neigh w={w}")

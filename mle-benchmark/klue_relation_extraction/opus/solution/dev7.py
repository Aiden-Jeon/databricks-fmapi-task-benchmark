"""Fine sweep of neighbour-feature weight at the chosen C."""
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score

from dev3 import A, ytr, B, yva, ALL, assemble
from neighbors import NeighborFeatures

tr = pd.read_csv("../train.csv")
a, b, ya, yb = train_test_split(
    tr, tr["label"].values, test_size=0.2, random_state=42,
    stratify=tr["label"].values)
Xa0, Xb0 = assemble(ALL)
nf = NeighborFeatures().fit(a, ya)
Na, Nb = nf.transform(a), nf.transform(b)

for W in [0.5, 0.75, 1.25, 2.0]:
    Xa = sp.hstack([Xa0, Na * W], format="csr")
    Xb = sp.hstack([Xb0, Nb * W], format="csr")
    m = LinearSVC(C=0.15, max_iter=4000, dual=True)
    t = time.time()
    m.fit(Xa, ytr)
    p = m.predict(Xb)
    print(f"neigh_w={W}: acc={accuracy_score(yva,p):.4f} "
          f"f1m={f1_score(yva,p,average='macro'):.4f} ({time.time()-t:.0f}s)", flush=True)

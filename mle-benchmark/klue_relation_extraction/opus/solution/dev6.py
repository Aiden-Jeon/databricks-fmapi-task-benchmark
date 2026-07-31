"""Holdout sweep with neighbour features + blend search."""
import sys, time, pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import ComplementNB
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

W = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
Xa = sp.hstack([Xa0, Na * W], format="csr")
Xb = sp.hstack([Xb0, Nb * W], format="csr")
print("X", Xa.shape, "neigh_w", W, flush=True)

CLASSES = np.unique(ytr)


def zs(S):
    return (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)


def sc(m, X):
    if hasattr(m, "predict_proba"):
        P = np.log(np.clip(m.predict_proba(X), 1e-9, None))
    else:
        P = m.decision_function(X)
    return zs(P[:, [list(m.classes_).index(c) for c in CLASSES]])


S = {}


def ev(name, m):
    t = time.time()
    m.fit(Xa, ytr)
    Sv = sc(m, Xb)
    S[name] = Sv
    p = CLASSES[Sv.argmax(1)]
    print(f"{name}: acc={accuracy_score(yva,p):.4f} f1m={f1_score(yva,p,average='macro'):.4f} "
          f"({time.time()-t:.0f}s)", flush=True)


mode = sys.argv[1] if len(sys.argv) > 1 else "sweep"

if mode == "sweep":
    for C in [0.15, 0.2, 0.3, 0.5, 0.8]:
        ev(f"svc{C}", LinearSVC(C=C, max_iter=4000, dual=True))
elif mode == "blend":
    ev("svc02", LinearSVC(C=0.2, max_iter=4000, dual=True))
    ev("svc05", LinearSVC(C=0.5, max_iter=4000, dual=True))
    ev("svcbal", LinearSVC(C=0.25, max_iter=4000, dual=True, class_weight="balanced"))
    ev("sgdlog", SGDClassifier(loss="log_loss", alpha=1e-6, max_iter=40, n_jobs=4, random_state=0))
    ev("cnb", ComplementNB(alpha=0.3))
    with open(f"scores6_{W}.pkl", "wb") as f:
        pickle.dump((S, yva, CLASSES), f)
    print("\n--- greedy blend (with repetition) ---", flush=True)
    cur = np.zeros_like(next(iter(S.values())))
    best, chosen = 0.0, []
    for _ in range(10):
        bn, ba = None, best
        for n, M in S.items():
            acc = accuracy_score(yva, CLASSES[(cur + M).argmax(1)])
            if acc > ba + 1e-6:
                ba, bn = acc, n
        if bn is None:
            break
        cur += S[bn]; chosen.append(bn); best = ba
        print(f"  + {bn} -> {best:.4f}", flush=True)
    print("chosen:", chosen, flush=True)

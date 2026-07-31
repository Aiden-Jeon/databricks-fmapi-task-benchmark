"""Model zoo + blend search on holdout, using cached feature blocks."""
import sys, time, pickle
import numpy as np
import scipy.sparse as sp
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import normalize

from dev3 import A, ytr, B, yva, ALL, assemble

Xa, Xb = assemble(ALL)
Xan = normalize(Xa)
Xbn = normalize(Xb)
print("assembled", Xa.shape, flush=True)

CLASSES = np.unique(ytr)
c2i = {c: i for i, c in enumerate(CLASSES)}
yi = np.array([c2i[c] for c in ytr])


def dfun(m, X):
    """Return (n, n_classes) score matrix aligned with CLASSES."""
    if hasattr(m, "predict_proba"):
        try:
            P = m.predict_proba(X)
            P = np.log(np.clip(P, 1e-9, None))
        except Exception:
            P = m.decision_function(X)
    else:
        P = m.decision_function(X)
    idx = [list(m.classes_).index(c) for c in CLASSES]
    return P[:, idx]


def zs(S):
    return (S - S.mean(axis=1, keepdims=True)) / (S.std(axis=1, keepdims=True) + 1e-9)


def evalm(name, m, X1, X2, out):
    t = time.time()
    m.fit(X1, ytr)
    S = dfun(m, X2)
    p = CLASSES[S.argmax(1)]
    acc = accuracy_score(yva, p)
    out[name] = zs(S)
    print(f"{name}: acc={acc:.4f} f1m={f1_score(yva,p,average='macro'):.4f} ({time.time()-t:.0f}s)", flush=True)
    return acc


S = {}
mode = sys.argv[1] if len(sys.argv) > 1 else "zoo"

ZOO = [
    ("svc02", lambda: LinearSVC(C=0.2, max_iter=4000, dual=True), False),
    ("svc05", lambda: LinearSVC(C=0.5, max_iter=4000, dual=True), False),
    ("svcbal", lambda: LinearSVC(C=0.2, max_iter=4000, dual=True, class_weight="balanced"), False),
    ("sgdmh", lambda: SGDClassifier(loss="modified_huber", alpha=1e-6, max_iter=30, n_jobs=4, random_state=0), False),
    ("sgdlog", lambda: SGDClassifier(loss="log_loss", alpha=1e-6, max_iter=30, n_jobs=4, random_state=0), False),
    ("sgdmh2", lambda: SGDClassifier(loss="modified_huber", alpha=3e-7, max_iter=40, n_jobs=4, random_state=7), False),
    ("cnb", lambda: ComplementNB(alpha=0.3), False),
    ("mnb", lambda: MultinomialNB(alpha=0.05), False),
    ("knn", lambda: KNeighborsClassifier(n_neighbors=25, metric="cosine", weights="distance", n_jobs=4), True),
    ("lrliblin", lambda: LogisticRegression(C=2.0, solver="liblinear", max_iter=400), False),
]

if mode == "zoo":
    for name, fn, use_norm in ZOO:
        try:
            evalm(name, fn(), Xan if use_norm else Xa, Xbn if use_norm else Xb, S)
        except Exception as e:
            print(f"{name} FAILED: {e}", flush=True)
        with open("scores_dev.pkl", "wb") as f:
            pickle.dump((S, yva, CLASSES), f)

    # greedy blend search
    print("\n--- greedy blend ---", flush=True)
    cur = np.zeros_like(next(iter(S.values())))
    chosen = []
    best = 0.0
    for step in range(12):
        bn, ba = None, best
        for n, M in S.items():
            a = accuracy_score(yva, CLASSES[(cur + M).argmax(1)])
            if a > ba:
                ba, bn = a, n
        if bn is None:
            break
        cur = cur + S[bn]
        chosen.append(bn)
        best = ba
        print(f"  + {bn} -> {best:.4f}", flush=True)
    print("chosen:", chosen, flush=True)

"""Build holdout score matrices for several models, save for blend tuning."""
import time, sys
import numpy as np
import pandas as pd
from scipy import sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, RidgeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
sys.path.insert(0, "solution")
from features2 import build2, VEC_SPECS

OUT = "/tmp/opencode/"
t0 = time.time()
tr = pd.read_csv("train.csv")
Xt, Nt = build2(tr)
y = tr.label.values
classes = np.array(sorted(set(y)))
skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
tri, vai = next(iter(skf.split(Xt, y)))

A, B = [], []
for col, kw in VEC_SPECS:
    v = TfidfVectorizer(**kw)
    A.append(v.fit_transform(Xt[col].iloc[tri]))
    B.append(v.transform(Xt[col].iloc[vai]))
sc = StandardScaler()
A.append(sp.csr_matrix(sc.fit_transform(Nt[tri])))
B.append(sp.csr_matrix(sc.transform(Nt[vai])))
Xa = sp.hstack(A).tocsr()
Xb = sp.hstack(B).tocsr()
ya, yb = y[tri], y[vai]
np.save(OUT + "yb.npy", yb)
np.save(OUT + "classes.npy", classes)
print("feat", Xa.shape, round(time.time() - t0, 1), flush=True)

# kNN cosine on the combined tfidf space
s = time.time()
Xan = normalize(Xa)
Xbn = normalize(Xb)
cmap = {c: i for i, c in enumerate(classes)}
yai = np.array([cmap[c] for c in ya])
K = 30
knn = np.zeros((Xb.shape[0], len(classes)), dtype=np.float32)
knn1 = np.zeros_like(knn)
for st in range(0, Xb.shape[0], 1000):
    S = (Xbn[st:st + 1000] @ Xan.T).toarray()
    idx = np.argpartition(-S, K, axis=1)[:, :K]
    for r in range(S.shape[0]):
        sv = S[r, idx[r]]
        np.add.at(knn[st + r], yai[idx[r]], sv ** 3)
        np.add.at(knn1[st + r], yai[idx[r]], sv ** 6)
np.save(OUT + "knn.npy", knn)
np.save(OUT + "knn1.npy", knn1)
print("knn", round(accuracy_score(yb, classes[knn.argmax(1)]), 5),
      round(accuracy_score(yb, classes[knn1.argmax(1)]), 5), round(time.time() - s, 1), flush=True)

for name, clf in [

    ("svc03", LinearSVC(C=0.3, dual=True)),
    ("sgd", SGDClassifier(loss="modified_huber", alpha=2e-7, max_iter=30, tol=None, random_state=0)),
]:
    s = time.time()
    clf.fit(Xa, ya)
    M = clf.decision_function(Xb)
    assert np.array_equal(clf.classes_, classes)
    np.save(OUT + f"{name}.npy", M.astype(np.float32))
    print(name, round(accuracy_score(yb, classes[M.argmax(1)]), 5), round(time.time() - s, 1), flush=True)
print("total", round(time.time() - t0, 1))

"""Sparse pair-representation ridge on jamo + word-bigram spaces (diverse view)."""
import os, sys, time
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.stats import pearsonr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feats import clean, depunct, stems
from feats2 import to_jamo

tr = pd.read_csv("train.csv"); te = pd.read_csv("test.csv")
y = tr.score.values
ntr = len(tr)
S1 = [clean(s) for s in pd.concat([tr.sentence1, te.sentence1])]
S2 = [clean(s) for s in pd.concat([tr.sentence2, te.sentence2])]
J1 = [to_jamo(depunct(s)) for s in S1]
J2 = [to_jamo(depunct(s)) for s in S2]
T1 = [" ".join(stems(s)) for s in S1]
T2 = [" ".join(stems(s)) for s in S2]

SPECS = [
    ("jamo", (J1, J2), dict(analyzer="char", ngram_range=(3, 5), min_df=3, sublinear_tf=True)),
    ("wbi", (T1, T2), dict(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"\S+")),
]
blocks = []
for name, (a, b), kw in SPECS:
    v = TfidfVectorizer(**kw).fit(a + b)
    A = normalize(v.transform(a)); B = normalize(v.transform(b))
    blocks += [A.minimum(B), abs(A - B)]
    print(name, A.shape[1], flush=True)

X = sp.hstack(blocks).tocsr().astype(np.float32)
print("pair matrix", X.shape, X.nnz, flush=True)
Xtr, Xte = X[:ntr], X[ntr:]
kf = list(KFold(5, shuffle=True, random_state=42).split(Xtr))
best = None
for alpha in (0.5, 1.0, 3.0):
    oof = np.zeros(ntr); test = np.zeros(Xte.shape[0]); t0 = time.time()
    for i_tr, i_va in kf:
        m = Ridge(alpha=alpha, solver="sparse_cg", tol=1e-4).fit(Xtr[i_tr], y[i_tr])
        oof[i_va] = m.predict(Xtr[i_va]); test += m.predict(Xte) / 5
    r = pearsonr(oof, y)[0]
    print(f"  alpha={alpha}: {r:.5f} ({time.time()-t0:.0f}s)", flush=True)
    if best is None or r > best[0]:
        best = (r, alpha, oof, test)
print("best sparse2", best[0], "alpha", best[1])
np.savez_compressed("solution/_sparse2.npz", oof=best[2], test=best[3],
                    alpha=best[1], r=best[0])

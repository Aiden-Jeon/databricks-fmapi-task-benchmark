import numpy as np, pandas as pd
from scipy import sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from clogit import CondLogit
from feats import build_features, OPTS

RS = 42
tr = pd.read_csv("../train.csv")
n = len(tr); y = tr.label.values - 1
opt = [str(tr[c].iloc[i]) for i in range(n) for c in OPTS]
Xd, _, _ = build_features(tr)
kf = KFold(5, shuffle=True, random_state=RS)
folds = list(kf.split(np.arange(n)))
def expand(i): return (i[:, None] * 4 + np.arange(4)).ravel()
def acc(s): return (s.reshape(n, 4).argmax(1) == y).mean()


def run(make_X, Cs, name):
    for C in Cs:
        oof = np.zeros(n * 4)
        for tri, vai in folds:
            A, B = make_X(tri, vai)
            m = CondLogit(C=C, max_iter=400).fit(A, y[tri])
            oof[expand(vai)] = m.decision(B)
        print(f"{name} C={C} acc={acc(oof):.4f}", flush=True)
        np.save(f"/tmp/oof_cl_{name}_{C}.npy", oof)


sc = StandardScaler().fit(Xd)
Xds = sc.transform(Xd)
Xds = np.hstack([Xds, np.ones((n * 4, 1))])
run(lambda tri, vai: (Xds[expand(tri)], Xds[expand(vai)]), [0.03, 0.1, 0.3, 1.0], "dense")


def mk_text(ngr, an, mindf):
    def f(tri, vai):
        v = TfidfVectorizer(analyzer=an, ngram_range=ngr, min_df=mindf, sublinear_tf=True)
        A = v.fit_transform([opt[i] for i in expand(tri)])
        B = v.transform([opt[i] for i in expand(vai)])
        return A, B
    return f


run(mk_text((2, 4), "char_wb", 3), [0.3, 1.0, 3.0], "chartxt")


def mk_both(ngr, an, mindf, scale=0.5):
    def f(tri, vai):
        v = TfidfVectorizer(analyzer=an, ngram_range=ngr, min_df=mindf, sublinear_tf=True)
        A = v.fit_transform([opt[i] for i in expand(tri)])
        B = v.transform([opt[i] for i in expand(vai)])
        A = sp.hstack([sp.csr_matrix(Xds[expand(tri)] * scale), A]).tocsr()
        B = sp.hstack([sp.csr_matrix(Xds[expand(vai)] * scale), B]).tocsr()
        return A, B
    return f


run(mk_both((2, 4), "char_wb", 3), [0.3, 1.0, 3.0], "both")

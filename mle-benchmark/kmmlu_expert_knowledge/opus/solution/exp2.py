"""Test retrieval / LSA cross features."""
import numpy as np, pandas as pd
from scipy import sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.model_selection import KFold
from sklearn.ensemble import HistGradientBoostingClassifier
from feats import build_features, OPTS

RS = 42
tr = pd.read_csv("../train.csv")
n = len(tr)
y = tr.label.values - 1
yb = np.zeros(n * 4, int); yb[np.arange(n) * 4 + y] = 1
q = tr.question.astype(str).tolist()
opt = [str(tr[c].iloc[i]) for i in range(n) for c in OPTS]

kf = KFold(5, shuffle=True, random_state=RS)
folds = list(kf.split(np.arange(n)))
def expand(idx): return (idx[:, None] * 4 + np.arange(4)).ravel()
def acc(s): return (s.reshape(n, 4).argmax(1) == y).mean()

# global tfidf (fit on all text incl test is fine/unsupervised, but keep to train here)
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3, sublinear_tf=True)
vec.fit(q + opt)
Q = normalize(vec.transform(q))
P = normalize(vec.transform(opt))

svd = TruncatedSVD(200, random_state=RS)
svd.fit(sp.vstack([Q, P]))
Qs = normalize(svd.transform(Q)); Ps = normalize(svd.transform(P))
lsa = (Ps * np.repeat(Qs, 4, axis=0)).sum(1)
print("LSA qo sim as sole score acc:", acc(lsa))

# retrieval features, fold-safe
K = 25
RF = np.zeros((n * 4, 6))
for tri, vai in folds:
    Qt = Q[tri]; ytr = y[tri]
    S = (Q[vai] @ Qt.T).toarray()
    # correct answer text matrix / distractor
    corr_idx = tri * 4 + ytr
    Pc = P[corr_idx]
    for a, i in enumerate(vai):
        nb = np.argpartition(-S[a], K)[:K]
        w = S[a][nb]
        ordr = np.argsort(-w); nb = nb[ordr]; w = w[ordr]
        w = np.maximum(w, 0)
        Po = P[i * 4:i * 4 + 4]
        simc = (Po @ Pc[nb].T).toarray()          # 4 x K sim to neighbor correct answers
        dis = []
        for b in nb:
            gi = tri[b]
            idxs = [gi * 4 + jj for jj in range(4) if jj != ytr[b]]
            dis.append(idxs)
        dis = np.array(dis).ravel()
        simd = (Po @ P[dis].T).toarray().reshape(4, K, 3).mean(2)
        ww = w / (w.sum() + 1e-9)
        fc = simc @ ww; fd = simd @ ww
        RF[i * 4:i * 4 + 4, 0] = fc
        RF[i * 4:i * 4 + 4, 1] = fd
        RF[i * 4:i * 4 + 4, 2] = fc - fd
        RF[i * 4:i * 4 + 4, 3] = simc[:, :5].mean(1) - simd[:, :5].mean(1)
        RF[i * 4:i * 4 + 4, 4] = simc.max(1)
        RF[i * 4:i * 4 + 4, 5] = w[0]
print("retrieval fc-fd as sole score acc:", acc(RF[:, 2]), acc(RF[:, 3]))

Xd, _, _ = build_features(tr)
sets = {
    "dense": Xd,
    "dense+lsa": np.hstack([Xd, lsa[:, None]]),
    "dense+ret": np.hstack([Xd, RF]),
    "dense+lsa+ret": np.hstack([Xd, lsa[:, None], RF]),
}
for name, XX in sets.items():
    oof = np.zeros(n * 4)
    for tri, vai in folds:
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
            max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0, random_state=RS)
        m.fit(XX[expand(tri)], yb[expand(tri)])
        oof[expand(vai)] = m.predict_proba(XX[expand(vai)])[:, 1]
    print(name, XX.shape[1], round(acc(oof), 4), flush=True)
    np.save(f"/tmp/oof_{name}.npy", oof)
np.save("/tmp/RF.npy", RF); np.save("/tmp/lsa.npy", lsa)

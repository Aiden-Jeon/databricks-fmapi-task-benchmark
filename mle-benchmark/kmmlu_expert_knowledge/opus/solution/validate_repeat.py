"""Repeated 5-fold CV with the FIXED final configuration (no tuning inside)."""
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from feats import build_features
from pipeline import (TextSpace, lsa_sim, retrieval_feats, hgb_scores, text_lr_scores,
                      opt_texts, expand, zgroup)

W_CHAR, W_WORD = 0.25, 0.12
tr = pd.read_csv("../train.csv"); te = pd.read_csv("../test.csv")
n = len(tr); y = tr.label.values - 1
ybin = np.zeros(n * 4, int); ybin[np.arange(n) * 4 + y] = 1
q = tr.question.astype(str).tolist(); opt = opt_texts(tr)
space = TextSpace(q + opt + te.question.astype(str).tolist() + opt_texts(te))
Xd, _, _ = build_features(tr)
lsa = lsa_sim(space, q, opt)


def acc(s):
    return (np.asarray(s).reshape(n, 4).argmax(1) == y).mean()


res = {"hgb": [], "blend": []}
for seed in [7, 202, 1234]:
    oh = np.zeros(n * 4); oc = np.zeros(n * 4); ow = np.zeros(n * 4)
    for tri, vai in KFold(5, shuffle=True, random_state=seed).split(np.arange(n)):
        qtr = [q[i] for i in tri]; otr = [opt[i] for i in expand(tri)]
        RFtr = retrieval_feats(space, qtr, otr, y[tri], qtr, otr, exclude_self=True)
        RFva = retrieval_feats(space, qtr, otr, y[tri], [q[i] for i in vai],
                               [opt[i] for i in expand(vai)])
        A = np.hstack([Xd[expand(tri)], lsa[expand(tri), None], RFtr])
        B = np.hstack([Xd[expand(vai)], lsa[expand(vai), None], RFva])
        oh[expand(vai)] = hgb_scores(A, ybin[expand(tri)], B, seeds=(0,))
        c, w = text_lr_scores(otr, ybin[expand(tri)], [opt[i] for i in expand(vai)])
        oc[expand(vai)] = c; ow[expand(vai)] = w
    a1 = acc(oh)
    a2 = acc(zgroup(oh) + W_CHAR * zgroup(oc) + W_WORD * zgroup(ow))
    res["hgb"].append(a1); res["blend"].append(a2)
    print(f"seed {seed}: hgb={a1:.4f} blend={a2:.4f}", flush=True)
print("always-D baseline:", round((y == 3).mean(), 4))
for k, v in res.items():
    print(k, "mean", round(np.mean(v), 4), "+-", round(np.std(v), 4))

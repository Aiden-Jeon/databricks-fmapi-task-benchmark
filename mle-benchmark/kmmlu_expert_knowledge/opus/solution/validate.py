"""5-fold CV of the final pipeline (fixed blend weights, no per-fold tuning)."""
import sys
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from feats import build_features
from pipeline import (TextSpace, lsa_sim, retrieval_feats, hgb_scores, text_lr_scores,
                      opt_texts, expand, zgroup, RS)

tr = pd.read_csv("../train.csv")
te = pd.read_csv("../test.csv")
n = len(tr)
y = tr.label.values - 1
ybin = np.zeros(n * 4, int); ybin[np.arange(n) * 4 + y] = 1
q = tr.question.astype(str).tolist()
opt = opt_texts(tr)

space = TextSpace(q + opt + te.question.astype(str).tolist() + opt_texts(te))
Xd, _, _ = build_features(tr)
lsa = lsa_sim(space, q, opt)
print("dense feats:", Xd.shape, flush=True)

folds = list(KFold(5, shuffle=True, random_state=RS).split(np.arange(n)))


def acc(s):
    return (np.asarray(s).reshape(n, 4).argmax(1) == y).mean()


variants = {"dense": None, "dense+lsa": None, "dense+lsa+ret": None}
oofs = {k: np.zeros(n * 4) for k in variants}
oof_c = np.zeros(n * 4); oof_w = np.zeros(n * 4)

for f, (tri, vai) in enumerate(folds):
    qtr = [q[i] for i in tri]; otr = [opt[i] for i in expand(tri)]
    # LOO retrieval features for the training rows (pool == query set, self excluded)
    RFtr = retrieval_feats(space, qtr, otr, y[tri], qtr, otr, exclude_self=True)
    RFva = retrieval_feats(space, qtr, otr, y[tri], [q[i] for i in vai],
                           [opt[i] for i in expand(vai)])
    A0, B0 = Xd[expand(tri)], Xd[expand(vai)]
    A1 = np.hstack([A0, lsa[expand(tri), None]])
    B1 = np.hstack([B0, lsa[expand(vai), None]])
    A2 = np.hstack([A1, RFtr]); B2 = np.hstack([B1, RFva])
    for name, (A, B) in {"dense": (A0, B0), "dense+lsa": (A1, B1),
                         "dense+lsa+ret": (A2, B2)}.items():
        oofs[name][expand(vai)] = hgb_scores(A, ybin[expand(tri)], B)
    c, w = text_lr_scores(otr, ybin[expand(tri)], [opt[i] for i in expand(vai)])
    oof_c[expand(vai)] = c; oof_w[expand(vai)] = w
    print("fold", f, "done", flush=True)

print("prior only (always D):", round((y == 3).mean(), 4))
for k, v in oofs.items():
    print(f"hgb bag {k:16s}: {acc(v):.4f}")
print("lr char      :", round(acc(oof_c), 4))
print("lr word      :", round(acc(oof_w), 4))
for k, v in oofs.items():
    for wc in [0.0, 0.1, 0.2, 0.3, 0.4]:
        for ww in [0.0, 0.1, 0.2]:
            print(f"  {k:14s} wc={wc} ww={ww} -> "
                  f"{acc(zgroup(v) + wc * zgroup(oof_c) + ww * zgroup(oof_w)):.4f}")
np.save("/tmp/final_oof.npy",
        np.column_stack([oofs["dense"], oofs["dense+lsa"], oofs["dense+lsa+ret"],
                         oof_c, oof_w]))

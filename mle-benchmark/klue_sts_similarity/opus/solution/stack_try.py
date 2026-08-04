"""Experiment: append sparse-model OOF preds as features to the dense HGB.

Uses a *different* fold split than the sparse models' own OOF split, which keeps
the stacked-feature leakage limited; reported number is still slightly
optimistic, so it is only adopted if the gain is clear.
"""
import os, sys, time
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models as M

d1 = np.load("solution/_cache.npz", allow_pickle=True)
d2 = np.load("solution/_cache2.npz", allow_pickle=True)
s1 = np.load("solution/_sparse.npz")
s2 = np.load("solution/_sparse2.npz")
y = d1["y"]
Xtr = np.hstack([d1["Xtr"], d2["Xtr"], s1["oof"][:, None], s2["oof"][:, None]])
Xte = np.hstack([d1["Xte"], d2["Xte"], s1["test"][:, None], s2["test"][:, None]])
print("stacked", Xtr.shape, flush=True)

kf = list(KFold(5, shuffle=True, random_state=1234).split(Xtr))
for w, fac in [("hgb_st", lambda: M.hgb(lr=0.02, it=2000, leaves=31, l2=5.0, mf=0.4))]:
    t0 = time.time()
    oof = np.zeros(len(y)); test = np.zeros(len(Xte))
    for i_tr, i_va in kf:
        m = fac(); m.fit(Xtr[i_tr], y[i_tr])
        oof[i_va] = m.predict(Xtr[i_va]); test += m.predict(Xte) / 5
    print(f"{w} {pearsonr(oof,y)[0]:.5f} ({time.time()-t0:.0f}s)", flush=True)
    np.savez_compressed("solution/_stack.npz", oof=oof, test=test, y=y)

# reference: same fold split, no stacked features
Xtr0 = np.hstack([d1["Xtr"], d2["Xtr"]]); Xte0 = np.hstack([d1["Xte"], d2["Xte"]])
oof = np.zeros(len(y))
for i_tr, i_va in kf:
    m = M.hgb(lr=0.02, it=2000, leaves=31, l2=5.0, mf=0.4)
    m.fit(Xtr0[i_tr], y[i_tr]); oof[i_va] = m.predict(Xtr0[i_va])
print(f"hgb_ref (same folds, no stack) {pearsonr(oof,y)[0]:.5f}", flush=True)

"""Second stacked model (different config / fold split) for ensemble diversity."""
import os, sys, time
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models as M

d1 = np.load("solution/_cache.npz", allow_pickle=True)
d2 = np.load("solution/_cache2.npz", allow_pickle=True)
s1 = np.load("solution/_sparse.npz"); s2 = np.load("solution/_sparse2.npz")
y = d1["y"]
Xtr = np.hstack([d1["Xtr"], d2["Xtr"], s1["oof"][:, None], s2["oof"][:, None]])
Xte = np.hstack([d1["Xte"], d2["Xte"], s1["test"][:, None], s2["test"][:, None]])

kf = list(KFold(5, shuffle=True, random_state=2024).split(Xtr))
out = {}
for name, fac in [
    ("stackb", lambda: M.hgb(seed=11, lr=0.03, it=1200, leaves=63, l2=3.0, mf=0.3)),
    ("stackc", lambda: M.svr(C=8.0)),
]:
    t0 = time.time()
    oof = np.zeros(len(y)); test = np.zeros(len(Xte))
    for i_tr, i_va in kf:
        m = fac(); m.fit(Xtr[i_tr], y[i_tr])
        oof[i_va] = m.predict(Xtr[i_va]); test += m.predict(Xte) / 5
    print(f"{name} {pearsonr(oof,y)[0]:.5f} ({time.time()-t0:.0f}s)", flush=True)
    np.savez_compressed(f"solution/_stack_{name}.npz", oof=oof, test=test, y=y)

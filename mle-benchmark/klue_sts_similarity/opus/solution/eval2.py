import os, sys, time
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models as M

d1 = np.load("solution/_cache.npz", allow_pickle=True)
d2 = np.load("solution/_cache2.npz", allow_pickle=True)
Xtr = np.hstack([d1["Xtr"], d2["Xtr"]])
Xte = np.hstack([d1["Xte"], d2["Xte"]])
y = d1["y"]
print("combined", Xtr.shape, flush=True)
kf = list(KFold(5, shuffle=True, random_state=42).split(Xtr))

FACTORIES = {
    "ridge": M.ridge,
    "svr": M.svr,
    "hgb": lambda: M.hgb(lr=0.02, it=2000, leaves=31, l2=5.0, mf=0.4),
    "hgb2": lambda: M.hgb(seed=7, lr=0.03, it=1200, leaves=63, l2=3.0, mf=0.3),
    "et": M.et,
    "hgb_abs": M.hgb_abs,
    "mlp": lambda: M.mlp(hidden=(256, 64)),
    "mlp2": lambda: M.mlp(seed=3, hidden=(512, 128, 32), alpha=3e-3),
}

which = sys.argv[1:] or list(FACTORIES)
out = {}
for w in which:
    t0 = time.time()
    oof = np.zeros(len(y)); test = np.zeros(len(Xte))
    for i_tr, i_va in kf:
        m = FACTORIES[w]()
        m.fit(Xtr[i_tr], y[i_tr])
        oof[i_va] = m.predict(Xtr[i_va])
        test += m.predict(Xte) / 5
    print(f"{w:8s} {pearsonr(oof,y)[0]:.5f} ({time.time()-t0:.0f}s)", flush=True)
    out[f"oof_{w}"] = oof; out[f"test_{w}"] = test
np.savez_compressed(f"solution/_oof2_{'_'.join(which)}.npz", y=y, **out)

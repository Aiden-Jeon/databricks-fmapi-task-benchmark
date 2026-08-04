import os, sys, time
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cv import load
import models as M

Xtr, Xte, y = load()
print("data", Xtr.shape, Xte.shape)
NF = 5
kf = list(KFold(NF, shuffle=True, random_state=42).split(Xtr))


def run(name, factory, X=Xtr, Xt=Xte):
    t0 = time.time()
    oof = np.zeros(len(y)); test = np.zeros(len(Xt))
    for f, (i_tr, i_va) in enumerate(kf):
        m = factory()
        m.fit(X[i_tr], y[i_tr])
        oof[i_va] = m.predict(X[i_va])
        test += m.predict(Xt) / NF
    r = pearsonr(oof, y)[0]
    print(f"{name:12s} oof pearson = {r:.5f}   ({time.time()-t0:.0f}s)", flush=True)
    return oof, test, r


if __name__ == "__main__":
    which = sys.argv[1:] or ["ridge", "svr", "krr", "hgb", "et"]
    res = {}
    for w in which:
        res[w] = run(w, M.ZOO[w])
    np.savez_compressed("solution/_oof.npz",
                        **{f"oof_{k}": v[0] for k, v in res.items()},
                        **{f"test_{k}": v[1] for k, v in res.items()}, y=y)

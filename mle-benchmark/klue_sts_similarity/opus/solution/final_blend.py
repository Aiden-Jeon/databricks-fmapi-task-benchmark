"""Combine all base-model OOF/test predictions into the final submission."""
import glob, os
import numpy as np, pandas as pd
from scipy.optimize import nnls
from scipy.stats import pearsonr

y = None
oof, test, keys = {}, {}, []
for f in sorted(glob.glob("solution/_oof2_*.npz")):
    d = np.load(f)
    y = d["y"]
    for k in d.files:
        if k.startswith("oof_"):
            n = k[4:]
            oof[n] = d[k]; test[n] = d["test_" + n]
for f in sorted(glob.glob("solution/_sparse*.npz") + glob.glob("solution/_stack*.npz")):
    d = np.load(f)
    n = os.path.basename(f)[1:-4]
    oof[n] = d["oof"]; test[n] = d["test"]
keys = sorted(oof)

O = np.column_stack([oof[k] for k in keys])
T = np.column_stack([test[k] for k in keys])
for k in keys:
    print(f"  {k:8s} {pearsonr(oof[k], y)[0]:.5f}")

mu, sd = O.mean(0), O.std(0)
Oz, Tz = (O - mu) / sd, (T - mu) / sd
yz = (y - y.mean()) / y.std()

# NNLS weights, cross-validated so the reported number is not weight-fitted in-sample
from sklearn.model_selection import KFold
bl_oof = np.zeros(len(y))
for i_tr, i_va in KFold(5, shuffle=True, random_state=7).split(Oz):
    w, _ = nnls(Oz[i_tr], yz[i_tr])
    bl_oof[i_va] = Oz[i_va] @ w
print("blend (nested-CV weights) %.5f" % pearsonr(bl_oof, y)[0])

w, _ = nnls(Oz, yz)
w = w / max(w.sum(), 1e-9)
print("final weights:", dict(zip(keys, np.round(w, 3))))
print("blend (in-sample weights) %.5f" % pearsonr(Oz @ w, y)[0])

# map blended z-score back to the 0..5 score scale via linear fit on OOF
z_oof = Oz @ w
a, b = np.polyfit(z_oof, y, 1)
pred = np.clip(a * (Tz @ w) + b, 0, 5)

te = pd.read_csv("test.csv")
os.makedirs("outputs", exist_ok=True)
sub = pd.DataFrame({"id": te.id, "score": pred})
sub.to_csv("outputs/submission.csv", index=False)
print(sub.describe())
print("rows", len(sub), "unique ids", sub.id.nunique())

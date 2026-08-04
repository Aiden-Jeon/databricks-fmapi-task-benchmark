"""Find blend weights on OOF predictions (maximize pearson)."""
import sys
import numpy as np
from scipy.optimize import nnls
from scipy.stats import pearsonr, rankdata

f = np.load(sys.argv[1] if len(sys.argv) > 1 else "solution/_oof.npz")
y = f["y"]
keys = sorted(k[4:] for k in f.files if k.startswith("oof_"))
O = np.column_stack([f["oof_" + k] for k in keys])
T = np.column_stack([f["test_" + k] for k in keys])
for i, k in enumerate(keys):
    print(f"  {k:14s} {pearsonr(O[:,i],y)[0]:.5f}")

# z-score each model then NNLS
mu, sd = O.mean(0), O.std(0)
Oz, Tz = (O - mu) / sd, (T - mu) / sd
w, _ = nnls(Oz, (y - y.mean()) / y.std())
w = w / w.sum() if w.sum() > 0 else np.ones(len(keys)) / len(keys)
print("nnls weights:", dict(zip(keys, np.round(w, 3))))
print("blend        %.5f" % pearsonr(Oz @ w, y)[0])
print("simple mean  %.5f" % pearsonr(Oz.mean(1), y)[0])
R = np.column_stack([rankdata(O[:, i]) for i in range(O.shape[1])])
print("rank mean    %.5f" % pearsonr(R.mean(1), y)[0])
np.save("solution/_blend_w.npy", w)
np.savez("solution/_blend_meta.npz", keys=np.array(keys), mu=mu, sd=sd, w=w)

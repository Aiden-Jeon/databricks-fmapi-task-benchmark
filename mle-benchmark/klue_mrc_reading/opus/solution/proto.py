import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.ensemble import HistGradientBoostingRegressor
from common import char_f1
from features import IdfTable
from pipeline import make_ranker_data, score_df

NFIT = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
NVAL = int(sys.argv[2]) if len(sys.argv) > 2 else 500

t0 = time.time()
tr = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "train.csv"),
                 keep_default_na=False)
ctxs = tr.context.drop_duplicates().tolist()
rng = np.random.default_rng(0)
rng.shuffle(ctxs)
nval_ctx = int(0.2 * len(ctxs))
val_ctx = set(ctxs[:nval_ctx])
va = tr[tr.context.isin(val_ctx)]
fi = tr[~tr.context.isin(val_ctx)]
idf = IdfTable(fi.context.drop_duplicates().tolist())

fi_a = fi[fi.answer != ""].sample(min(NFIT, (fi.answer != "").sum()), random_state=0)
print("building ranker data on", len(fi_a), flush=True)
X, y = make_ranker_data(fi_a, idf)
print("X", X.shape, "pos frac", (y > 0.5).mean(), "t", round(time.time() - t0, 1), flush=True)

m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, max_leaf_nodes=63,
                                  min_samples_leaf=40, l2_regularization=1.0,
                                  early_stopping=False, random_state=RS if False else 0)
m.fit(X, y)
print("ranker fit t", round(time.time() - t0, 1), flush=True)

vs = va.sample(min(NVAL, len(va)), random_state=0)
preds, stats = score_df(vs, idf, m)
f1 = np.array([char_f1(p, g) for p, g in zip(preds, vs.answer)])
isans = (vs.answer != "").values
print("t", round(time.time() - t0, 1))
print("answerable-only top1 charF1:", round(f1[isans].mean(), 4), "n", isans.sum())
print("exact:", round(np.mean([p == g for p, g in zip(np.array(preds)[isans], vs.answer[isans])]), 4))
print("all-empty score:", round((~isans).mean(), 4))
print("answer-everything score:", round(f1.mean(), 4))
np.save("/tmp/proto_stats.npy", stats)
np.save("/tmp/proto_f1.npy", f1)
np.save("/tmp/proto_isans.npy", isans)
for p, g in list(zip(preds, vs.answer))[:20]:
    print(repr(p), "|", repr(g))

import os, sys, time
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cv import load

Xtr, Xte, y = load()
kf = list(KFold(5, shuffle=True, random_state=42).split(Xtr))

CONFIGS = {
    "base":    dict(learning_rate=0.05, max_iter=600, max_leaf_nodes=31, max_features=0.6, l2_regularization=1.0),
    "lr03_1k": dict(learning_rate=0.03, max_iter=1200, max_leaf_nodes=31, max_features=0.5, l2_regularization=1.0),
    "leaf15":  dict(learning_rate=0.04, max_iter=1000, max_leaf_nodes=15, max_features=0.5, l2_regularization=1.0),
    "leaf63":  dict(learning_rate=0.03, max_iter=800, max_leaf_nodes=63, max_features=0.4, l2_regularization=3.0),
    "mf03":    dict(learning_rate=0.03, max_iter=1200, max_leaf_nodes=31, max_features=0.3, l2_regularization=2.0),
    "deep_reg": dict(learning_rate=0.02, max_iter=2000, max_leaf_nodes=31, max_features=0.4, l2_regularization=5.0, min_samples_leaf=30),
}

for name, kw in CONFIGS.items():
    t0 = time.time()
    oof = np.zeros(len(y))
    for i_tr, i_va in kf:
        m = HistGradientBoostingRegressor(early_stopping=False, random_state=0,
                                          min_samples_leaf=kw.pop("min_samples_leaf", 20) if False else kw.get("min_samples_leaf", 20),
                                          **{k: v for k, v in kw.items() if k != "min_samples_leaf"})
        m.fit(Xtr[i_tr], y[i_tr])
        oof[i_va] = m.predict(Xtr[i_va])
    print(f"{name:10s} {pearsonr(oof,y)[0]:.5f}  ({time.time()-t0:.0f}s)", flush=True)

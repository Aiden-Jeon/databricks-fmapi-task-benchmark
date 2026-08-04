"""Sanity check: is the gain from graph features real, or an artefact of the
validation scheme?

Strict variant: for fold k the sentence graph is built ONLY from the training
rows of that fold (train rows additionally mask their own edge).  This is a
pessimistic bound (graph is 20% smaller than at test time); the run2 scheme
(full graph, own edge masked) is the faithful simulation of test conditions.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import features2 as F2  # noqa: E402

CACHE = os.path.join(HERE, "cache")
SEED, NFOLD = 42, 5

tr = pd.read_csv(os.path.join(ROOT, "train.csv"))
for c in ("sentence1", "sentence2"):
    tr[c] = tr[c].fillna("")
y = tr["label"].values.astype(int)
s1 = tr["sentence1"].tolist()
s2 = tr["sentence2"].tolist()

D = np.load(os.path.join(CACHE, "dense_tr_a.npy"))
L = np.load(os.path.join(CACHE, "align_tr_a.npy"))
base = np.hstack([D, L]).astype(np.float32)

folds = list(StratifiedKFold(NFOLD, shuffle=True, random_state=SEED).split(y, y))


def masked_rows(adj, idx):
    rows = []
    for i in idx:
        a, b = s1[i], s2[i]
        if a in adj and b in adj[a]:
            lab = adj[a].pop(b)
            adj[b].pop(a, None)
            rows.append(F2.graph_row(adj, a, b))
            adj[a][b] = lab
            adj[b][a] = lab
        else:
            rows.append(F2.graph_row(adj, a, b))
    return np.asarray(rows, dtype=np.float32)


def evaluate(tag, use_graph, strict):
    oof = np.zeros(len(y))
    for k, (itr, iva) in enumerate(folds):
        if not use_graph:
            Xtr, Xva = base[itr], base[iva]
        else:
            if strict:
                adj = F2.build_graph((s1[i], s2[i], y[i]) for i in itr)
                Gtr = masked_rows(adj, itr)
                Gva = np.asarray([F2.graph_row(adj, s1[i], s2[i]) for i in iva],
                                 dtype=np.float32)
            else:
                adj = F2.build_graph(zip(s1, s2, y))
                Gtr = masked_rows(adj, itr)
                Gva = masked_rows(adj, iva)
            Xtr = np.hstack([base[itr], Gtr])
            Xva = np.hstack([base[iva], Gva])
        m = HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.06, max_leaf_nodes=31,
            min_samples_leaf=30, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.12, n_iter_no_change=40,
            random_state=SEED + k)
        m.fit(Xtr, y[itr])
        oof[iva] = m.predict_proba(Xva)[:, 1]
    acc = ((oof > 0.5).astype(int) == y).mean()
    print(f"{tag:34s} oof acc = {acc:.4f}", flush=True)
    return oof


if __name__ == "__main__":
    evaluate("no graph", False, False)
    evaluate("graph (strict, fold-only)", True, True)
    evaluate("graph (full, own-edge masked)", True, False)

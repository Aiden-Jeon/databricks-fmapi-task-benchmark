"""Outer cross-validation of the full pipeline (question-level splits)."""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import fit_predict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(n_splits=5, seed=0, **kw):
    tr = pd.read_csv(os.path.join(ROOT, "train.csv"))
    lab = tr["label"].values
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs, parts_acc = [], {}
    oof = np.zeros((len(tr), 4))
    for a, b in kf.split(tr):
        S, parts = fit_predict(tr.iloc[a].reset_index(drop=True),
                               tr.iloc[b].reset_index(drop=True),
                               return_parts=True, **kw)
        oof[b] = S
        accs.append((S.argmax(1) + 1 == lab[b]).mean())
        for k, v in parts.items():
            p = (v.reshape(-1, 4).argmax(1) + 1 == lab[b]).mean()
            parts_acc.setdefault(k, []).append(p)
    full = (oof.argmax(1) + 1 == lab).mean()
    print(f"seed={seed} folds={np.round(accs,4)}  OOF acc={full:.4f}")
    for k, v in parts_acc.items():
        print(f"   part {k}: {np.mean(v):.4f}")
    return full, oof


if __name__ == "__main__":
    kw = {}
    if "--nostack" in sys.argv:
        kw["use_stack"] = False
    tot = []
    for sd in [0, 1]:
        f, oof = run(seed=sd, **kw)
        tot.append(f)
        np.save(os.path.join(ROOT, "solution", f"_oof_full_seed{sd}.npy"), oof)
    print("mean OOF acc:", round(float(np.mean(tot)), 4))

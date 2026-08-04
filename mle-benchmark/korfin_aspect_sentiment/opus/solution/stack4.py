"""Final stack: base OOF probs + aspect/sibling target encoding + numeric
+ LEAK-FREE sibling soft-prediction features (see sibpred.py)."""
import os
import sys
import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CLASSES, TASK  # noqa
import stack2 as S2  # noqa
from stack3 import sibling_soft  # noqa

CACHE = os.path.join(TASK, "solution", "cache")


def build():
    tr, te, y, folds, A, B, names, prior, onehot = S2.build_matrices()
    p_tr = np.load(f"{CACHE}/SIBP_tr.npy")
    p_te = np.load(f"{CACHE}/SIBP_te.npy")
    Str, Ste = sibling_soft(tr, te, p_tr, p_te, prior)
    return tr, te, y, folds, np.c_[A, Str], np.c_[B, Ste], A, B


def main():
    tr, te, y, folds, A2, B2, A, B = build()
    print("matrix", A2.shape)
    for tag, (X, Z) in [("S4", (A2, B2))]:
        oof, tep = S2.run(X, Z, y, folds, "lgb", seeds=(0, 1, 2))
        print(f"{tag} lgb: {f1_score(y, oof.argmax(1), average='macro'):.4f}")
        np.save(f"{CACHE}/{tag}lgb_oof.npy", oof)
        np.save(f"{CACHE}/{tag}lgb_test.npy", tep)
        oof2, tep2 = S2.run(X, Z, y, folds, "lr")
        print(f"{tag} lr : {f1_score(y, oof2.argmax(1), average='macro'):.4f}")
        for w in [0.15, 0.25, 0.35]:
            o = w * oof2 + (1 - w) * oof
            print(f"   mix {w}: {f1_score(y, o.argmax(1), average='macro'):.4f}")
        np.save(f"{CACHE}/{tag}lr_oof.npy", oof2)
        np.save(f"{CACHE}/{tag}lr_test.npy", tep2)
    np.save(f"{CACHE}/S4A.npy", A2)
    np.save(f"{CACHE}/S4B.npy", B2)


if __name__ == "__main__":
    main()

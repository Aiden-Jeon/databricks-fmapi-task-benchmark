"""Stack v3 = stack2 features + label-free sibling soft-prediction features.

For every row we look at the *other* aspects of the same sentence (siblings) anywhere
in train+test and aggregate the base model's predicted probabilities for them.
Train rows use out-of-fold predictions, test rows use the averaged test predictions,
so no label information leaks.
"""
import os
import sys
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CLASSES, TASK  # noqa
import stack2 as S2  # noqa

CACHE = os.path.join(TASK, "solution", "cache")


def sibling_soft(tr, te, p_tr, p_te, prior):
    """Aggregate base-model probabilities of siblings (train+test combined)."""
    sent = np.concatenate([tr.sentence.values, te.sentence.values])
    asp = np.concatenate([tr.aspect.fillna("").values, te.aspect.fillna("").values])
    P = np.vstack([p_tr, p_te])
    idx = defaultdict(list)
    for i, s in enumerate(sent):
        idx[s].append(i)
    n = len(sent)
    F = np.zeros((n, 3 + 1 + 3 + 1), np.float32)
    for i in range(n):
        s, a = sent[i], asp[i]
        sibs = [j for j in idx[s] if j != i and asp[j] != a]
        if not sibs:
            F[i, 0:3] = prior
            F[i, 3] = 0
            F[i, 4:7] = prior
            F[i, 7] = -1
            continue
        F[i, 0:3] = P[sibs].mean(0)
        F[i, 3] = len(sibs)
        p_self = s.index(a) if a and a in s else -1
        best_d, best_j = None, None
        for j in sibs:
            if p_self >= 0 and asp[j] and asp[j] in s:
                d = abs(s.index(asp[j]) - p_self)
                if best_d is None or d < best_d:
                    best_d, best_j = d, j
        if best_j is not None:
            F[i, 4:7] = P[best_j]
            F[i, 7] = best_d / max(len(s), 1)
        else:
            F[i, 4:7] = P[sibs].mean(0)
            F[i, 7] = -1
    return F[:len(tr)], F[len(tr):]


def main():
    tr, te, y, folds, A, B, names, prior, onehot = S2.build_matrices()
    base = "BLEND" if os.path.exists(f"{CACHE}/BLEND_oof.npy") else names[0]
    p_tr = np.load(f"{CACHE}/{base}_oof.npy")
    p_te = np.load(f"{CACHE}/{base}_test.npy")
    Str, Ste = sibling_soft(tr, te, p_tr, p_te, prior)
    A2 = np.c_[A, Str]
    B2 = np.c_[B, Ste]
    print("stack3 matrix", A2.shape)
    res = {}
    for tag, (X, Z) in [("S2", (A, B)), ("S3", (A2, B2))]:
        oof, tep = S2.run(X, Z, y, folds, "lgb", seeds=(0, 1, 2))
        s = f1_score(y, oof.argmax(1), average="macro")
        print(f"{tag} lgb: {s:.4f}")
        res[tag] = (s, oof, tep)
        np.save(f"{CACHE}/{tag}lgbF_oof.npy", oof)
        np.save(f"{CACHE}/{tag}lgbF_test.npy", tep)
    np.save(f"{CACHE}/S3A.npy", A2)
    np.save(f"{CACHE}/S3B.npy", B2)


if __name__ == "__main__":
    main()

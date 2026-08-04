"""Blend OOF decision matrices: weight search + per-class bias, maximizing macro F1.

Includes an honest half-split check to verify the tuning does not just overfit OOF.
"""
import sys, glob, os, itertools
import numpy as np, pandas as pd
from sklearn.metrics import f1_score

CACHE = "/tmp/opencode"
TASK = "/tmp/kmle/M1_t3_ynat_full_20260804_033458/task"


def load(keys):
    O, T = [], []
    for k in keys:
        z = np.load(f"{CACHE}/oof_{k}.npz", allow_pickle=True)
        o, t, cls = z["oof"], z["test"], z["classes"]
        # standardize each model's score matrix to comparable scale
        m, s = o.mean(), o.std()
        O.append((o - m) / s)
        T.append((t - m) / s)
    return np.stack(O), np.stack(T), cls  # (M,n,7)


def mf1(S, y, cls, bias=None):
    Z = S if bias is None else S + bias
    return f1_score(y, cls[Z.argmax(1)], average="macro")


def fit_bias(S, y, cls, rounds=4, grid=None):
    """Coordinate ascent on per-class additive bias to maximize macro F1."""
    if grid is None:
        grid = np.arange(-0.60, 0.601, 0.02)
    b = np.zeros(S.shape[1])
    best = mf1(S, y, cls, b)
    for _ in range(rounds):
        improved = False
        for j in range(S.shape[1]):
            cur = b[j]
            cand, cbest = cur, best
            for g in grid:
                b[j] = g
                v = mf1(S, y, cls, b)
                if v > cbest + 1e-6:
                    cbest, cand = v, g
            b[j] = cand
            if cbest > best + 1e-6:
                best, improved = cbest, True
        if not improved:
            break
    return b, best


def fit_weights(O, y, cls, rounds=3):
    """Coordinate ascent on non-negative model weights."""
    M = O.shape[0]
    w = np.zeros(M); w[0] = 1.0
    best = mf1(np.tensordot(w, O, axes=1), y, cls)
    for _ in range(rounds):
        improved = False
        for i in range(M):
            cur, cand, cbest = w[i], w[i], best
            for g in [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.4]:
                if i == 0 and g == 0:
                    continue
                w[i] = g
                v = mf1(np.tensordot(w, O, axes=1), y, cls)
                if v > cbest + 1e-6:
                    cbest, cand = v, g
            w[i] = cand
            if cbest > best + 1e-6:
                best, improved = cbest, True
        if not improved:
            break
    return w, best


if __name__ == "__main__":
    keys = sys.argv[1].split(",")
    tr = pd.read_csv(f"{TASK}/train.csv")
    y = tr.label.values
    O, T, cls = load(keys)
    for k, o in zip(keys, O):
        print(f"  {k}: {mf1(o, y, cls):.5f}")
    print("simple mean :", f"{mf1(O.mean(0), y, cls):.5f}")

    # ---- honest check: tune on half A, evaluate on half B (and vice versa) ----
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(y)); h1, h2 = idx[::2], idx[1::2]
    gains = []
    for a, b in [(h1, h2), (h2, h1)]:
        w, _ = fit_weights(O[:, a], y[a], cls)
        Sa = np.tensordot(w, O[:, a], axes=1)
        bi, _ = fit_bias(Sa, y[a], cls)
        Sb = np.tensordot(w, O[:, b], axes=1)
        base = mf1(O[0][b], y[b], cls)
        tuned = mf1(Sb, y[b], cls, bi)
        gains.append(tuned - base)
        print(f"  holdout: base(A only)={base:.5f} -> blended+bias={tuned:.5f} (w={np.round(w,2)})")
    print(f"mean honest gain over single best: {np.mean(gains):+.5f}")

    # ---- final fit on all OOF ----
    w, sw = fit_weights(O, y, cls)
    S = np.tensordot(w, O, axes=1)
    bi, sb = fit_bias(S, y, cls)
    print(f"FINAL weights={dict(zip(keys, np.round(w,2)))} oof_w={sw:.5f} oof_w+bias={sb:.5f}")
    print("bias=", dict(zip(cls, np.round(bi, 3))))
    np.savez(f"{CACHE}/blend_cfg.npz", keys=np.array(keys), w=w, bias=bi, classes=cls)

    Ts = np.tensordot(w, T, axes=1) + bi
    te = pd.read_csv(f"{TASK}/test.csv")
    pred = cls[Ts.argmax(1)]
    pd.DataFrame({"id": te.id, "label": pred}).to_csv(f"{CACHE}/blend_submission.csv", index=False)
    print(pd.Series(pred).value_counts().to_dict())

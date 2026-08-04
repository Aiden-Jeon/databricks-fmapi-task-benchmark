"""Level-2 blending / stacking over cached base-model predictions."""
import os
import sys
import itertools
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression

import common as C
import zoo

SEED = 42
NFOLD = 5


def load_cached(names):
    oofs, tsts, kept = [], [], []
    for n in names:
        p = os.path.join(zoo.CACHE, n + ".npz")
        if not os.path.exists(p):
            print("  (missing)", n)
            continue
        d = np.load(p)
        oofs.append(d["oof"])
        tsts.append(d["tst"])
        kept.append(n)
    return kept, np.stack(oofs), np.stack(tsts)


def rank_norm(P):
    """Column-wise rank normalisation to [0,1] (scale-free averaging)."""
    R = np.empty_like(P, dtype=float)
    n = P.shape[0]
    for j in range(P.shape[1]):
        order = P[:, j].argsort()
        r = np.empty(n)
        r[order] = np.arange(n)
        R[:, j] = r / max(n - 1, 1)
    return R


def holdout_eval(Y, P, seed=0, n_rep=6):
    """Tune thresholds on half the OOF rows, score on the other half.
    Gives an honest estimate that is not inflated by threshold overfitting."""
    rng = np.random.RandomState(seed)
    scores = []
    for _ in range(n_rep):
        idx = rng.permutation(len(Y))
        a, b = idx[: len(idx) // 2], idx[len(idx) // 2:]
        for tr_i, te_i in ((a, b), (b, a)):
            thr, _ = C.tune_thresholds(Y[tr_i], P[tr_i], rounds=2)
            scores.append(C.macro_f1(Y[te_i], C.apply_thresholds(P[te_i], thr)))
    return float(np.mean(scores)), float(np.std(scores))


def stack(Y, oofs, tsts, Cval=1.0, use_all_labels=True):
    """Per-label level-2 logistic regression on base-model probabilities."""
    M = oofs.shape[0]
    kf = KFold(n_splits=NFOLD, shuffle=True, random_state=SEED)
    folds = list(kf.split(np.arange(len(Y))))
    n_te = tsts.shape[1]
    oof2 = np.zeros((len(Y), C.N_LABELS))
    tst2 = np.zeros((n_te, C.N_LABELS))

    def feat(P):  # P: (M, n, 10) -> (n, F)
        return np.concatenate([P[m] for m in range(M)], axis=1)

    Ftr_all = feat(oofs)
    Fte_all = feat(tsts)
    for j in range(C.N_LABELS):
        if use_all_labels:
            cols = slice(None)
            Ftr, Fte = Ftr_all, Fte_all
        else:
            cols = [m * C.N_LABELS + j for m in range(M)]
            Ftr, Fte = Ftr_all[:, cols], Fte_all[:, cols]
        for trn, val in folds:
            m = LogisticRegression(C=Cval, max_iter=2000)
            m.fit(Ftr[trn], Y[trn, j])
            oof2[val, j] = m.predict_proba(Ftr[val])[:, 1]
        m = LogisticRegression(C=Cval, max_iter=2000)
        m.fit(Ftr, Y[:, j])
        tst2[:, j] = m.predict_proba(Fte)[:, 1]
    return oof2, tst2


def greedy_blend(Y, oofs, names, n_iter=25):
    """Greedy forward selection with replacement on rank-normalised scores."""
    R = np.stack([rank_norm(o) for o in oofs])
    chosen = []
    cur = np.zeros_like(R[0])
    best_hist = []
    for it in range(n_iter):
        best, best_m = -1, None
        for m in range(len(names)):
            cand = (cur * len(chosen) + R[m]) / (len(chosen) + 1)
            _, s = C.tune_thresholds(Y, cand, rounds=2)
            if s > best:
                best, best_m = s, m
        chosen.append(best_m)
        cur = (cur * (len(chosen) - 1) + R[best_m]) / len(chosen)
        best_hist.append(best)
        print(f"  iter {it+1}: +{names[best_m]:14s} -> {best:.4f}", flush=True)
        if it >= 4 and best <= max(best_hist[:-1]) + 1e-5:
            break
    return chosen, best_hist


if __name__ == "__main__":
    tr, te = C.load(os.environ.get("DATA_DIR", "."))
    Y = C.labels_to_matrix(tr.labels)
    names = sys.argv[1:] or sorted(
        n[:-4] for n in os.listdir(zoo.CACHE) if n.endswith(".npz"))
    names, oofs, tsts = load_cached(names)
    print("models:", names)

    for n, o in zip(names, oofs):
        thr, s = C.tune_thresholds(Y, o, rounds=2)
        print(f"  {n:14s} tuned={s:.4f}")

    print("\n-- simple rank average --")
    Ravg = np.mean([rank_norm(o) for o in oofs], axis=0)
    thr, s = C.tune_thresholds(Y, Ravg, rounds=3)
    print("rank-avg tuned:", round(s, 4), " holdout:", holdout_eval(Y, Ravg))

    print("\n-- prob average --")
    Pavg = oofs.mean(0)
    thr, s = C.tune_thresholds(Y, Pavg, rounds=3)
    print("prob-avg tuned:", round(s, 4), " holdout:", holdout_eval(Y, Pavg))

    print("\n-- stacking (per-label, all-label features) --")
    for cv in (0.3, 1.0, 3.0):
        o2, t2 = stack(Y, oofs, tsts, Cval=cv, use_all_labels=True)
        thr, s = C.tune_thresholds(Y, o2, rounds=3)
        print(f"  C={cv}: tuned={s:.4f}  holdout={holdout_eval(Y, o2)}")

    print("\n-- stacking (per-label, own-label features) --")
    for cv in (1.0, 3.0):
        o2, t2 = stack(Y, oofs, tsts, Cval=cv, use_all_labels=False)
        thr, s = C.tune_thresholds(Y, o2, rounds=3)
        print(f"  C={cv}: tuned={s:.4f}  holdout={holdout_eval(Y, o2)}")

    print("\n-- greedy blend --")
    greedy_blend(Y, oofs, names)

"""OOF evaluation + blend search for the model zoo."""
import sys, time, itertools, json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from common import Featurizer, make_views, model_zoo, decision

N_SPLITS = 5
N_REPEATS = 4
SEED = 42


def oof_scores(name, blocks, est_factory, views, y):
    """Returns (n_repeats, n_samples) matrix of OOF signed scores."""
    out = np.zeros((N_REPEATS, len(y)))
    for r in range(N_REPEATS):
        skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED + r)
        for tr_i, va_i in skf.split(np.zeros(len(y)), y):
            vtr = {k: v[tr_i] for k, v in views.items()}
            vva = {k: v[va_i] for k, v in views.items()}
            f = Featurizer(blocks)
            Xtr = f.fit_transform(vtr)
            Xva = f.transform(vva)
            est = est_factory()
            est.fit(Xtr, y[tr_i])
            out[r, va_i] = decision(est, Xva)
    return out


def acc_of(score, y):
    return ((score > 0).astype(int) == y).mean()


def main():
    tr = pd.read_csv("../train.csv")
    y = tr.label.values
    views = make_views(tr.sentence.values)

    zoo = model_zoo()
    oof = {}
    for name, (blocks, fac) in zoo.items():
        t = time.time()
        s = oof_scores(name, blocks, fac, views, y)
        oof[name] = s
        accs = [acc_of(s[r], y) for r in range(N_REPEATS)]
        print(f"{name:10s} acc={np.mean(accs):.4f} +-{np.std(accs):.4f}  ({time.time()-t:.0f}s)", flush=True)
    np.savez("oof.npz", y=y, **{k: v for k, v in oof.items()})

    # rank-normalise each model's scores per repeat so weights are comparable
    def z(s):
        return (s - s.mean(axis=1, keepdims=True)) / (s.std(axis=1, keepdims=True) + 1e-12)

    Z = {k: z(v) for k, v in oof.items()}
    names = list(Z)
    print("\n--- greedy forward blend (with replacement) ---")
    cur = np.zeros_like(Z[names[0]])
    chosen = []
    best_hist = []
    for step in range(12):
        best, bestn = -1, None
        for n in names:
            cand = cur + Z[n]
            a = np.mean([acc_of(cand[r], y) for r in range(N_REPEATS)])
            if a > best:
                best, bestn = a, n
        cur = cur + Z[bestn]
        chosen.append(bestn)
        best_hist.append(best)
        print(f"step {step+1}: +{bestn:10s} -> {best:.4f}")
    k = int(np.argmax(best_hist))
    print("best blend:", chosen[:k + 1], f"acc={best_hist[k]:.4f}")
    json.dump(chosen[:k + 1], open("blend.json", "w"))


if __name__ == "__main__":
    main()

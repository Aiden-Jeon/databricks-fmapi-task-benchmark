"""Final model: stack4 features (+ leak-free self prediction), LGBM tuning,
multi-seed averaging and macro-F1 class-multiplier calibration."""
import os
import sys
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CLASSES, TASK  # noqa
import stack2 as S2  # noqa
import stack4 as S4  # noqa

CACHE = os.path.join(TASK, "solution", "cache")


def tune_mult(oof, y, grid=None):
    """Search per-class multipliers maximising macro-F1 (coordinate ascent)."""
    m = np.ones(3)
    best = f1_score(y, (oof * m).argmax(1), average="macro")
    for _ in range(4):
        for c in range(3):
            for v in np.linspace(0.6, 1.6, 41):
                mm = m.copy()
                mm[c] = v
                s = f1_score(y, (oof * mm).argmax(1), average="macro")
                if s > best + 1e-6:
                    best, m = s, mm
    return m, best


def main():
    tr, te, y, folds, A2, B2, A, B = S4.build()
    p_tr = np.load(f"{CACHE}/SIBP_tr.npy")
    p_te = np.load(f"{CACHE}/SIBP_te.npy")
    A3 = np.c_[A2, p_tr]
    B3 = np.c_[B2, p_te]
    print("feature matrix", A3.shape)

    cand = [
        dict(num_leaves=15, learning_rate=0.03, min_child_samples=30, colsample_bytree=0.7),
        dict(num_leaves=31, learning_rate=0.03, min_child_samples=20, colsample_bytree=0.5),
        dict(num_leaves=7, learning_rate=0.04, min_child_samples=40, colsample_bytree=0.8),
        dict(num_leaves=15, learning_rate=0.02, min_child_samples=15, colsample_bytree=0.4,
             reg_lambda=3.0),
    ]
    results = []
    for i, p in enumerate(cand):
        oof, tep = S2.run(A3, B3, y, folds, "lgb", seeds=(0,), lgb_params=p)
        s = f1_score(y, oof.argmax(1), average="macro")
        print(f"cfg{i} {p} -> {s:.4f}")
        results.append((s, i, p, oof, tep))
    results.sort(reverse=True, key=lambda r: r[0])
    best_cfgs = [r[2] for r in results[:2]]

    # multi-seed averaging over the two best configs
    oof_t = np.zeros((len(tr), 3))
    tep_t = np.zeros((len(te), 3))
    for p in best_cfgs:
        o, t = S2.run(A3, B3, y, folds, "lgb", seeds=(0, 1, 2, 3, 4), lgb_params=p)
        oof_t += o / len(best_cfgs)
        tep_t += t / len(best_cfgs)
    s_lgb = f1_score(y, oof_t.argmax(1), average="macro")
    print("multi-seed lgb:", round(s_lgb, 4))

    o_lr, t_lr = S2.run(A3, B3, y, folds, "lr")
    print("lr:", round(f1_score(y, o_lr.argmax(1), average="macro"), 4))
    best = (s_lgb, oof_t, tep_t, "lgb")
    for w in [0.1, 0.15, 0.2, 0.3]:
        o = (1 - w) * oof_t + w * o_lr
        s = f1_score(y, o.argmax(1), average="macro")
        print(f"  mix lr {w}: {s:.4f}")
        if s > best[0]:
            best = (s, o, (1 - w) * tep_t + w * t_lr, f"mix{w}")
    print("selected:", best[3], round(best[0], 4))

    m, s_cal = tune_mult(best[1], y)
    print("class multipliers", np.round(m, 3), "oof f1", round(s_cal, 4))
    # keep calibration only if it is a clear improvement (guard vs overfitting)
    use_m = m if s_cal > best[0] + 0.004 else np.ones(3)
    print("using multipliers:", np.round(use_m, 3))
    np.save(f"{CACHE}/FINAL_oof.npy", best[1] * use_m)
    np.save(f"{CACHE}/FINAL_test.npy", best[2] * use_m)
    print("final oof f1:", round(f1_score(y, (best[1] * use_m).argmax(1),
                                          average="macro"), 4))


if __name__ == "__main__":
    main()

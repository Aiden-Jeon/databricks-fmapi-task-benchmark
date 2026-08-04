"""Blend model probabilities, apply the premise-sibling prior, write submission.

Weights and the prior strength are tuned on the held-out 10% split
(the same split used by train_nn.py / train_linear.py: fold 0 of
StratifiedKFold(10, shuffle=True, random_state=7)).
"""
import os, sys, glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import LABELS, L2I, load
from sibling import estimate_prior, apply_prior, sibling_labels


def norm_p(p):
    p = np.clip(p, 1e-9, None)
    return p / p.sum(1, keepdims=True)


def main():
    tr, te = load()
    y = tr.label.map(L2I).values
    g = tr.sentence1.values
    gt = te.sentence1.values

    from sklearn.model_selection import StratifiedKFold
    tri, vai = next(iter(StratifiedKFold(10, shuffle=True,
                                         random_state=7).split(y, y)))

    val, test, names = [], [], []
    # neural nets: oof array has predictions only on the validation rows
    for f in sorted(glob.glob("work/nn_oof_*.npy")):
        tag = os.path.basename(f)[7:-4]
        tf = f"work/nn_test_{tag}.npy"
        if not os.path.exists(tf):
            continue
        o = np.load(f)
        if o[vai].sum() <= 0:
            continue
        val.append(norm_p(o[vai])); test.append(norm_p(np.load(tf)))
        names.append("nn_" + tag)
    for f in sorted(glob.glob("work/lin_val_*.npy")):
        tag = os.path.basename(f)[8:-4]
        tf = f"work/lin_test_{tag}.npy"
        if not os.path.exists(tf):
            continue
        val.append(norm_p(np.load(f))); test.append(norm_p(np.load(tf)))
        names.append("lin_" + tag)
    assert val, "no model predictions found"
    yv = y[vai]
    for n, p in zip(names, val):
        print(f"{n:10s} val acc {(p.argmax(1)==yv).mean():.4f}")

    # ---- weight search in log space (coarse grid / coordinate ascent) ----
    LV = np.stack([np.log(p) for p in val])
    LT = np.stack([np.log(p) for p in test])
    w = np.ones(len(val)) / len(val)
    best = (np.tensordot(w, LV, 1).argmax(1) == yv).mean()
    for _ in range(30):
        improved = False
        for i in range(len(w)):
            for d in (0.4, 0.2, -0.2, -0.4):
                w2 = w.copy(); w2[i] = max(w2[i] + d, 0.0)
                if w2.sum() == 0:
                    continue
                a = (np.tensordot(w2 / w2.sum(), LV, 1).argmax(1) == yv).mean()
                if a > best + 1e-6:
                    best, w, improved = a, w2 / w2.sum(), True
        if not improved:
            break
    print("weights", dict(zip(names, np.round(w, 3))), "blend val", round(best, 4))
    PV = norm_p(np.exp(np.tensordot(w, LV, 1)))
    PT = norm_p(np.exp(np.tensordot(w, LT, 1)))

    # ---- sibling prior ----
    prior = estimate_prior(y[tri], g[tri])
    sv = sibling_labels(g[vai], g[tri], y[tri])
    bw, ba = 0.0, best
    for ww in np.arange(0.0, 1.81, 0.1):
        a = (apply_prior(PV, sv, prior, ww).argmax(1) == yv).mean()
        if a > ba + 1e-9:
            ba, bw = a, ww
        print(f"  prior w={ww:.1f} val {a:.4f}")
    print("chosen prior weight", bw, "val acc", round(ba, 4))

    prior_full = estimate_prior(y, g)
    st = sibling_labels(gt, g, y)
    PT = apply_prior(PT, st, prior_full, bw)

    pred = [LABELS[i] for i in PT.argmax(1)]
    os.makedirs("outputs", exist_ok=True)
    sub = pd.DataFrame({"id": te.id.values, "label": pred})
    ss = pd.read_csv("sample_submission.csv")
    assert list(sub.columns) == list(ss.columns)
    assert len(sub) == len(ss) and set(sub.id) == set(ss.id) and sub.id.is_unique
    sub.to_csv("outputs/submission.csv", index=False)
    print(sub.label.value_counts().to_dict())
    print("wrote outputs/submission.csv", sub.shape)


if __name__ == "__main__":
    main()

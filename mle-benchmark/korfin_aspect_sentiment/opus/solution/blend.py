"""Blend cached OOF probabilities (greedy forward selection with replacement)."""
import os
import sys
import glob
import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CLASSES, TASK  # noqa

CACHE = os.path.join(TASK, "solution", "cache")


def load_all():
    names, oofs, teps = [], [], []
    for f in sorted(glob.glob(f"{CACHE}/*_oof.npy")):
        n = os.path.basename(f)[:-8]
        t = f"{CACHE}/{n}_test.npy"
        if not os.path.exists(t):
            continue
        names.append(n)
        oofs.append(np.load(f))
        teps.append(np.load(t))
    return names, np.stack(oofs), np.stack(teps)


def greedy(oofs, y, names, rounds=40):
    n = len(oofs)
    w = np.zeros(n)
    cur = np.zeros_like(oofs[0])
    best_hist = []
    for r in range(rounds):
        best, bi = -1, -1
        for i in range(n):
            c = (cur * r + oofs[i]) / (r + 1)
            s = f1_score(y, c.argmax(1), average="macro")
            if s > best:
                best, bi = s, i
        cur = (cur * r + oofs[bi]) / (r + 1)
        w[bi] += 1
        best_hist.append((best, names[bi]))
    w = w / w.sum()
    return w, best_hist


if __name__ == "__main__":
    y = np.load(f"{CACHE}/y.npy")
    names, oofs, teps = load_all()
    print(f"{len(names)} models")
    for nm, o in zip(names, oofs):
        print(f"  {nm:14s} {f1_score(y, o.argmax(1), average='macro'):.4f}")
    print("mean-all:", f1_score(y, oofs.mean(0).argmax(1), average="macro"))
    w, hist = greedy(oofs, y, names)
    for s, nm in hist:
        print(f"   +{nm:14s} -> {s:.4f}")
    print("weights:", {n: round(float(x), 3) for n, x in zip(names, w) if x > 0})
    blend_oof = np.tensordot(w, oofs, axes=1)
    blend_te = np.tensordot(w, teps, axes=1)
    print("greedy blend oof f1:", f1_score(y, blend_oof.argmax(1), average="macro"))
    np.save(f"{CACHE}/BLEND_oof.npy", blend_oof)
    np.save(f"{CACHE}/BLEND_test.npy", blend_te)

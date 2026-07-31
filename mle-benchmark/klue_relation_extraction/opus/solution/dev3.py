"""Per-block cached harness: ablation + model/blend search."""
import os, sys, time, pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder

from features import build_text_fields
from model import make_vectorizers, WEIGHTS

CACHE = "blocks_dev.pkl"


def build_blocks():
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    tr = pd.read_csv("../train.csv")
    a, b, ytr, yva = train_test_split(
        tr, tr["label"].values, test_size=0.2, random_state=42,
        stratify=tr["label"].values)
    t = time.time()
    tfa, numa = build_text_fields(a)
    tfb, numb = build_text_fields(b)
    print("text fields %.0fs" % (time.time() - t), flush=True)
    A, B = {}, {}
    for name, col, v in make_vectorizers():
        t = time.time()
        A[name] = v.fit_transform(tfa[col].values).astype(np.float32)
        B[name] = v.transform(tfb[col].values).astype(np.float32)
        print(f"  {name}: {A[name].shape[1]} feats ({time.time()-t:.0f}s)", flush=True)
    ohe = OneHotEncoder(handle_unknown="ignore")
    tpa = np.stack([tfa["stype"].values, tfa["otype"].values], axis=1)
    tpb = np.stack([tfb["stype"].values, tfb["otype"].values], axis=1)
    A["type_oh"] = ohe.fit_transform(tpa).astype(np.float32)
    B["type_oh"] = ohe.transform(tpb).astype(np.float32)
    A["num"] = sp.csr_matrix(numa)
    B["num"] = sp.csr_matrix(numb)
    obj = (A, ytr, B, yva)
    with open(CACHE, "wb") as f:
        pickle.dump(obj, f, protocol=4)
    return obj


A, ytr, B, yva = build_blocks()
ALL = list(A.keys())


def assemble(names, w=None):
    w = w or WEIGHTS
    Xa = sp.hstack([A[n] * w.get(n, 1.0) for n in names], format="csr")
    Xb = sp.hstack([B[n] * w.get(n, 1.0) for n in names], format="csr")
    return Xa, Xb


def run(names, C=0.2, w=None, tag=""):
    Xa, Xb = assemble(names, w)
    m = LinearSVC(C=C, max_iter=4000, dual=True)
    t = time.time()
    m.fit(Xa, ytr)
    p = m.predict(Xb)
    acc = accuracy_score(yva, p)
    print(f"{tag or 'run'} n={len(names)} d={Xa.shape[1]} C={C}: acc={acc:.4f} "
          f"f1m={f1_score(yva,p,average='macro'):.4f} ({time.time()-t:.0f}s)", flush=True)
    return acc


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "base"
    if mode == "base":
        base = run(ALL, C=0.2, tag="ALL")
    elif mode == "ablate":
        base = run(ALL, C=0.2, tag="ALL")
        for n in ALL:
            sub = [x for x in ALL if x != n]
            a = run(sub, C=0.2, tag=f"-{n}")
            print(f"   delta {n}: {a-base:+.4f}", flush=True)

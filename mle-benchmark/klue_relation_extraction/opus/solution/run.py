"""Final reproducible pipeline: train on full train.csv, predict test.csv.

Usage:  python run.py [--models svc02,svc05,...]
Writes ../outputs/submission.csv
"""
import argparse, os, time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, RidgeClassifier, LogisticRegression
from sklearn.naive_bayes import ComplementNB

from model import FeatureBuilder

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def make_model(name):
    if name == "svc015":
        return LinearSVC(C=0.15, max_iter=4000, dual=True)
    if name == "svc025bal":
        return LinearSVC(C=0.25, max_iter=4000, dual=True, class_weight="balanced")
    if name == "svc02":
        return LinearSVC(C=0.2, max_iter=4000, dual=True)
    if name == "svc05":
        return LinearSVC(C=0.5, max_iter=4000, dual=True)
    if name == "svc035":
        return LinearSVC(C=0.35, max_iter=4000, dual=True)
    if name == "svcbal":
        return LinearSVC(C=0.2, max_iter=4000, dual=True, class_weight="balanced")
    if name == "svchinge":
        return LinearSVC(C=0.05, max_iter=6000, dual=True, loss="hinge")
    if name == "ridge":
        return RidgeClassifier(alpha=1.0)
    if name == "sgdlog":
        return SGDClassifier(loss="log_loss", alpha=1e-6, max_iter=30, n_jobs=4, random_state=0)
    if name == "sgdmh":
        return SGDClassifier(loss="modified_huber", alpha=1e-6, max_iter=30, n_jobs=4, random_state=0)
    if name == "cnb":
        return ComplementNB(alpha=0.3)
    if name == "lrliblin":
        return LogisticRegression(C=2.0, solver="liblinear", max_iter=500)
    raise ValueError(name)


def scores(m, X, classes):
    if hasattr(m, "predict_proba"):
        P = np.log(np.clip(m.predict_proba(X), 1e-9, None))
    else:
        P = m.decision_function(X)
    idx = [list(m.classes_).index(c) for c in classes]
    P = P[:, idx]
    return (P - P.mean(1, keepdims=True)) / (P.std(1, keepdims=True) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="svc02")
    ap.add_argument("--neigh_w", type=float, default=1.0)
    ap.add_argument("--no_neigh", action="store_true")
    args = ap.parse_args()
    spec = []
    for part in args.models.split(","):
        if not part:
            continue
        n, _, w = part.partition(":")
        spec.append((n, float(w) if w else 1.0))

    t0 = time.time()
    tr = pd.read_csv(os.path.join(ROOT, "train.csv"))
    te = pd.read_csv(os.path.join(ROOT, "test.csv"))
    y = tr["label"].values
    classes = np.unique(y)

    fb = FeatureBuilder(use_neighbors=not args.no_neigh, neigh_w=args.neigh_w)
    Xtr = fb.fit_transform(tr, y)
    Xte = fb.transform(te)
    print("features", Xtr.shape, "%.0fs" % (time.time() - t0), flush=True)

    total = np.zeros((Xte.shape[0], len(classes)), dtype=np.float64)
    for n, w in spec:
        t = time.time()
        m = make_model(n)
        m.fit(Xtr, y)
        total += w * scores(m, Xte, classes)
        print(f"  {n} (w={w}) done ({time.time()-t:.0f}s)", flush=True)

    pred = classes[total.argmax(1)]
    out = pd.DataFrame({"id": te["id"].values, "label": pred})
    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    out.to_csv(os.path.join(ROOT, "outputs", "submission.csv"), index=False)
    print("wrote submission", out.shape, "total %.0fs" % (time.time() - t0))
    print(out.label.value_counts().head(8))


if __name__ == "__main__":
    main()

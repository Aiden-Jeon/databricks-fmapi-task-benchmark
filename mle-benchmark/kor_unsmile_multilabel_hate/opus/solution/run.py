"""Final pipeline: build submission from an ensemble of linear models on TF-IDF features.

Usage: python run.py            # from solution/ directory
Writes ../outputs/submission.csv
"""
import os, time, json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import KFold

from common import load, norm, macro_f1, tune_thresholds, decide, NL, NAMES

BASE = os.environ.get("TASK_DIR", "..")
SEEDS = [0, 1, 2]
NFOLD = 5


def vec_a():
    return FeatureUnion([
        ("cw", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                               sublinear_tf=True, max_features=400000)),
        ("w", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
    ])


def vec_b():
    return FeatureUnion([
        ("c", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2,
                              sublinear_tf=True, max_features=600000)),
        ("w", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
    ])


MODELS = [
    ("svcA", vec_a, lambda: LinearSVC(C=1.0, max_iter=5000, dual=True)),
    ("svcB", vec_b, lambda: LinearSVC(C=1.0, max_iter=5000, dual=True)),
    ("lrA", vec_a, lambda: LogisticRegression(C=20.0, max_iter=3000, solver="liblinear")),
]


def sc(m, X):
    if hasattr(m, "predict_proba"):
        return m.predict_proba(X)[:, 1]
    return 1.0 / (1.0 + np.exp(-m.decision_function(X)))


def run_model(name, vecf, modf, texts, Y, tte):
    n = len(texts)
    P = np.zeros((n, NL)); Pte = np.zeros((len(tte), NL)); cnt = 0
    for seed in SEEDS:
        kf = KFold(NFOLD, shuffle=True, random_state=seed)
        Ps = np.zeros((n, NL))
        for tri, vai in kf.split(np.arange(n)):
            vec = vecf()
            Xtr = vec.fit_transform(texts[tri])
            Xva = vec.transform(texts[vai]); Xte = vec.transform(tte)
            for j in range(NL):
                y = Y[tri, j]
                if y.sum() < 2:
                    continue
                m = modf(); m.fit(Xtr, y)
                Ps[vai, j] = sc(m, Xva)
                Pte[:, j] += sc(m, Xte)
                cnt = cnt  # noqa
        P += Ps
    P /= len(SEEDS)
    Pte /= (len(SEEDS) * NFOLD)
    return P, Pte


def main():
    t0 = time.time()
    tr, te, Y = load(BASE)
    texts = tr["sentence"].map(norm).values
    tte = te["sentence"].map(norm).values

    oofs, tests = {}, {}
    for name, vecf, modf in MODELS:
        P, Pte = run_model(name, vecf, modf, texts, Y, tte)
        oofs[name], tests[name] = P, Pte
        th = tune_thresholds(Y, P)
        print(f"[{name}] oof tuned macroF1 = {macro_f1(Y, decide(P, th)):.4f} ({time.time()-t0:.0f}s)", flush=True)

    # rank-average ensemble (equal weights)
    def ranknorm(P):
        R = np.zeros_like(P)
        for j in range(P.shape[1]):
            o = P[:, j].argsort().argsort()
            R[:, j] = o / max(len(o) - 1, 1)
        return R

    Pe = np.mean([ranknorm(oofs[k]) for k in oofs], 0)
    Pte_e = np.mean([ranknorm(tests[k]) for k in tests], 0)
    th = tune_thresholds(Y, Pe)
    print(f"[ens] oof tuned macroF1 = {macro_f1(Y, decide(Pe, th)):.4f}", flush=True)

    Yp = decide(Pte_e, th)
    labels = ["".join(map(str, r)) for r in Yp]
    os.makedirs(f"{BASE}/outputs", exist_ok=True)
    pd.DataFrame({"id": te["id"], "labels": labels}).to_csv(f"{BASE}/outputs/submission.csv", index=False)
    print("wrote submission", time.time() - t0)


if __name__ == "__main__":
    main()

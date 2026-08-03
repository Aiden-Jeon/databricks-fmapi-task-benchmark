"""CV experiment harness: evaluate model configs with 5-fold OOF + threshold tuning."""
import sys, time
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import ComplementNB

from common import load, norm, make_vec, macro_f1, tune_thresholds, decide, NL, NAMES


def get_model(kind, C):
    if kind == "lr":
        return LogisticRegression(C=C, max_iter=2000, solver="liblinear")
    if kind == "svc":
        return LinearSVC(C=C, max_iter=5000)
    if kind == "sgd":
        return SGDClassifier(loss="modified_huber", alpha=C, max_iter=3000, random_state=0)
    if kind == "cnb":
        return ComplementNB(alpha=C)
    raise ValueError(kind)


def scores(m, X):
    if hasattr(m, "predict_proba"):
        return m.predict_proba(X)[:, 1]
    d = m.decision_function(X)
    return 1.0 / (1.0 + np.exp(-d))


def oof(texts, Y, Xte_texts, kind, C, nfold=5, seed=0):
    """Returns OOF prob matrix and test prob matrix (mean over folds)."""
    n = len(texts)
    P = np.zeros((n, NL))
    Pte = np.zeros((len(Xte_texts), NL))
    kf = KFold(nfold, shuffle=True, random_state=seed)
    for tri, vai in kf.split(np.arange(n)):
        vec = make_vec()
        Xtr = vec.fit_transform(texts[tri])
        Xva = vec.transform(texts[vai])
        Xte = vec.transform(Xte_texts)
        for j in range(NL):
            y = Y[tri, j]
            if y.sum() < 2:
                continue
            m = get_model(kind, C)
            m.fit(Xtr, y)
            P[vai, j] = scores(m, Xva)
            Pte[:, j] += scores(m, Xte) / nfold
    return P, Pte


if __name__ == "__main__":
    tr, te, Y = load("..")
    texts = tr["sentence"].map(norm).values
    tte = te["sentence"].map(norm).values
    for kind, C in [("lr", 4.0), ("lr", 10.0), ("svc", 0.5), ("sgd", 1e-5)]:
        t0 = time.time()
        P, _ = oof(texts, Y, tte[:50], kind, C)
        th = tune_thresholds(Y, P)
        print(f"{kind} C={C}: macroF1@0.5={macro_f1(Y, decide(P, np.full(NL,0.5))):.4f} "
              f"tuned={macro_f1(Y, decide(P, th)):.4f} ({time.time()-t0:.0f}s)", flush=True)

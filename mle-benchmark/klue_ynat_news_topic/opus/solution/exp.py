"""Experiment harness: 5-fold CV macro-F1 for candidate models on YNAT."""
import sys, time, re
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score
from sklearn.calibration import CalibratedClassifierCV

TASK = "/tmp/kmle/M1_t3_ynat_full_20260804_033458/task"
tr = pd.read_csv(f"{TASK}/train.csv")
X, y = tr.title.values, tr.label.values


def feats(word_ng=(1, 2), char_ng=(2, 5), min_df=2, sub=True):
    w = TfidfVectorizer(analyzer="word", ngram_range=word_ng, min_df=min_df,
                        sublinear_tf=sub, lowercase=True)
    c = TfidfVectorizer(analyzer="char_wb", ngram_range=char_ng, min_df=min_df,
                        sublinear_tf=sub, lowercase=True)
    return make_union(w, c, n_jobs=1)


def cv(name, pipe, n=5):
    t0 = time.time()
    skf = StratifiedKFold(n_splits=n, shuffle=True, random_state=42)
    p = cross_val_predict(pipe, X, y, cv=skf, n_jobs=1)
    s = f1_score(y, p, average="macro")
    print(f"{name:55s} macroF1={s:.5f}  ({time.time()-t0:.0f}s)", flush=True)
    return s


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "base"
    if which == "base":
        cv("word12+char25 | LinearSVC C=1", make_pipeline(feats(), LinearSVC(C=1)))
    elif which == "svc":
        for C in [0.3, 0.5, 1.0, 2.0]:
            cv(f"LinearSVC C={C}", make_pipeline(feats(), LinearSVC(C=C)))
    elif which == "lr":
        for C in [2, 5, 10, 20]:
            cv(f"LogReg C={C}", make_pipeline(
                feats(), LogisticRegression(C=C, max_iter=2000, n_jobs=4)))
    elif which == "nb":
        for a in [0.1, 0.3, 1.0]:
            cv(f"CNB a={a}", make_pipeline(feats(), ComplementNB(alpha=a)))
    elif which == "featgrid":
        for wn in [(1, 1), (1, 2)]:
            for cn in [(1, 4), (2, 5), (2, 6), (1, 5)]:
                cv(f"w{wn} c{cn} SVC1", make_pipeline(feats(wn, cn), LinearSVC(C=1)))
    elif which == "mindf":
        for md in [1, 2, 3]:
            cv(f"min_df={md} SVC1", make_pipeline(feats(min_df=md), LinearSVC(C=1)))

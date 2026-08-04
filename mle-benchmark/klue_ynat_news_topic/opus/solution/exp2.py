"""Focused grid round 2."""
import sys, time
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score

TASK = "/tmp/kmle/M1_t3_ynat_full_20260804_033458/task"
tr = pd.read_csv(f"{TASK}/train.csv")
X, y = tr.title.values, tr.label.values


def feats(word_ng=(1, 2), char_ng=(1, 4), min_df=1, sub=True):
    return make_union(
        TfidfVectorizer(analyzer="word", ngram_range=word_ng, min_df=min_df, sublinear_tf=sub),
        TfidfVectorizer(analyzer="char_wb", ngram_range=char_ng, min_df=min_df, sublinear_tf=sub),
    )


def cv(name, pipe, n=5):
    t0 = time.time()
    skf = StratifiedKFold(n_splits=n, shuffle=True, random_state=42)
    p = cross_val_predict(pipe, X, y, cv=skf, n_jobs=1)
    print(f"{name:55s} macroF1={f1_score(y,p,average='macro'):.5f} ({time.time()-t0:.0f}s)", flush=True)


w = sys.argv[1]
if w == "cgrid":
    for C in [0.08, 0.12, 0.2, 0.3, 0.45]:
        cv(f"md1 c14 SVC C={C}", make_pipeline(feats(), LinearSVC(C=C, dual=True)))
elif w == "cw":
    for C in [0.12, 0.2, 0.3]:
        cv(f"md1 c14 SVC C={C} balanced", make_pipeline(
            feats(), LinearSVC(C=C, dual=True, class_weight="balanced")))
elif w == "charng":
    for cn in [(1, 3), (1, 4), (1, 5), (1, 6), (2, 4)]:
        cv(f"md1 w12 c{cn} SVC.2", make_pipeline(feats(char_ng=cn), LinearSVC(C=0.2, dual=True)))
elif w == "lr":
    for C in [3, 8, 20, 50]:
        cv(f"md1 c14 LogReg C={C}", make_pipeline(
            feats(), LogisticRegression(C=C, max_iter=3000)))
elif w == "wordng":
    for wn in [(1, 1), (1, 2), (1, 3)]:
        cv(f"md1 w{wn} c14 SVC.2", make_pipeline(feats(word_ng=wn), LinearSVC(C=0.2, dual=True)))
elif w == "ridge":
    for a in [0.3, 1.0, 3.0]:
        cv(f"md1 c14 Ridge a={a}", make_pipeline(feats(), RidgeClassifier(alpha=a)))
elif w == "sub":
    for s in [True, False]:
        cv(f"md1 c14 sub={s} SVC.2", make_pipeline(feats(sub=s), LinearSVC(C=0.2, dual=True)))

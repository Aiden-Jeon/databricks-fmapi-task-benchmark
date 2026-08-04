"""KLUE-RE relation extraction — reproducible end-to-end solution.

Environment: CPU only, no torch/transformers available, no internet.
Approach: heavy surface-feature engineering around the two entity mentions +
sparse TF-IDF n-grams, classified with a linear SVM (squared-hinge, OvR),
blended with a small-weight SGD (modified-huber) model.

Pipeline
--------
1. `features2.build2` locates the subject/object mentions in the sentence
   (choosing the closest occurrence pair), infers a coarse entity type for
   each (PER/ORG/LOC/DAT/NOH/MSC via regex + Korean suffix lexicons) and
   emits several textual views:
     marked      : sentence with typed entity markers @S#TYPE# / ^O@TYPE^
     between     : the text between the two mentions, prefixed by the type pair
     near_s/o    : +-18 chars around each mention
     subj/obj    : the entity strings themselves
     sent        : raw sentence (bag of words)
     *_ex, pat_* : exact-string one-hot views (entity / between-text / type pair)
     edge, heads : adjacent tokens, first/last token of the between-span
     *_tail      : entity prefix/suffix characters
   plus ~60 dense numeric features (lengths, distances, order, type one-hots,
   trigger-word indicators such as 출생/사망/졸업/설립/취임/소속 ...).
2. Each textual view is vectorised with TF-IDF (char_wb or word n-grams,
   see `VEC_SPECS`), numeric features are standardised, everything is
   hstacked into one sparse matrix (~770k features, ~525 nnz/row).
3. LinearSVC(C=0.3) is the main model; SGDClassifier is blended at weight
   0.3 after per-row z-scoring of the decision scores.

Holdout (stratified 25% of train.csv) accuracy
----------------------------------------------
  TF-IDF v1 + LinearSVC(C=0.5)          0.7311
  TF-IDF v2 + LinearSVC(C=1.0)          0.7364
  TF-IDF v2 + LinearSVC(C=0.5)          0.7378
  TF-IDF v2 + LinearSVC(C=0.3)          0.7390
  TF-IDF v2 + SGD(modified_huber)       0.7023
  cosine kNN (k=30) on the same space   0.5916   (hurts when blended)
  0.3*SGD + 1.0*LinearSVC(C=0.3)        0.7404   <- submitted blend

Usage
-----
  python solution/run.py            # writes outputs/submission.csv
Runtime: ~10 min on 4 CPU cores.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features2 import VEC_SPECS, build2

BLEND = {"svc": 1.0, "sgd": 0.3}


def zs(M):
    M = M.astype(np.float64)
    return (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-9)


def main():
    t0 = time.time()
    tr = pd.read_csv("train.csv")
    te = pd.read_csv("test.csv")
    Xt, Nt = build2(tr)
    Xs, Ns = build2(te)
    y = tr.label.values
    classes = np.array(sorted(set(y)))

    A, B = [], []
    for col, kw in VEC_SPECS:
        v = TfidfVectorizer(**kw)
        A.append(v.fit_transform(Xt[col]))
        B.append(v.transform(Xs[col]))
    sc = StandardScaler()
    A.append(sp.csr_matrix(sc.fit_transform(Nt)))
    B.append(sp.csr_matrix(sc.transform(Ns)))
    Xa = sp.hstack(A).tocsr()
    Xb = sp.hstack(B).tocsr()
    print("features", Xa.shape, round(time.time() - t0, 1), flush=True)

    models = {
        "svc": LinearSVC(C=0.3, dual=True),
        "sgd": SGDClassifier(loss="modified_huber", alpha=2e-7, max_iter=30,
                             tol=None, random_state=0),
    }
    total = np.zeros((Xb.shape[0], len(classes)))
    for name, w in BLEND.items():
        s = time.time()
        clf = models[name]
        clf.fit(Xa, y)
        assert np.array_equal(clf.classes_, classes)
        total += w * zs(clf.decision_function(Xb))
        print(name, "fitted", round(time.time() - s, 1), flush=True)

    pred = classes[total.argmax(1)]
    os.makedirs("outputs", exist_ok=True)
    pd.DataFrame({"id": te.id, "label": pred}).to_csv("outputs/submission.csv",
                                                     index=False)
    print("wrote outputs/submission.csv", len(pred), round(time.time() - t0, 1))


if __name__ == "__main__":
    main()

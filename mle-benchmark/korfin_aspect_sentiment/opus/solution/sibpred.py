"""Leak-free sibling predictions.

Every sentence (train OR test) is assigned to one of K sentence-folds. For fold k a
model is trained on train rows whose sentence is NOT in fold k, and used to predict
all rows (train and test) in fold k. Hence for any row, the predictions available for
its siblings come from a model that never saw *any* row of that sentence -> the row's
own label cannot leak through its siblings, and train/test are treated identically.
"""
import os
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, CLASSES, TASK  # noqa
from views import add_views, SPECS  # noqa

CACHE = os.path.join(TASK, "solution", "cache")
K = 5
SEED = 42


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def main():
    tr, te = load()
    tr, te = add_views(tr), add_views(te)
    le = LabelEncoder().fit(CLASSES)
    y = le.transform(tr.label.values)
    sents = pd.unique(pd.concat([tr.sentence, te.sentence]))
    rng = np.random.RandomState(SEED)
    fold_of = {s: i for s, i in zip(sents, rng.randint(0, K, len(sents)))}
    ftr = tr.sentence.map(fold_of).values
    fte = te.sentence.map(fold_of).values

    Ptr = np.zeros((len(tr), 3))
    Pte = np.zeros((len(te), 3))
    for k in range(K):
        i_tr = np.where(ftr != k)[0]
        j_tr = np.where(ftr == k)[0]
        j_te = np.where(fte == k)[0]
        specs = SPECS["multi"] + [("text2", dict(analyzer="char_wb", ngram_range=(2, 5),
                                                 min_df=2, sublinear_tf=True))]
        Xa, Xb, Xc = [], [], []
        for col, kw in specs:
            v = TfidfVectorizer(**kw)
            Xa.append(v.fit_transform(tr[col].iloc[i_tr]))
            Xb.append(v.transform(tr[col].iloc[j_tr]))
            Xc.append(v.transform(te[col].iloc[j_te]))
        Xa, Xb, Xc = sp.hstack(Xa).tocsr(), sp.hstack(Xb).tocsr(), sp.hstack(Xc).tocsr()
        m1 = LogisticRegression(C=2, max_iter=3000).fit(Xa, y[i_tr])
        m2 = LinearSVC(C=0.3, dual=True, max_iter=5000).fit(Xa, y[i_tr])
        Ptr[j_tr] = 0.5 * m1.predict_proba(Xb) + 0.5 * softmax(2 * m2.decision_function(Xb))
        Pte[j_te] = 0.5 * m1.predict_proba(Xc) + 0.5 * softmax(2 * m2.decision_function(Xc))
        print(f"  fold{k}: train={len(i_tr)} pred_tr={len(j_tr)} pred_te={len(j_te)}")
    print("sentence-group OOF f1 (train):",
          round(f1_score(y, Ptr.argmax(1), average="macro"), 4))
    np.save(f"{CACHE}/SIBP_tr.npy", Ptr)
    np.save(f"{CACHE}/SIBP_te.npy", Pte)


if __name__ == "__main__":
    main()

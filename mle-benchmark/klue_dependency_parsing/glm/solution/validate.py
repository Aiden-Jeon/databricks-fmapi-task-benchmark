import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solution import (
    build_pair_dataset,
    build_deprel_dataset,
    decode_sentence,
    pair_features,
)

HERE = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.dirname(HERE)
TRAIN_CSV = os.path.join(TASK_DIR, "train.csv")


def compute_las(tokens, gold_parse, pred_parse):
    n = len(tokens)
    g = gold_parse.split("|")
    p = pred_parse.split("|")
    if len(g) != n or len(p) != n:
        return 0.0, n
    correct = 0
    for i in range(n):
        gh, gd = g[i].rsplit(":", 1)
        ph, pd_ = p[i].rsplit(":", 1)
        if int(gh) == int(ph) and gd == pd_:
            correct += 1
    return correct, n


def main():
    train = pd.read_csv(TRAIN_CSV)
    rng = np.random.RandomState(123)
    idx = rng.permutation(len(train))
    n_tr = int(len(train) * 0.8)
    tr = train.iloc[idx[:n_tr]].reset_index(drop=True)
    va = train.iloc[idx[n_tr:]].reset_index(drop=True)
    print("train", len(tr), "val", len(va))

    rng2 = np.random.RandomState(42)
    max_neg = int(os.environ.get("MAX_NEG", "3"))
    Xh, yh, _ = build_pair_dataset(tr, include_negative=True, max_neg_per_mod=max_neg, rng=rng2)
    head_vec = DictVectorizer(sparse=True)
    Xhv = head_vec.fit_transform(Xh)
    del Xh
    print("head feat dim", Xhv.shape[1], "max_neg", max_neg)
    solver = os.environ.get("SOLVER", "lbfgs")
    C = float(os.environ.get("C", "0.3"))
    head_clf = LogisticRegression(C=C, max_iter=300, solver=solver, n_jobs=1)
    head_clf.fit(Xhv, yh)
    print("head train acc", (head_clf.predict(Xhv) == yh).mean(), "solver", solver, "C", C)
    del Xhv, yh

    Xd, yd = build_deprel_dataset(tr)
    dep_vec = DictVectorizer(sparse=True)
    Xdv = dep_vec.fit_transform(Xd)
    del Xd
    dep_le = LabelEncoder()
    yde = dep_le.fit_transform(yd)
    dsolver = os.environ.get("DSOLVER", "lbfgs")
    dC = float(os.environ.get("DC", "0.5"))
    dep_clf = LogisticRegression(C=dC, max_iter=300, solver=dsolver, n_jobs=1)
    dep_clf.fit(Xdv, yde)
    print("dep train acc", (dep_clf.predict(Xdv) == yde).mean(), "solver", dsolver, "C", dC)
    del Xdv, yde

    total_correct = 0
    total_tokens = 0
    uas_correct = 0
    for _, row in va.iterrows():
        tok = json.loads(row["tokens"])
        pred = decode_sentence(tok, head_clf, head_vec, dep_clf, dep_vec, dep_le)
        c, n = compute_las(tok, row["parse"], pred)
        total_correct += c
        total_tokens += n
        g = row["parse"].split("|")
        p = pred.split("|")
        for i in range(n):
            if int(g[i].rsplit(":", 1)[0]) == int(p[i].rsplit(":", 1)[0]):
                uas_correct += 1
    print("UAS", uas_correct / total_tokens)
    print("LAS", total_correct / total_tokens)


if __name__ == "__main__":
    main()

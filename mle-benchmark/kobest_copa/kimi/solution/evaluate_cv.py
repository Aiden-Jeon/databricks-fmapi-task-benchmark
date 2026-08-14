#!/usr/bin/env python3
"""CV evaluation of the final model (for reference / reproducibility).

Runs 5-fold stratified CV of the exact pipeline in solution.py and prints
mean accuracy. Expected: ~0.60 (single 5-fold split, seed 42).
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from solution import (swap_augment, pointwise_texts, BASE)
import os


def main():
    train = pd.read_csv(os.path.join(BASE, "train.csv"))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs = []
    for trn, val in skf.split(train, train["label"]):
        trd, vad = train.iloc[trn], train.iloc[val]
        aug = swap_augment(trd, seed=42)
        t_texts, t_y = pointwise_texts(aug, with_label=True)
        v_texts, _ = pointwise_texts(vad, with_label=False)
        vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2,
                              sublinear_tf=True)
        Xtr = vec.fit_transform(t_texts)
        Xva = vec.transform(v_texts)
        clf = LogisticRegression(C=1.0, max_iter=4000, solver="liblinear",
                                 random_state=42)
        clf.fit(Xtr, t_y)
        s = np.asarray(clf.decision_function(Xva)).ravel().reshape(-1, 2)
        pick = (s[:, 1] > s[:, 0]).astype(int)
        accs.append((pick == vad["label"].values).mean())
    print(f"5-fold CV accuracy: {np.mean(accs):.4f} +- {np.std(accs):.4f}")
    print("folds:", [f"{a:.4f}" for a in accs])


if __name__ == "__main__":
    main()

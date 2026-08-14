"""Final model: TF-IDF(char_wb 2-5, min_df=3) + [LogisticRegression(0.75) x 0.75 + ComplementNB(0.3) x 0.25]
probability blend, averaged over 3 CV seeds. Repeated-CV macro F1 ~ 0.572.
Writes outputs/submission.csv.
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

SEEDS = [0, 1, 2]
CLASSES = np.array(["hate", "none", "offensive"])  # sklearn sorted order
LR_W, CNB_W = 0.75, 0.25


def fit_proba(Xtr, ytr, Xev):
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.75)
    lr.fit(Xtr, ytr)
    nb = ComplementNB(alpha=0.3)
    nb.fit(Xtr, ytr)
    return LR_W * lr.predict_proba(Xev) + CNB_W * nb.predict_proba(Xev)


def main():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    y = train["label"].values

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True)
    Xtr = vec.fit_transform(train["comment"].astype(str))
    Xte = vec.transform(test["comment"].astype(str))

    # ---- repeated-CV estimate ----
    oof = np.zeros((len(y), 3))
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for tr_idx, va_idx in skf.split(Xtr, y):
            oof[va_idx] += fit_proba(Xtr[tr_idx], y[tr_idx], Xtr[va_idx]) / len(SEEDS)
    cv_f1 = f1_score(y, CLASSES[oof.argmax(axis=1)], average="macro")
    print(f"repeated-CV (3 seeds) macro F1: {cv_f1:.4f}")

    # ---- full-data fit & predict ----
    proba = np.zeros((len(test), 3))
    for seed in SEEDS:
        proba += fit_proba(Xtr, y, Xte) / len(SEEDS)
    pred = CLASSES[proba.argmax(axis=1)]

    sub = pd.DataFrame({"id": test["id"], "label": pred})
    sub.to_csv("outputs/submission.csv", index=False)
    print(sub["label"].value_counts())
    print("saved outputs/submission.csv")


if __name__ == "__main__":
    main()

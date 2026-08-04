"""Baseline solution for KorNLI (t10) using TF-IDF + Logistic Regression.

Combines premise+hypothesis with word and char n-grams. Trains on train.csv
and writes predictions to outputs/submission.csv.
"""
import os
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(BASE, "train.csv")
TEST = os.path.join(BASE, "test.csv")
OUT = os.path.join(BASE, "outputs", "submission.csv")

RANDOM_STATE = 42


def make_text(df):
    return (df["sentence1"].fillna("") + " [SEP] " + df["sentence2"].fillna("")).values


def main():
    train = pd.read_csv(TRAIN)
    test = pd.read_csv(TEST)

    X_text = make_text(train)
    X_test_text = make_text(test)

    le = LabelEncoder()
    y = le.fit_transform(train["label"])

    word_vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        sublinear_tf=True,
        max_features=200000,
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        min_df=3,
        max_df=0.9,
        sublinear_tf=True,
        max_features=200000,
    )

    Xw = word_vec.fit_transform(X_text)
    Xcw = char_vec.fit_transform(X_text)
    X = hstack([Xw, Xcw]).tocsr()

    Xtw = word_vec.transform(X_test_text)
    Xtcw = char_vec.transform(X_test_text)
    Xt = hstack([Xtw, Xtcw]).tocsr()

    clf = LogisticRegression(
        C=4.0,
        max_iter=1000,
        n_jobs=-1,
        solver="liblinear",
        random_state=RANDOM_STATE,
    )
    clf.fit(X, y)

    # quick CV score
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    from sklearn.metrics import accuracy_score
    cv_scores = []
    for tr, va in skf.split(X, y):
        c = LogisticRegression(C=4.0, max_iter=1000, n_jobs=-1, solver="liblinear",
                                random_state=RANDOM_STATE)
        c.fit(X[tr], y[tr])
        pred = c.predict(X[va])
        cv_scores.append(accuracy_score(y[va], pred))
    print("CV scores:", cv_scores, "mean:", np.mean(cv_scores))

    pred = clf.predict(Xt)
    labels = le.inverse_transform(pred)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out = pd.DataFrame({"id": test["id"], "label": labels})
    out.to_csv(OUT, index=False)
    print("Wrote", OUT, out.shape)
    print(out.head())


if __name__ == "__main__":
    main()

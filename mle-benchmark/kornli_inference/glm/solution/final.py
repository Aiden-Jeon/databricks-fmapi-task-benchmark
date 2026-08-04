"""Final solution for KorNLI (t10) using TF-IDF + Logistic Regression.

Approach: char n-gram TF-IDF features computed separately on sentence1 (premise)
and sentence2 (hypothesis), concatenated. Logistic Regression with low C.
Trains on train.csv and writes predictions to outputs/submission.csv.
"""
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(BASE, "train.csv")
TEST = os.path.join(BASE, "test.csv")
OUT = os.path.join(BASE, "outputs", "submission.csv")

RANDOM_STATE = 42


def main():
    train = pd.read_csv(TRAIN)
    test = pd.read_csv(TEST)

    s1 = train["sentence1"].fillna("").values
    s2 = train["sentence2"].fillna("").values
    ts1 = test["sentence1"].fillna("").values
    ts2 = test["sentence2"].fillna("").values

    le = LabelEncoder()
    y = le.fit_transform(train["label"])

    vec1 = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=3,
                           max_df=0.9, sublinear_tf=True, max_features=100000)
    vec2 = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=3,
                           max_df=0.9, sublinear_tf=True, max_features=100000)

    X1 = vec1.fit_transform(s1)
    X2 = vec2.fit_transform(s2)
    X = hstack([X1, X2]).tocsr()

    Xt1 = vec1.transform(ts1)
    Xt2 = vec2.transform(ts2)
    Xt = hstack([Xt1, Xt2]).tocsr()

    # CV for validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y), dtype=int)
    test_proba = np.zeros((len(test), len(le.classes_)))
    for tr, va in skf.split(X, y):
        c = LogisticRegression(C=0.1, max_iter=2000, solver="liblinear",
                                random_state=RANDOM_STATE)
        c.fit(X[tr], y[tr])
        oof[va] = c.predict(X[va])
        test_proba += c.predict_proba(Xt) / skf.n_splits
    print("OOF accuracy:", accuracy_score(y, oof))

    pred = np.argmax(test_proba, axis=1)
    labels = le.inverse_transform(pred)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out = pd.DataFrame({"id": test["id"], "label": labels})
    out.to_csv(OUT, index=False)
    print("Wrote", OUT, out.shape)


if __name__ == "__main__":
    main()

"""Baseline: TF-IDF + Logistic Regression for KoBEST BoolQ."""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold, cross_val_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "train.csv")
TEST = os.path.join(ROOT, "test.csv")
SUB = os.path.join(ROOT, "outputs", "submission.csv")

RANDOM_STATE = 42


class TextSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key].astype(str).values


class CharNgramTfidf(TfidfVectorizer):
    pass


def build_model():
    word_vec = TfidfVectorizer(
        sublinear_tf=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=50000,
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
    )
    char_vec = TfidfVectorizer(
        sublinear_tf=True,
        ngram_range=(2, 4),
        min_df=2,
        max_df=0.95,
        max_features=50000,
        analyzer="char_wb",
    )
    features = FeatureUnion([
        ("word", word_vec),
        ("char", char_vec),
    ])
    clf = LogisticRegression(
        C=4.0,
        max_iter=2000,
        solver="liblinear",
        random_state=RANDOM_STATE,
    )
    return Pipeline([("features", features), ("clf", clf)])


def combine(p, q):
    return (p + " " + q).str.replace("\n", " ")


def main():
    train = pd.read_csv(TRAIN)
    test = pd.read_csv(TEST)
    train["text"] = combine(train["paragraph"], train["question"])
    test["text"] = combine(test["paragraph"], test["question"])

    model = build_model()

    # CV estimate
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(
        model, train["text"], train["label"],
        cv=skf, scoring="accuracy", n_jobs=-1,
    )
    print("CV accuracy: %.4f +/- %.4f" % (scores.mean(), scores.std()), file=sys.stderr)

    model.fit(train["text"], train["label"])
    preds = model.predict(test["text"])

    out = pd.DataFrame({"id": test["id"], "label": preds.astype(int)})
    os.makedirs(os.path.dirname(SUB), exist_ok=True)
    out.to_csv(SUB, index=False)
    print("Saved", SUB, "with", len(out), "rows", file=sys.stderr)


if __name__ == "__main__":
    main()

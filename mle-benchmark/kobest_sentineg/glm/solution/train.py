"""Solution for KoBEST SentiNeg sentiment classification task.

Approach:
  - Korean character n-gram TF-IDF features (combined char + char_wb analyzers)
  - Light preprocessing: wrap each sentence with surrounding spaces so that
    word-boundary n-grams ("char_wb") capture word-initial/word-final substrings.
  - RidgeClassifier linear model (robust to high-dim sparse TF-IDF features).
  - 5-fold StratifiedKFold OOF evaluation for reference; final model trained on
    the full train.csv and used to predict test.csv.

Reproducible: all randomness controlled via random_state=42.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.dirname(HERE)
TRAIN_CSV = os.path.join(TASK_DIR, "train.csv")
TEST_CSV = os.path.join(TASK_DIR, "test.csv")
OUT_CSV = os.path.join(TASK_DIR, "outputs", "submission.csv")

RANDOM_STATE = 42

# Final hyperparameters (selected via 6-seed x 5-fold CV sweep).
SPECS = [
    {"analyzer": "char", "ngram_range": (1, 4), "min_df": 2, "binary": False},
    {"analyzer": "char_wb", "ngram_range": (1, 4), "min_df": 2, "binary": False},
]
RIDGE_ALPHA = 1.75


def preprocess(text):
    """Wrap each sentence with surrounding spaces to help char_wb word boundaries."""
    return " " + str(text).strip() + " "


class CharMultiVectorizer(BaseEstimator, TransformerMixin):
    """Concatenate TF-IDF feature matrices from multiple TfidfVectorizer configs."""

    def __init__(self, specs):
        self.specs = specs
        self.vecs = [
            TfidfVectorizer(
                analyzer=s.get("analyzer", "char"),
                ngram_range=s["ng"],
                min_df=s.get("min_df", 2),
                sublinear_tf=True,
                binary=s.get("binary", False),
            )
            for s in specs
        ]

    def fit(self, X, y=None):
        for v in self.vecs:
            v.fit(X)
        return self

    def transform(self, X):
        return hstack([v.transform(X) for v in self.vecs]).tocsr()


def build_model():
    specs = [
        {"analyzer": s["analyzer"], "ng": s["ngram_range"], "min_df": s["min_df"], "binary": s["binary"]}
        for s in SPECS
    ]
    return Pipeline([
        ("feat", CharMultiVectorizer(specs)),
        ("clf", RidgeClassifier(alpha=RIDGE_ALPHA, random_state=RANDOM_STATE)),
    ])


def main():
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)

    X = train["sentence"].astype(str).map(preprocess)
    y = train["label"].values
    X_test = test["sentence"].astype(str).map(preprocess)

    # Reference CV score (uses only train.csv; does not touch test).
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(build_model(), X, y, cv=cv, scoring="accuracy", n_jobs=1)
    print(f"[CV] accuracy: {scores.mean():.4f} +/- {scores.std():.4f} (folds={scores.tolist()})")

    # Train final model on the full training set and predict test.
    model = build_model()
    model.fit(X, y)
    preds = model.predict(X_test)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    sub = pd.DataFrame({"id": test["id"], "label": preds.astype(int)})
    sub.to_csv(OUT_CSV, index=False)
    print(f"[Submission] wrote {len(sub)} rows to {OUT_CSV}")
    print(sub.head())
    print("Label distribution:", sub["label"].value_counts().to_dict())


if __name__ == "__main__":
    main()

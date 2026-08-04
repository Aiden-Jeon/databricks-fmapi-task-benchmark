#!/usr/bin/env python3
"""Small local model comparison used to select the final text features."""

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from train import ROOT


def char(ngram_range: tuple[int, int], min_df: int = 2) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char", ngram_range=ngram_range, min_df=min_df, sublinear_tf=True
    )


def word(ngram_range: tuple[int, int], min_df: int = 2) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="word", ngram_range=ngram_range, min_df=min_df, sublinear_tf=True
    )


def main() -> None:
    train = pd.read_csv(ROOT / "train.csv")
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=2026)
    candidates = {
        "char_2_5": char((2, 5)),
        "char_1_5": char((1, 5)),
        "char_2_6": char((2, 6)),
        "word_1_2": word((1, 2)),
        "char25_word12": FeatureUnion(
            [("char", char((2, 5))), ("word", word((1, 2)))],
            transformer_weights={"char": 1.0, "word": 0.7},
        ),
        "char15_word13": FeatureUnion(
            [("char", char((1, 5))), ("word", word((1, 3), min_df=1))],
            transformer_weights={"char": 1.0, "word": 0.7},
        ),
    }
    for name, features in candidates.items():
        model = Pipeline(
            [("features", features), ("classifier", LinearSVC(C=1.0, class_weight="balanced"))]
        )
        prediction = cross_val_predict(model, train.comment, train.label, cv=folds, n_jobs=1)
        score = f1_score(train.label, prediction, average="macro")
        print(f"{name}: {score:.6f}", flush=True)


if __name__ == "__main__":
    main()

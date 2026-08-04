#!/usr/bin/env python3
"""Train the NSMC classifiers and create a submission file."""

from __future__ import annotations

import argparse
import gc
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


SEED = 2026


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=root / "train.csv")
    parser.add_argument("--test", type=Path, default=root / "test.csv")
    parser.add_argument(
        "--output", type=Path, default=root / "outputs" / "submission.csv"
    )
    return parser.parse_args()


def fit_scores(
    train_text: np.ndarray,
    train_label: np.ndarray,
    test_text: np.ndarray,
    vectorizer: TfidfVectorizer,
    c: float,
) -> np.ndarray:
    train_matrix = vectorizer.fit_transform(train_text)
    test_matrix = vectorizer.transform(test_text)
    model = LinearSVC(C=c, dual=True, random_state=SEED)
    model.fit(train_matrix, train_label)
    scores = model.decision_function(test_matrix)
    del train_matrix, test_matrix, model, vectorizer
    gc.collect()
    return scores


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)

    expected_train = {"id", "document", "label"}
    expected_test = {"id", "document"}
    if set(train.columns) != expected_train:
        raise ValueError(f"Unexpected train columns: {list(train.columns)}")
    if set(test.columns) != expected_test:
        raise ValueError(f"Unexpected test columns: {list(test.columns)}")
    if train[list(expected_train)].isna().any().any():
        raise ValueError("train.csv contains missing values")
    if test[list(expected_test)].isna().any().any():
        raise ValueError("test.csv contains missing values")
    if not set(train["label"].unique()).issubset({0, 1}):
        raise ValueError("Training labels must be 0 or 1")
    if not test["id"].is_unique:
        raise ValueError("Test IDs must be unique")

    train_text = train["document"].astype(str).to_numpy()
    test_text = test["document"].astype(str).to_numpy()
    labels = train["label"].to_numpy()

    raw_char_scores = fit_scores(
        train_text,
        labels,
        test_text,
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(1, 5),
            min_df=2,
            max_features=700_000,
            sublinear_tf=True,
            dtype=np.float32,
        ),
        c=0.4,
    )

    no_space_train = np.array([re.sub(r"\s+", "", text) for text in train_text])
    no_space_test = np.array([re.sub(r"\s+", "", text) for text in test_text])
    no_space_scores = fit_scores(
        no_space_train,
        labels,
        no_space_test,
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 6),
            min_df=2,
            max_features=500_000,
            sublinear_tf=True,
            dtype=np.float32,
        ),
        c=0.25,
    )
    del no_space_train, no_space_test

    word_scores = fit_scores(
        train_text,
        labels,
        test_text,
        TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_features=300_000,
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b",
            dtype=np.float32,
        ),
        c=0.25,
    )

    scores = 0.7 * raw_char_scores + 0.2 * no_space_scores + 0.1 * word_scores
    submission = pd.DataFrame(
        {"id": test["id"].to_numpy(), "label": (scores > 0).astype(np.int8)}
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"Wrote {len(submission):,} predictions to {args.output}")


if __name__ == "__main__":
    main()

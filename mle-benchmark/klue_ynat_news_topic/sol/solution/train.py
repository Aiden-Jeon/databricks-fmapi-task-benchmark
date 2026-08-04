#!/usr/bin/env python3
"""Train a TF-IDF linear classifier and create the YNAT submission."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


CLASSES = {"IT과학", "경제", "사회", "생활문화", "세계", "스포츠", "정치"}


def make_vectorizers() -> tuple[TfidfVectorizer, TfidfVectorizer]:
    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 6),
        min_df=2,
        max_df=0.995,
        sublinear_tf=True,
        max_features=350_000,
        dtype=np.float32,
    )
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.995,
        sublinear_tf=True,
        max_features=180_000,
        dtype=np.float32,
    )
    return char_vectorizer, word_vectorizer


def vectorize(
    train_text: pd.Series, predict_text: pd.Series
) -> tuple[object, object]:
    char_vectorizer, word_vectorizer = make_vectorizers()
    train_char = char_vectorizer.fit_transform(train_text)
    predict_char = char_vectorizer.transform(predict_text)
    train_word = word_vectorizer.fit_transform(train_text)
    predict_word = word_vectorizer.transform(predict_text)

    # L2-normalized blocks get equal influence before the SVM learns its weights.
    train_features = hstack((train_char, train_word), format="csr")
    predict_features = hstack((predict_char, predict_word), format="csr")
    return train_features, predict_features


def validate(train: pd.DataFrame, c_values: list[float]) -> None:
    fit_rows, valid_rows = train_test_split(
        np.arange(len(train)),
        test_size=0.2,
        random_state=2026,
        stratify=train["label"],
    )
    fit_x, valid_x = vectorize(
        train.loc[fit_rows, "title"], train.loc[valid_rows, "title"]
    )
    fit_y = train.loc[fit_rows, "label"]
    valid_y = train.loc[valid_rows, "label"]

    for c_value in c_values:
        model = LinearSVC(C=c_value, class_weight="balanced", dual="auto")
        model.fit(fit_x, fit_y)
        prediction = model.predict(valid_x)
        score = f1_score(valid_y, prediction, average="macro")
        print(f"C={c_value:g} macro_f1={score:.6f}")
        print(classification_report(valid_y, prediction, digits=4))


def train_and_submit(
    train: pd.DataFrame, test: pd.DataFrame, output_path: Path, c_value: float
) -> None:
    train_x, test_x = vectorize(train["title"], test["title"])
    model = LinearSVC(C=c_value, class_weight="balanced", dual="auto")
    model.fit(train_x, train["label"])
    prediction = model.predict(test_x)

    submission = pd.DataFrame({"id": test["id"], "label": prediction})
    if submission.columns.tolist() != ["id", "label"]:
        raise ValueError("Submission columns are invalid")
    if len(submission) != len(test) or submission["id"].duplicated().any():
        raise ValueError("Submission must contain each test id exactly once")
    if submission["id"].tolist() != test["id"].tolist():
        raise ValueError("Submission ids are not in test.csv order")
    if not set(submission["label"]).issubset(CLASSES):
        raise ValueError("Submission contains an invalid label")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Wrote {len(submission)} predictions to {output_path}")
    print(submission["label"].value_counts().sort_index())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("train.csv"))
    parser.add_argument("--test", type=Path, default=Path("test.csv"))
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/submission.csv")
    )
    parser.add_argument("--c", type=float, default=0.1)
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Evaluate C values on a fixed stratified holdout instead of submitting",
    )
    parser.add_argument(
        "--c-grid", type=float, nargs="+", default=[0.06, 0.1, 0.15, 0.2]
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.train)
    expected_train_columns = ["id", "title", "label"]
    if train.columns.tolist() != expected_train_columns:
        raise ValueError(f"Expected train columns {expected_train_columns}")
    if set(train["label"]) != CLASSES:
        raise ValueError("Training labels do not match the seven expected classes")

    if args.validate:
        validate(train, args.c_grid)
        return

    test = pd.read_csv(args.test)
    if test.columns.tolist() != ["id", "title"]:
        raise ValueError("Expected test columns ['id', 'title']")
    train_and_submit(train, test, args.output, args.c)


if __name__ == "__main__":
    main()

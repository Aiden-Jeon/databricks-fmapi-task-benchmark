#!/usr/bin/env python3
"""Train the KMMLU model and create outputs/submission.csv."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


CHOICES = ["A", "B", "C", "D"]
TOKEN_PATTERN = re.compile(r"[가-힣A-Za-z0-9]+")
EXPLICIT_ANSWER = re.compile(r"정답[^.]{0,80}?(?:은|인|이|:)?\s*([1-4])번")


def structure_features(frame: pd.DataFrame) -> np.ndarray:
    """Extract position-aware style and length features from each question."""
    rows: list[list[float]] = []
    for record in frame.itertuples(index=False):
        question = str(record.question)
        question_tokens = set(TOKEN_PATTERN.findall(question.lower()))
        features: list[float] = []
        raw: list[list[float]] = []

        for column in CHOICES:
            answer = str(getattr(record, column))
            tokens = TOKEN_PATTERN.findall(answer.lower())
            token_set = set(tokens)
            values = [
                len(answer),
                len(tokens),
                sum(char.isdigit() for char in answer),
                sum(char in ".,()%" for char in answer),
                len(question_tokens & token_set) / (len(token_set) + 1),
                int("없" in answer),
                int("아니" in answer),
                int("모두" in answer),
                int("항상" in answer),
                int("수 있다" in answer),
                int("하여야" in answer),
            ]
            raw.append(values)
            features.extend(values)

        lengths = np.asarray([values[0] for values in raw])
        word_counts = np.asarray([values[1] for values in raw])
        features.extend(lengths / (lengths.mean() + 1))
        features.extend(word_counts / (word_counts.mean() + 1))
        features.extend(lengths == lengths.max())
        features.extend(lengths == lengths.min())
        rows.append(features)

    return np.asarray(rows, dtype=np.float32)


def retrieval_predictions(
    train: pd.DataFrame, test: pd.DataFrame, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transfer answers from lexically similar training questions."""
    question_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=2,
        max_features=150_000,
        sublinear_tf=True,
    )
    train_questions = question_vectorizer.fit_transform(train["question"])
    test_questions = question_vectorizer.transform(test["question"])

    train_options = train[CHOICES].astype(str).to_numpy()
    test_options = test[CHOICES].astype(str).to_numpy()
    option_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=2,
        max_features=180_000,
        sublinear_tf=True,
    )
    train_option_vectors = option_vectorizer.fit_transform(train_options.ravel())
    test_option_vectors = option_vectorizer.transform(test_options.ravel())

    similarities = (test_questions @ train_questions.T).toarray()
    neighbor_count = min(12, len(train))
    neighbors = np.argpartition(similarities, -neighbor_count, axis=1)[
        :, -neighbor_count:
    ]
    neighbor_similarities = np.take_along_axis(similarities, neighbors, axis=1)
    order = np.argsort(-neighbor_similarities, axis=1)
    neighbors = np.take_along_axis(neighbors, order, axis=1)
    neighbor_similarities = np.take_along_axis(
        neighbor_similarities, order, axis=1
    )

    scores = np.zeros((len(test), 4), dtype=np.float32)
    for row in range(len(test)):
        neighbor_rows = neighbors[row]
        correct_option_rows = neighbor_rows * 4 + labels[neighbor_rows]
        answer_similarities = (
            test_option_vectors[row * 4 : (row + 1) * 4]
            @ train_option_vectors[correct_option_rows].T
        ).toarray()
        weighted = answer_similarities * neighbor_similarities[row][None, :] ** 2
        scores[row] = weighted.max(axis=1)

    return scores.argmax(axis=1), scores.max(axis=1), neighbor_similarities[:, 0]


def run(root: Path) -> Path:
    train = pd.read_csv(root / "train.csv").fillna("")
    test = pd.read_csv(root / "test.csv").fillna("")
    sample = pd.read_csv(root / "sample_submission.csv")

    expected_train = {"id", "question", *CHOICES, "label"}
    expected_test = {"id", "question", *CHOICES}
    if not expected_train.issubset(train.columns):
        raise ValueError("train.csv does not contain the required columns")
    if not expected_test.issubset(test.columns):
        raise ValueError("test.csv does not contain the required columns")
    if not train["label"].isin([1, 2, 3, 4]).all():
        raise ValueError("Training labels must be in 1..4")

    labels = train["label"].to_numpy(dtype=np.int64) - 1
    train_features = structure_features(train)
    test_features = structure_features(test)
    classifier = ExtraTreesClassifier(
        n_estimators=1_500,
        min_samples_leaf=10,
        max_features=0.8,
        n_jobs=-1,
        random_state=20260801,
    )
    classifier.fit(train_features, labels)
    predictions = classifier.predict(test_features)

    retrieval, retrieval_score, question_score = retrieval_predictions(
        train, test, labels
    )
    use_retrieval = (retrieval_score > 0.20) & (question_score > 0.20)
    predictions[use_retrieval] = retrieval[use_retrieval]

    # A few source questions explicitly state the accepted answer after revisions.
    for row, question in enumerate(test["question"].astype(str)):
        match = EXPLICIT_ANSWER.search(question)
        if match:
            predictions[row] = int(match.group(1)) - 1

    submission = pd.DataFrame(
        {"id": test["id"].to_numpy(), "label": predictions.astype(int) + 1}
    )
    if list(submission.columns) != list(sample.columns):
        raise ValueError("Submission columns do not match sample_submission.csv")
    if len(submission) != len(test) or not submission["id"].is_unique:
        raise ValueError("Submission IDs are incomplete or duplicated")
    if set(submission["id"]) != set(test["id"]):
        raise ValueError("Submission IDs differ from test.csv")
    if not submission["label"].isin([1, 2, 3, 4]).all():
        raise ValueError("Predicted labels must be in 1..4")

    output_dir = root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "submission.csv"
    submission.to_csv(output_path, index=False)
    print(f"Wrote {len(submission)} predictions to {output_path}")
    print(f"Label counts: {submission['label'].value_counts().sort_index().to_dict()}")
    print(f"Retrieval overrides: {int(use_retrieval.sum())}")
    return output_path


def main() -> None:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=default_root)
    args = parser.parse_args()
    run(args.root.resolve())


if __name__ == "__main__":
    main()

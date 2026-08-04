#!/usr/bin/env python3
"""Train a reproducible KoBEST BoolQ model and create the submission."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC


NEGATION_TERMS = (
    "않",
    "아니",
    "없",
    "못",
    "금지",
    "불가능",
    "제외",
    "반대",
    "달리",
    "더 이상",
    "없이",
    "실패",
    "거부",
    "중단",
    "폐지",
    "부정",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def normalize(text: str) -> str:
    return re.sub(r"[^가-힣a-z0-9]", "", text.lower())


def bigram_recall(question: str, sentence: str) -> float:
    question_bigrams = {
        question[index : index + 2] for index in range(max(0, len(question) - 1))
    }
    sentence = normalize(sentence)
    sentence_bigrams = {
        sentence[index : index + 2] for index in range(max(0, len(sentence) - 1))
    }
    return len(question_bigrams & sentence_bigrams) / max(1, len(question_bigrams))


def add_relation_markers(row: dict[str, str]) -> str:
    paragraph = row["paragraph"]
    question = row["question"]
    normalized_question = normalize(question)
    sentences = [
        sentence
        for sentence in re.split(r"(?<=[.!?])\s+|다\.\s*", paragraph)
        if sentence
    ]
    nearest_sentence = max(
        sentences,
        key=lambda sentence: bigram_recall(normalized_question, sentence),
        default=paragraph,
    )

    parts = [question]
    for term in NEGATION_TERMS:
        paragraph_state = int(term in paragraph)
        nearest_state = int(term in nearest_sentence)
        question_state = int(term in question)
        parts.append(f"WHOLEPOL{term}{paragraph_state}{question_state}")
        parts.append(f"NEARPOL{term}{nearest_state}{question_state}")

    paragraph_numbers = set(re.findall(r"\d+", paragraph))
    question_numbers = set(re.findall(r"\d+", question))
    parts.append(f"NUMMISS{int(bool(question_numbers - paragraph_numbers))}")
    return " ".join(parts)


def make_vectorizer(kind: str) -> TfidfVectorizer:
    if kind == "char":
        return TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            min_df=2,
            max_features=200_000,
            sublinear_tf=True,
        )
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=100_000,
        sublinear_tf=True,
    )


def cross_validated_scale(
    texts: list[str], labels: np.ndarray, kind: str, c_value: float
) -> float:
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    scores = np.zeros(len(labels), dtype=float)
    for train_indices, validation_indices in folds.split(texts, labels):
        vectorizer = make_vectorizer(kind)
        train_matrix = vectorizer.fit_transform(
            [texts[index] for index in train_indices]
        )
        validation_matrix = vectorizer.transform(
            [texts[index] for index in validation_indices]
        )
        model = LinearSVC(C=c_value, dual="auto")
        model.fit(train_matrix, labels[train_indices])
        scores[validation_indices] = model.decision_function(validation_matrix)
    return float(scores.std())


def fit_predict(
    train_texts: list[str],
    test_texts: list[str],
    labels: np.ndarray,
    kind: str,
    c_value: float,
) -> np.ndarray:
    vectorizer = make_vectorizer(kind)
    train_matrix = vectorizer.fit_transform(train_texts)
    test_matrix = vectorizer.transform(test_texts)
    model = LinearSVC(C=c_value, dual="auto")
    model.fit(train_matrix, labels)
    return model.decision_function(test_matrix)


def main() -> None:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=default_root)
    parser.add_argument(
        "--output", type=Path, default=default_root / "outputs" / "submission.csv"
    )
    args = parser.parse_args()

    train_rows = read_csv(args.data_dir / "train.csv")
    test_rows = read_csv(args.data_dir / "test.csv")
    sample_rows = read_csv(args.data_dir / "sample_submission.csv")
    labels = np.asarray([int(row["label"]) for row in train_rows], dtype=int)

    test_ids = [row["id"] for row in test_rows]
    sample_ids = [row["id"] for row in sample_rows]
    if len(test_ids) != len(set(test_ids)) or test_ids != sample_ids:
        raise ValueError("Test IDs must be unique and match sample_submission.csv order")

    train_texts = [add_relation_markers(row) for row in train_rows]
    test_texts = [add_relation_markers(row) for row in test_rows]

    char_scale = cross_validated_scale(train_texts, labels, "char", 0.25)
    word_scale = cross_validated_scale(train_texts, labels, "word", 0.30)
    char_scores = fit_predict(train_texts, test_texts, labels, "char", 0.25)
    word_scores = fit_predict(train_texts, test_texts, labels, "word", 0.30)
    scores = char_scores + 0.4 * (char_scale / word_scale) * word_scores
    predictions = (scores > 0.0).astype(int)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "label"])
        writer.writeheader()
        writer.writerows(
            {"id": sample_id, "label": int(label)}
            for sample_id, label in zip(sample_ids, predictions, strict=True)
        )

    print(f"Wrote {len(predictions)} predictions to {args.output}")
    print(f"Predicted class counts: {np.bincount(predictions, minlength=2).tolist()}")


if __name__ == "__main__":
    main()

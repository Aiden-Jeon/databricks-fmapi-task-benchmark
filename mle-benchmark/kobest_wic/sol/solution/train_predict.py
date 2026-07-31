#!/usr/bin/env python3
"""Train a local-only KoBEST WiC model and create outputs/submission.csv."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict


SEED = 42
TARGET_RE = re.compile(r"\[[^]]+\]")


def clean_context(text: str) -> str:
    return TARGET_RE.sub(" TARGET ", text)


def basic_features(frame: pd.DataFrame, representations: list[tuple]) -> np.ndarray:
    features: list[list[float]] = []
    similarities = [
        np.asarray(left.multiply(right).sum(axis=1)).ravel()
        for left, right in representations
    ]

    for row_number, row in enumerate(frame.itertuples(index=False)):
        left = TARGET_RE.sub("", row.context_1)
        right = TARGET_RE.sub("", row.context_2)
        left_chars, right_chars = set(left), set(right)
        left_words, right_words = set(left.split()), set(right.split())
        features.append(
            [
                *(similarity[row_number] for similarity in similarities),
                len(left),
                len(right),
                abs(len(left) - len(right)),
                len(left_chars & right_chars) / max(1, len(left_chars | right_chars)),
                len(left_words & right_words) / max(1, len(left_words | right_words)),
            ]
        )
    return np.asarray(features, dtype=np.float32)


def reference_features(
    query: pd.DataFrame,
    reference: pd.DataFrame,
    query_representations: list[tuple],
    reference_representations: list[tuple],
    reference_labels: np.ndarray,
) -> np.ndarray:
    """Compare each pair with labeled pairs having the same target word."""
    reference_by_word = {
        word: np.flatnonzero(reference["word"].to_numpy() == word)
        for word in reference["word"].unique()
    }
    output = np.zeros((len(query), 16 * len(query_representations)), dtype=np.float32)

    for query_number, word in enumerate(query["word"]):
        candidates = reference_by_word.get(word)
        if candidates is None or not len(candidates):
            continue

        row_features: list[float] = []
        candidate_labels = reference_labels[candidates]
        for (query_left, query_right), (ref_left, ref_right) in zip(
            query_representations, reference_representations
        ):
            s11 = (query_left[query_number] @ ref_left[candidates].T).toarray().ravel()
            s12 = (query_left[query_number] @ ref_right[candidates].T).toarray().ravel()
            s21 = (query_right[query_number] @ ref_left[candidates].T).toarray().ravel()
            s22 = (query_right[query_number] @ ref_right[candidates].T).toarray().ravel()

            aligned = np.maximum(s11 + s22, s12 + s21) / 2
            any_context = np.maximum.reduce([s11, s12, s21, s22])
            both_contexts = np.maximum(np.minimum(s11, s22), np.minimum(s12, s21))

            for scores in (aligned, any_context, both_contexts):
                positive = scores[candidate_labels == 1]
                negative = scores[candidate_labels == 0]
                nearest = int(np.argmax(scores))
                weights = np.exp(np.minimum(10 * scores, 20))
                row_features.extend(
                    [
                        float(positive.max()) if len(positive) else 0.0,
                        float(negative.max()) if len(negative) else 0.0,
                        float(candidate_labels[nearest]),
                        float(np.average(candidate_labels, weights=weights)),
                    ]
                )
            row_features.extend(
                [
                    float(aligned.max()),
                    float(any_context.max()),
                    float(len(candidates)),
                    float(candidate_labels.mean()),
                ]
            )
        output[query_number] = row_features
    return output


def subset_representations(representations: list[tuple], rows: np.ndarray) -> list[tuple]:
    return [(left[rows], right[rows]) for left, right in representations]


def run(data_dir: Path, output_path: Path) -> None:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    labels = train["label"].to_numpy(dtype=np.int8)

    train_texts = [clean_context(text) for text in train["context_1"]] + [
        clean_context(text) for text in train["context_2"]
    ]
    test_texts = [clean_context(text) for text in test["context_1"]] + [
        clean_context(text) for text in test["context_2"]
    ]

    vectorizers = [
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            min_df=2,
            max_features=80_000,
            sublinear_tf=True,
        ),
        TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True
        ),
    ]
    train_representations, test_representations = [], []
    train_size, test_size = len(train), len(test)
    for vectorizer in vectorizers:
        train_matrix = vectorizer.fit_transform(train_texts)
        test_matrix = vectorizer.transform(test_texts)
        train_representations.append(
            (train_matrix[:train_size], train_matrix[train_size:])
        )
        test_representations.append((test_matrix[:test_size], test_matrix[test_size:]))

    train_basic = basic_features(train, train_representations)
    test_basic = basic_features(test, test_representations)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_reference = np.zeros((train_size, 16 * len(vectorizers)), dtype=np.float32)

    for fit_rows, valid_rows in splitter.split(train, labels):
        train_reference[valid_rows] = reference_features(
            train.iloc[valid_rows].reset_index(drop=True),
            train.iloc[fit_rows].reset_index(drop=True),
            subset_representations(train_representations, valid_rows),
            subset_representations(train_representations, fit_rows),
            labels[fit_rows],
        )

    test_reference = reference_features(
        test,
        train,
        test_representations,
        train_representations,
        labels,
    )
    train_features = np.hstack([train_basic, train_reference])
    test_features = np.hstack([test_basic, test_reference])

    tree_model = ExtraTreesClassifier(
        n_estimators=1_000,
        min_samples_leaf=5,
        max_features=0.8,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    cv_probability = cross_val_predict(
        tree_model,
        train_features,
        labels,
        cv=splitter,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    print(f"OOF accuracy: {accuracy_score(labels, cv_probability >= 0.5):.4f}")
    tree_model.fit(train_features, labels)
    tree_probability = tree_model.predict_proba(test_features)[:, 1]

    # A low-variance global similarity model stabilizes sparse-word predictions.
    global_model = LogisticRegression(C=1.0, max_iter=2_000, random_state=SEED)
    global_model.fit(train_basic, labels)
    global_probability = global_model.predict_proba(test_basic)[:, 1]
    probability = 0.8 * tree_probability + 0.2 * global_probability

    submission = pd.DataFrame(
        {"id": test["id"].to_numpy(), "label": (probability >= 0.5).astype(int)}
    )
    if list(submission.columns) != list(sample.columns):
        raise ValueError("Submission columns do not match sample_submission.csv")
    if len(submission) != len(test) or submission["id"].nunique() != len(test):
        raise ValueError("Submission must contain every test id exactly once")
    if set(submission["id"]) != set(test["id"]):
        raise ValueError("Submission ids do not match test.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Wrote {len(submission)} predictions to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "submission.csv",
    )
    arguments = parser.parse_args()
    run(arguments.data_dir, arguments.output)

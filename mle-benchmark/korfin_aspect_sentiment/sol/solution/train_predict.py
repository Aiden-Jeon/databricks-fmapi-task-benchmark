#!/usr/bin/env python3
"""Train a character NB-SVM ensemble and create the competition submission."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC


LABELS = np.array(["NEGATIVE", "NEUTRAL", "POSITIVE"])


def build_texts(frame: pd.DataFrame, window: int = 25) -> np.ndarray:
    """Mark the target aspect and emphasize its surrounding context."""
    texts: list[str] = []
    for sentence, aspect in zip(frame["sentence"].astype(str), frame["aspect"].astype(str)):
        starts = [match.start() for match in re.finditer(re.escape(aspect), sentence)]
        if starts:
            local = " ".join(
                sentence[max(0, start - window) : min(len(sentence), start + len(aspect) + window)]
                for start in starts
            )
        else:
            local = sentence

        marked = sentence.replace(aspect, f" [ASP] {aspect} [/ASP] ")
        local_block = f"{aspect} [LOCAL] {local} "
        texts.append(local_block * 2 + "[FULL] " + marked)
    return np.asarray(texts)


def make_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 7),
        min_df=2,
        max_features=220_000,
        sublinear_tf=True,
        dtype=np.float32,
    )


def fit_predict_nbsvm(
    train_matrix: sparse.spmatrix,
    labels: np.ndarray,
    predict_matrix: sparse.spmatrix,
    c: float = 0.5,
) -> np.ndarray:
    """Fit one NB-SVM classifier per class and return decision scores."""
    scores = np.zeros((predict_matrix.shape[0], len(LABELS)), dtype=np.float64)
    for class_index, label in enumerate(LABELS):
        positive = labels == label
        positive_rate = (np.asarray(train_matrix[positive].sum(axis=0)).ravel() + 1.0) / (
            positive.sum() + 1.0
        )
        negative_rate = (np.asarray(train_matrix[~positive].sum(axis=0)).ravel() + 1.0) / (
            (~positive).sum() + 1.0
        )
        log_ratio = np.log(positive_rate / negative_rate).astype(np.float32)

        model = LinearSVC(C=c, class_weight="balanced", dual=True, random_state=42)
        model.fit(train_matrix.multiply(log_ratio), positive)
        scores[:, class_index] = model.decision_function(predict_matrix.multiply(log_ratio))
    return scores


def run(data_dir: Path, output_path: Path) -> None:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")

    expected_train = {"id", "sentence", "aspect", "label"}
    expected_test = {"id", "sentence", "aspect"}
    if set(train.columns) != expected_train or set(test.columns) != expected_test:
        raise ValueError("Unexpected train.csv or test.csv schema")
    if not set(train["label"]).issubset(LABELS):
        raise ValueError("train.csv contains an unknown label")

    train_text = build_texts(train)
    test_text = build_texts(test)
    y = train["label"].to_numpy()

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    test_score_sum = np.zeros((len(test), len(LABELS)), dtype=np.float64)
    oof_scores = np.zeros((len(train), len(LABELS)), dtype=np.float64)

    for fold, (fit_indices, valid_indices) in enumerate(folds.split(train_text, y), start=1):
        vectorizer = make_vectorizer()
        fit_matrix = vectorizer.fit_transform(train_text[fit_indices])
        valid_matrix = vectorizer.transform(train_text[valid_indices])
        test_matrix = vectorizer.transform(test_text)

        oof_scores[valid_indices] = fit_predict_nbsvm(
            fit_matrix, y[fit_indices], valid_matrix
        )
        test_score_sum += fit_predict_nbsvm(fit_matrix, y[fit_indices], test_matrix)
        print(f"Finished fold {fold}/5")

    oof_predictions = LABELS[oof_scores.argmax(axis=1)]
    print(f"5-fold OOF macro F1: {f1_score(y, oof_predictions, average='macro'):.6f}")

    # Add a model trained on every available labeled row to the fold ensemble.
    full_vectorizer = make_vectorizer()
    full_train_matrix = full_vectorizer.fit_transform(train_text)
    full_test_matrix = full_vectorizer.transform(test_text)
    test_score_sum += fit_predict_nbsvm(full_train_matrix, y, full_test_matrix)

    predictions = LABELS[test_score_sum.argmax(axis=1)]
    submission = pd.DataFrame({"id": test["id"], "label": predictions})

    if list(submission.columns) != list(sample.columns):
        raise ValueError("Submission columns do not match sample_submission.csv")
    if len(submission) != len(test) or submission["id"].duplicated().any():
        raise ValueError("Submission IDs are missing or duplicated")
    if set(submission["id"]) != set(test["id"]):
        raise ValueError("Submission IDs do not exactly match test.csv")
    if not set(submission["label"]).issubset(LABELS):
        raise ValueError("Submission contains an invalid label")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Wrote {len(submission)} predictions to {output_path}")
    print(submission["label"].value_counts().sort_index().to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    default_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data-dir", type=Path, default=default_root)
    parser.add_argument(
        "--output",
        type=Path,
        default=default_root / "outputs" / "submission.csv",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.data_dir, args.output)

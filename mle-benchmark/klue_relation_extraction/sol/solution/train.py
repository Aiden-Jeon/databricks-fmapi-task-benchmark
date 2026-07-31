#!/usr/bin/env python3
"""Train a local-only KLUE-RE classifier and create the submission."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC


SEED = 20260731


def marked_text(row: pd.Series) -> str:
    sentence = str(row["sentence"])
    subject = str(row["subject_entity"])
    obj = str(row["object_entity"])

    if subject != obj:
        entities = sorted((subject, obj), key=len, reverse=True)
        pattern = re.compile("|".join(re.escape(entity) for entity in entities))

        def add_marker(match: re.Match[str]) -> str:
            entity = match.group(0)
            role = "SUBJ" if entity == subject else "OBJ"
            return f" [{role}] {entity} [/{role}] "

        sentence = pattern.sub(add_marker, sentence)

    # Repeating the explicit pair gives the sparse model direct entity features.
    return f"{sentence} [SUBJECT] {subject} [OBJECT] {obj} [PAIR] {subject} {obj}"


def make_text(frame: pd.DataFrame) -> list[str]:
    return [marked_text(row) for _, row in frame.iterrows()]


def entity_categories(frame: pd.DataFrame) -> pd.DataFrame:
    categories = frame[["subject_entity", "object_entity"]].astype(str).copy()
    categories["entity_pair"] = categories["subject_entity"] + "\x1f" + categories["object_entity"]
    return categories


def fit_features(train: pd.DataFrame, other: pd.DataFrame):
    train_text = make_text(train)
    other_text = make_text(other)
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=250_000,
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b",
        dtype=np.float32,
    )
    char = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=2,
        max_features=400_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    entities = OneHotEncoder(
        handle_unknown="ignore", min_frequency=2, dtype=np.float32
    )
    train_matrix = sparse.hstack(
        (
            word.fit_transform(train_text),
            char.fit_transform(train_text),
            entities.fit_transform(entity_categories(train)),
        ),
        format="csr",
    )
    other_matrix = sparse.hstack(
        (
            word.transform(other_text),
            char.transform(other_text),
            entities.transform(entity_categories(other)),
        ),
        format="csr",
    )
    return train_matrix, other_matrix


def pair_statistics(frame: pd.DataFrame) -> dict[tuple[str, str], tuple[str, int, float]]:
    stats: dict[tuple[str, str], tuple[str, int, float]] = {}
    for pair, group in frame.groupby(["subject_entity", "object_entity"], sort=False):
        counts = group["label"].value_counts()
        stats[pair] = (str(counts.index[0]), int(counts.iloc[0]), float(counts.iloc[0] / counts.sum()))
    return stats


def pair_override(
    predictions: np.ndarray,
    frame: pd.DataFrame,
    stats: dict[tuple[str, str], tuple[str, int, float]],
    min_majority: int,
    min_purity: float,
) -> np.ndarray:
    result = predictions.copy()
    if min_majority <= 0:
        return result
    for i, row in enumerate(frame.itertuples(index=False)):
        value = stats.get((str(row.subject_entity), str(row.object_entity)))
        if value is not None and value[1] >= min_majority and value[2] >= min_purity:
            result[i] = value[0]
    return result


def validate(data: pd.DataFrame) -> None:
    fit, valid = train_test_split(
        data, test_size=0.2, random_state=SEED, stratify=data["label"]
    )
    x_fit, x_valid = fit_features(fit, valid)
    stats = pair_statistics(fit)

    for c in (0.25, 0.5, 0.75, 1.0):
        model = LinearSVC(C=c, dual=True, random_state=SEED)
        model.fit(x_fit, fit["label"])
        base = model.predict(x_valid)
        print(f"C={c:g} SVM accuracy: {accuracy_score(valid['label'], base):.6f}")
        for min_majority, min_purity in ((1, 1.0), (2, 0.75), (3, 0.6), (3, 0.75), (5, 0.6)):
            pred = pair_override(base, valid, stats, min_majority, min_purity)
            print(
                f"  pair majority>={min_majority}, purity>={min_purity:.2f}: "
                f"{accuracy_score(valid['label'], pred):.6f}"
            )


def train_and_predict(root: Path, c: float, min_majority: int, min_purity: float) -> Path:
    train = pd.read_csv(root / "train.csv")
    test = pd.read_csv(root / "test.csv")
    sample = pd.read_csv(root / "sample_submission.csv")

    x_train, x_test = fit_features(train, test)
    model = LinearSVC(C=c, dual=True, random_state=SEED)
    model.fit(x_train, train["label"])
    predictions = model.predict(x_test)
    predictions = pair_override(
        predictions, test, pair_statistics(train), min_majority, min_purity
    )

    submission = pd.DataFrame({"id": test["id"], "label": predictions})
    if list(submission.columns) != list(sample.columns):
        raise ValueError("Submission columns do not match sample_submission.csv")
    if len(submission) != len(test) or not submission["id"].is_unique:
        raise ValueError("Submission must contain every test id exactly once")
    if submission["id"].tolist() != test["id"].tolist():
        raise ValueError("Submission id order differs from test.csv")
    if not set(submission["label"]).issubset(set(train["label"])):
        raise ValueError("Submission contains an unknown label")

    output = root / "outputs" / "submission.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    print(f"Wrote {len(submission)} predictions to {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true", help="run a fixed validation split first")
    parser.add_argument("--validate-only", action="store_true", help="run validation without submission")
    parser.add_argument("--c", type=float, default=0.5)
    parser.add_argument(
        "--pair-min-majority", type=int, default=0,
        help="enable majority override with this minimum count (0 disables it)",
    )
    parser.add_argument("--pair-min-purity", type=float, default=0.6)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    if args.validate or args.validate_only:
        validate(pd.read_csv(root / "train.csv"))
    if not args.validate_only:
        train_and_predict(root, args.c, args.pair_min_majority, args.pair_min_purity)


if __name__ == "__main__":
    main()

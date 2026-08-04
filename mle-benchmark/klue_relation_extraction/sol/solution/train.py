#!/usr/bin/env python3
"""Train a reproducible KLUE-RE classifier and create the submission."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]


def mark_entities(row: pd.Series) -> str:
    """Expose entity roles to a bag-of-ngrams model while retaining their text."""
    sentence = str(row["sentence"])
    subject = str(row["subject_entity"])
    obj = str(row["object_entity"])

    # Replace one mention of each entity. Longer-first avoids corrupting nested names.
    entities = sorted(((subject, "SUBJ"), (obj, "OBJ")), key=lambda x: len(x[0]), reverse=True)
    for entity, role in entities:
        sentence = sentence.replace(entity, f" {role}_START {entity} {role}_END ", 1)
    return sentence


def local_context(row: pd.Series, margin: int = 30) -> str:
    sentence = str(row["sentence"])
    subject = str(row["subject_entity"])
    obj = str(row["object_entity"])
    subject_at = sentence.find(subject)
    object_at = sentence.find(obj)
    if subject_at < 0 or object_at < 0:
        return mark_entities(row)
    left = min(subject_at, object_at)
    right = max(subject_at + len(subject), object_at + len(obj))
    local_row = row.copy()
    local_row["sentence"] = sentence[max(0, left - margin) : right + margin]
    return mark_entities(local_row)


def make_texts(frame: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    marked = frame.apply(mark_entities, axis=1).tolist()
    local = frame.apply(local_context, axis=1).tolist()
    entities = (
        "SUBJECT_"
        + frame["subject_entity"].astype(str)
        + " OBJECT_"
        + frame["object_entity"].astype(str)
    ).tolist()
    return marked, local, entities


def build_features(train: pd.DataFrame, other: pd.DataFrame):
    train_marked, train_local, train_entities = make_texts(train)
    other_marked, other_local, other_entities = make_texts(other)

    vectorizers = [
        TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b\w+\b",
            min_df=2,
            max_features=140_000,
            sublinear_tf=True,
            dtype=np.float32,
        ),
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            min_df=2,
            max_features=240_000,
            sublinear_tf=True,
            dtype=np.float32,
        ),
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            min_df=2,
            max_features=100_000,
            sublinear_tf=True,
            dtype=np.float32,
        ),
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            min_df=2,
            max_features=140_000,
            sublinear_tf=True,
            dtype=np.float32,
        ),
    ]

    train_parts = [
        vectorizers[0].fit_transform(train_marked),
        vectorizers[1].fit_transform(train_marked),
        vectorizers[2].fit_transform(train_entities),
        vectorizers[3].fit_transform(train_local),
    ]
    other_parts = [
        vectorizers[0].transform(other_marked),
        vectorizers[1].transform(other_marked),
        vectorizers[2].transform(other_entities),
        vectorizers[3].transform(other_local),
    ]

    categorical = OneHotEncoder(handle_unknown="ignore", dtype=np.float32)
    train_categories = train[["subject_entity", "object_entity"]].astype(str)
    other_categories = other[["subject_entity", "object_entity"]].astype(str)
    train_parts.append(categorical.fit_transform(train_categories))
    other_parts.append(categorical.transform(other_categories))

    return hstack(train_parts, format="csr"), hstack(other_parts, format="csr")


def validate(data: pd.DataFrame) -> None:
    fit, valid = train_test_split(
        data, test_size=0.2, random_state=2026, stratify=data["label"]
    )
    x_fit, x_valid = build_features(fit, valid)
    print(f"feature matrices: train={x_fit.shape}, valid={x_valid.shape}")
    for c in (0.7, 1.0, 1.4, 2.0):
        model = LinearSVC(C=c, dual="auto", random_state=2026)
        model.fit(x_fit, fit["label"])
        prediction = model.predict(x_valid)
        print(f"C={c:.1f} accuracy={accuracy_score(valid['label'], prediction):.6f}")
        if c == 1.0:
            for minimum in (2, 3, 4, 5):
                overridden, changed = override_stable_pairs(
                    fit, valid, prediction, minimum_count=minimum
                )
                score = accuracy_score(valid["label"], overridden)
                print(f"  stable-pair min={minimum} changed={changed} accuracy={score:.6f}")


def override_stable_pairs(
    train: pd.DataFrame,
    other: pd.DataFrame,
    prediction: np.ndarray,
    minimum_count: int = 3,
) -> tuple[np.ndarray, int]:
    """Use a relation only when every observed occurrence of a pair agrees."""
    keys = ["subject_entity", "object_entity"]
    counts = train.groupby(keys + ["label"]).size().rename("label_count").reset_index()
    totals = counts.groupby(keys)["label_count"].transform("sum")
    stable = counts[(counts["label_count"] == totals) & (totals >= minimum_count)]
    lookup = stable.set_index(keys)["label"]

    result = prediction.copy()
    other_keys = pd.MultiIndex.from_frame(other[keys])
    mapped = lookup.reindex(other_keys)
    mask = mapped.notna().to_numpy()
    result[mask] = mapped[mask].to_numpy()
    return result, int(mask.sum())


def train_and_predict(train: pd.DataFrame, test: pd.DataFrame, output: Path, c: float) -> None:
    x_train, x_test = build_features(train, test)
    print(f"feature matrices: train={x_train.shape}, test={x_test.shape}")
    model = LinearSVC(C=c, dual="auto", random_state=2026)
    model.fit(x_train, train["label"])
    prediction = model.predict(x_test)
    prediction, changed = override_stable_pairs(train, test, prediction, minimum_count=3)
    print(f"stable entity-pair predictions applied: {changed}")

    submission = pd.DataFrame({"id": test["id"], "label": prediction})
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    print(f"wrote {len(submission)} predictions to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true", help="run a fixed validation split")
    parser.add_argument("--c", type=float, default=1.0, help="LinearSVC regularization parameter")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs" / "submission.csv"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = pd.read_csv(ROOT / "train.csv")
    if args.validate:
        validate(train)
        return
    test = pd.read_csv(ROOT / "test.csv")
    train_and_predict(train, test, args.output, args.c)


if __name__ == "__main__":
    main()

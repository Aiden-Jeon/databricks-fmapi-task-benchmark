#!/usr/bin/env python3
"""Train a text classifier and create outputs/submission.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
LABELS = ["none", "offensive", "hate"]
OFFENSIVE_SCORE_BONUS = 0.08


def build_model(c: float = 0.4, class_weight: str | None = "balanced") -> Pipeline:
    features = TfidfVectorizer(
        analyzer="char",
        ngram_range=(1, 5),
        min_df=2,
        max_df=0.995,
        sublinear_tf=True,
        max_features=200_000,
    )
    return Pipeline(
        [
            ("features", features),
            ("classifier", LinearSVC(C=c, class_weight=class_weight, dual=True)),
        ]
    )


def validate(train: pd.DataFrame, c: float, class_weight: str | None) -> None:
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    scores = cross_val_predict(
        build_model(c, class_weight),
        train["comment"],
        train["label"],
        cv=folds,
        n_jobs=1,
        method="decision_function",
    )
    classes = np.array(sorted(train["label"].unique()))
    scores[:, np.where(classes == "offensive")[0][0]] += OFFENSIVE_SCORE_BONUS
    prediction = classes[scores.argmax(axis=1)]
    print(f"5-fold macro F1: {f1_score(train['label'], prediction, average='macro'):.6f}")
    print(classification_report(train["label"], prediction, labels=LABELS, digits=4))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", action="store_true", help="run 5-fold validation before training")
    parser.add_argument("--c", type=float, default=0.4, help="LinearSVC regularization strength")
    parser.add_argument(
        "--unweighted", action="store_true", help="disable balanced class weights"
    )
    args = parser.parse_args()

    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    sample = pd.read_csv(ROOT / "sample_submission.csv")
    class_weight = None if args.unweighted else "balanced"

    if args.cv:
        validate(train, args.c, class_weight)

    model = build_model(args.c, class_weight)
    model.fit(train["comment"], train["label"])
    scores = model.decision_function(test["comment"])
    classes = model.named_steps["classifier"].classes_
    scores[:, np.where(classes == "offensive")[0][0]] += OFFENSIVE_SCORE_BONUS
    prediction = classes[scores.argmax(axis=1)]

    submission = pd.DataFrame({"id": test["id"], "label": prediction})
    if list(submission.columns) != list(sample.columns):
        raise ValueError("Submission columns do not match sample_submission.csv")
    if len(submission) != len(test) or submission["id"].nunique() != len(test):
        raise ValueError("Submission must contain every test id exactly once")
    if submission["id"].tolist() != sample["id"].tolist():
        raise ValueError("Submission id order differs from sample_submission.csv")
    if not set(submission["label"]).issubset(LABELS):
        raise ValueError("Submission contains an invalid label")

    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    submission.to_csv(output_dir / "submission.csv", index=False)
    print(f"Wrote {len(submission)} predictions to {output_dir / 'submission.csv'}")
    print(submission["label"].value_counts().to_string())


if __name__ == "__main__":
    main()

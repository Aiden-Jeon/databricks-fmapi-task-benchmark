#!/usr/bin/env python3
"""Train a KMMLU text classifier and create the competition submission."""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parents[1]
TEXT_COLUMNS = ["question", "A", "B", "C", "D"]


def combine_text(frame: pd.DataFrame) -> pd.Series:
    return frame[TEXT_COLUMNS].astype(str).agg(" ".join, axis=1)


def main() -> None:
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    sample = pd.read_csv(ROOT / "sample_submission.csv")

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=120_000,
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b",
    )
    x_train = vectorizer.fit_transform(combine_text(train))
    x_test = vectorizer.transform(combine_text(test))

    model = LogisticRegression(
        C=0.2,
        max_iter=500,
        solver="liblinear",
        random_state=42,
    )
    model.fit(x_train, train["label"])
    predictions = model.predict(x_test).astype(int)

    submission = pd.DataFrame({"id": test["id"], "label": predictions})
    if list(submission.columns) != list(sample.columns):
        raise ValueError("Submission columns do not match sample_submission.csv")
    if len(submission) != len(test) or not submission["id"].is_unique:
        raise ValueError("Submission IDs are incomplete or duplicated")
    if set(submission["id"]) != set(test["id"]):
        raise ValueError("Submission IDs do not match test.csv")
    if not submission["label"].isin([1, 2, 3, 4]).all():
        raise ValueError("Predictions contain an invalid label")

    output_path = ROOT / "outputs" / "submission.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Wrote {len(submission)} predictions to {output_path}")


if __name__ == "__main__":
    main()

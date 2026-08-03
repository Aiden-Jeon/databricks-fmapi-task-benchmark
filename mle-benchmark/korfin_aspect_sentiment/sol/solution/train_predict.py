#!/usr/bin/env python3
"""Train the KorFin-ASC ensemble and create outputs/submission.csv."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
VALID_LABELS = {"NEGATIVE", "NEUTRAL", "POSITIVE"}


def make_texts(frame: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    full_texts: list[str] = []
    local_texts: list[str] = []
    aspects: list[str] = []

    for sentence, aspect in zip(frame["sentence"].astype(str), frame["aspect"].astype(str)):
        position = sentence.find(aspect)
        if position >= 0:
            local = sentence[
                max(0, position - 35) : position + len(aspect) + 35
            ]
        else:
            local = sentence

        # Explicit markers help character n-grams distinguish the target from
        # other companies or entities mentioned in the same sentence.
        marked = sentence.replace(aspect, f" [A] {aspect} [/A] ")
        full_texts.append(f"{aspect} [S] {marked}")
        local_texts.append(f"{aspect} [C] {local}")
        aspects.append(aspect)

    return full_texts, local_texts, aspects


def fit_char_svm(
    train_texts: list[str], labels: pd.Series, test_texts: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 6),
        min_df=2,
        max_features=250_000,
        sublinear_tf=True,
    )
    train_matrix = vectorizer.fit_transform(train_texts)
    test_matrix = vectorizer.transform(test_texts)
    model = LinearSVC(C=0.7, class_weight="balanced", dual=True, random_state=2026)
    model.fit(train_matrix, labels)
    return model.decision_function(test_matrix), model.classes_


def main() -> None:
    train = pd.read_csv(ROOT / "train.csv").fillna("")
    test = pd.read_csv(ROOT / "test.csv").fillna("")
    sample = pd.read_csv(ROOT / "sample_submission.csv")

    train_full, train_local, train_aspects = make_texts(train)
    test_full, test_local, test_aspects = make_texts(test)

    full_scores, classes = fit_char_svm(train_full, train["label"], test_full)
    local_scores, local_classes = fit_char_svm(train_local, train["label"], test_local)
    aspect_scores, aspect_classes = fit_char_svm(
        train_aspects, train["label"], test_aspects
    )
    if not (np.array_equal(classes, local_classes) and np.array_equal(classes, aspect_classes)):
        raise RuntimeError("The ensemble models produced inconsistent class orders")

    scores = 0.8 * full_scores + 0.2 * local_scores
    seen_sentence = test["sentence"].isin(set(train["sentence"])).to_numpy()
    scores[seen_sentence] = (
        0.6 * scores[seen_sentence] + 0.4 * aspect_scores[seen_sentence]
    )
    predictions = classes[np.argmax(scores, axis=1)]

    prediction_by_id = pd.Series(predictions, index=test["id"]).to_dict()
    submission = sample[["id"]].copy()
    submission["label"] = submission["id"].map(prediction_by_id)

    if len(submission) != len(test) or not submission["id"].is_unique:
        raise ValueError("Submission IDs are not unique or row count does not match test.csv")
    if set(submission["id"]) != set(test["id"]):
        raise ValueError("Submission and test.csv IDs differ")
    if submission["label"].isna().any() or not set(submission["label"]) <= VALID_LABELS:
        raise ValueError("Submission contains a missing or invalid label")

    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    submission.to_csv(output_dir / "submission.csv", index=False)
    print(f"Wrote {len(submission)} predictions to {output_dir / 'submission.csv'}")
    print(submission["label"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()

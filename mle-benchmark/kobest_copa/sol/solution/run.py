#!/usr/bin/env python3
"""Train a local KoBEST COPA ranker and create outputs/submission.csv."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parents[1]
TRAIN_COLUMNS = {
    "id",
    "premise",
    "question",
    "alternative_1",
    "alternative_2",
    "label",
}
TEST_COLUMNS = TRAIN_COLUMNS - {"label"}


def candidate_texts(data: pd.DataFrame, alternative: str) -> list[str]:
    """Mark each part explicitly so candidate plausibility can be learned."""
    return (
        data["question"].str.strip()
        + " 전제 "
        + data["premise"]
        + " 대안 "
        + data[alternative]
    ).tolist()


def neighbor_scores(
    similarities: np.ndarray,
    candidate_labels: np.ndarray,
    neighbors: int = 100,
) -> np.ndarray:
    """Return similarity-weighted correctness among the closest candidates."""
    neighbors = min(neighbors, similarities.shape[1])
    indices = np.argpartition(similarities, -neighbors, axis=1)[:, -neighbors:]
    values = np.take_along_axis(similarities, indices, axis=1)
    weights = np.maximum(values, 1e-8) ** 2
    return (weights * candidate_labels[indices]).sum(axis=1) / weights.sum(axis=1)


def main() -> None:
    train = pd.read_csv(ROOT / "train.csv").fillna("")
    test = pd.read_csv(ROOT / "test.csv").fillna("")
    sample = pd.read_csv(ROOT / "sample_submission.csv")

    if set(train.columns) != TRAIN_COLUMNS or set(test.columns) != TEST_COLUMNS:
        raise ValueError("Unexpected train/test columns")
    if not train["label"].isin([0, 1]).all():
        raise ValueError("Training labels must be 0 or 1")
    if test["id"].duplicated().any() or set(sample["id"]) != set(test["id"]):
        raise ValueError("Test and sample submission IDs do not match")

    train_questions = train["question"].str.strip().to_numpy()
    test_questions = test["question"].str.strip().to_numpy()
    labels = train["label"].astype(int).to_numpy()

    train_texts = candidate_texts(train, "alternative_1") + candidate_texts(
        train, "alternative_2"
    )
    # Correctness labels for alternative 1 followed by alternative 2.
    candidate_labels = np.concatenate([1 - labels, labels])

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=2,
        sublinear_tf=True,
        max_features=200_000,
        dtype=np.float64,
    )
    train_matrix = vectorizer.fit_transform(train_texts)
    test_alt1 = vectorizer.transform(candidate_texts(test, "alternative_1"))
    test_alt2 = vectorizer.transform(candidate_texts(test, "alternative_2"))

    # Retrieve only examples with the same causal direction (cause or effect).
    candidate_questions = np.concatenate([train_questions, train_questions])
    same_question = test_questions[:, None] == candidate_questions[None, :]
    similarity_1 = (test_alt1 @ train_matrix.T).toarray()
    similarity_2 = (test_alt2 @ train_matrix.T).toarray()
    similarity_1[~same_question] = -1.0
    similarity_2[~same_question] = -1.0
    retrieval_margin = neighbor_scores(
        similarity_2, candidate_labels
    ) - neighbor_scores(similarity_1, candidate_labels)

    linear_model = LogisticRegression(
        C=4.0,
        fit_intercept=False,
        solver="liblinear",
        max_iter=1_000,
        random_state=42,
    )
    linear_model.fit(train_matrix, candidate_labels)
    linear_margin = linear_model.decision_function(
        test_alt2
    ) - linear_model.decision_function(test_alt1)

    # Scales and weights were selected using fixed 5-fold training-only CV.
    combined_margin = 0.85 * retrieval_margin / 0.06262 + 0.15 * linear_margin / 0.56249
    predictions = (combined_margin > 0.025).astype(int)

    prediction_by_id = dict(zip(test["id"], predictions, strict=True))
    submission = sample[["id"]].copy()
    submission["label"] = submission["id"].map(prediction_by_id)
    if submission["label"].isna().any() or not submission["label"].isin([0, 1]).all():
        raise ValueError("Submission contains missing or invalid labels")
    submission["label"] = submission["label"].astype(int)

    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "submission.csv"
    submission.to_csv(output_path, index=False)
    print(f"Wrote {len(submission)} predictions to {output_path}")


if __name__ == "__main__":
    main()

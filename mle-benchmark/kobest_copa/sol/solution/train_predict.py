"""Train a pairwise TF-IDF model and create the KoBEST COPA submission."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


SEED = 42
N_SPLITS = 5


def make_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(1, 4),
        min_df=2,
        max_features=150_000,
        sublinear_tf=True,
    )


def fit_candidate_model(
    data: pd.DataFrame, indices: np.ndarray
) -> tuple[TfidfVectorizer, LogisticRegression]:
    """Learn whether a standalone alternative looks like the correct choice."""
    alternatives = pd.concat(
        [
            data.iloc[indices]["alternative_1"],
            data.iloc[indices]["alternative_2"],
        ],
        ignore_index=True,
    )
    labels = data.iloc[indices]["label"].to_numpy()
    candidate_labels = np.concatenate([1 - labels, labels])

    vectorizer = make_vectorizer()
    features = vectorizer.fit_transform(alternatives)
    model = LogisticRegression(
        C=1.5,
        max_iter=1_000,
        solver="liblinear",
        random_state=SEED,
    )
    model.fit(features, candidate_labels)
    return vectorizer, model


def score_rows(
    data: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    model: LogisticRegression,
) -> np.ndarray:
    score_1 = model.decision_function(
        vectorizer.transform(data["alternative_1"])
    )
    score_2 = model.decision_function(
        vectorizer.transform(data["alternative_2"])
    )
    return score_2 - score_1


def best_positive_rate(scores: np.ndarray, labels: np.ndarray) -> float:
    """Calibrate the position prior using only out-of-fold predictions."""
    order = np.argsort(scores)[::-1]
    sorted_labels = labels[order]
    correct = int((labels == 0).sum())
    best_correct = correct
    best_count = 0

    for count, label in enumerate(sorted_labels, start=1):
        correct += 1 if label == 1 else -1
        if correct > best_correct:
            best_correct = correct
            best_count = count
    return best_count / len(labels)


def labels_at_rate(scores: np.ndarray, positive_rate: float) -> np.ndarray:
    count = int(round(len(scores) * positive_rate))
    labels = np.zeros(len(scores), dtype=int)
    if count:
        labels[np.argsort(scores)[-count:]] = 1
    return labels


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    train = pd.read_csv(root / "train.csv").fillna("")
    test = pd.read_csv(root / "test.csv").fillna("")
    y = train["label"].astype(int).to_numpy()

    oof_scores = np.zeros(len(train))
    splitter = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=SEED
    )
    for train_indices, validation_indices in splitter.split(train, y):
        vectorizer, model = fit_candidate_model(train, train_indices)
        oof_scores[validation_indices] = score_rows(
            train.iloc[validation_indices], vectorizer, model
        )

    positive_rate = best_positive_rate(oof_scores, y)
    oof_labels = labels_at_rate(oof_scores, positive_rate)
    oof_accuracy = float((oof_labels == y).mean())

    all_indices = np.arange(len(train))
    vectorizer, model = fit_candidate_model(train, all_indices)
    test_scores = score_rows(test, vectorizer, model)
    predictions = labels_at_rate(test_scores, positive_rate)

    submission = pd.DataFrame({"id": test["id"], "label": predictions})
    output_path = root / "outputs" / "submission.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(
        f"OOF accuracy={oof_accuracy:.4f}, positive_rate={positive_rate:.4f}, "
        f"wrote {len(submission)} rows to {output_path}"
    )


if __name__ == "__main__":
    main()

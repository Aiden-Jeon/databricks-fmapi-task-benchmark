"""Train the UnSmile multilabel model and create outputs/submission.csv."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
N_LABELS = 10
N_FOLDS = 5
SEED = 2026


def parse_labels(values: pd.Series) -> np.ndarray:
    if not values.str.fullmatch(r"[01]{10}").all():
        raise ValueError("Every training label must be a 10-character binary string")
    return np.asarray([[int(bit) for bit in value] for value in values], dtype=np.int8)


def make_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(1, 5),
        min_df=2,
        max_df=0.995,
        sublinear_tf=True,
        max_features=200_000,
        dtype=np.float32,
    )


def optimal_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[int(np.argmax(f1))])


def main() -> None:
    train = pd.read_csv(ROOT / "train.csv", dtype=str)
    test = pd.read_csv(ROOT / "test.csv", dtype=str)
    sample = pd.read_csv(ROOT / "sample_submission.csv", dtype=str)

    if list(train.columns) != ["id", "sentence", "labels"]:
        raise ValueError("Unexpected train.csv columns")
    if list(test.columns) != ["id", "sentence"]:
        raise ValueError("Unexpected test.csv columns")
    if list(sample.columns) != ["id", "labels"]:
        raise ValueError("Unexpected sample_submission.csv columns")
    if test["id"].duplicated().any() or set(test["id"]) != set(sample["id"]):
        raise ValueError("Test IDs must be unique and match sample_submission.csv")

    train_text = train["sentence"].fillna("").to_numpy()
    test_text = test["sentence"].fillna("").to_numpy()
    labels = parse_labels(train["labels"])

    # Most examples have one label. The first positive label is a stable proxy for
    # stratifying the small number of multilabel combinations.
    stratification_label = labels.argmax(axis=1)
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_scores = np.zeros(labels.shape, dtype=np.float32)
    test_scores = np.zeros((len(test), N_LABELS), dtype=np.float32)

    for fold, (fit_indices, valid_indices) in enumerate(
        splitter.split(train_text, stratification_label), start=1
    ):
        vectorizer = make_vectorizer()
        fit_features = vectorizer.fit_transform(train_text[fit_indices])
        valid_features = vectorizer.transform(train_text[valid_indices])
        fold_test_features = vectorizer.transform(test_text)

        for column in range(N_LABELS):
            classifier = LinearSVC(C=0.25, dual="auto", max_iter=5000, random_state=SEED)
            classifier.fit(fit_features, labels[fit_indices, column])
            oof_scores[valid_indices, column] = classifier.decision_function(valid_features)
            test_scores[:, column] += (
                classifier.decision_function(fold_test_features) / N_FOLDS
            )
        print(f"Finished fold {fold}/{N_FOLDS} with {fit_features.shape[1]} features")

    thresholds = np.asarray(
        [optimal_threshold(labels[:, column], oof_scores[:, column]) for column in range(N_LABELS)]
    )
    oof_predictions = oof_scores >= thresholds
    test_predictions = test_scores >= thresholds
    print(f"OOF macro F1: {f1_score(labels, oof_predictions, average='macro'):.6f}")
    print("Thresholds:", np.round(thresholds, 4).tolist())

    encoded_predictions = [
        "".join(row.astype(np.uint8).astype(str).tolist()) for row in test_predictions
    ]
    submission = pd.DataFrame({"id": test["id"], "labels": encoded_predictions})

    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "submission.csv"
    submission.to_csv(output_path, index=False)

    written = pd.read_csv(output_path, dtype=str)
    if (
        list(written.columns) != ["id", "labels"]
        or len(written) != len(test)
        or written["id"].duplicated().any()
        or written["id"].tolist() != test["id"].tolist()
        or not written["labels"].str.fullmatch(r"[01]{10}").all()
    ):
        raise RuntimeError("Generated submission failed format validation")
    print(f"Wrote {len(written)} predictions to {output_path}")


if __name__ == "__main__":
    main()

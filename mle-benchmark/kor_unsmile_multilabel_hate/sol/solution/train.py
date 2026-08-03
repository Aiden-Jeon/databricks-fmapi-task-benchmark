"""Train the final UnSmile classifier and create outputs/submission.csv."""

import gc
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC


SEED = 20260801
N_FOLDS = 5
N_LABELS = 10
MODEL_WEIGHT = 0.5
ROOT = Path(__file__).resolve().parents[1]

FEATURE_CONFIGS = (
    {"analyzer": "char", "ngram_range": (1, 5)},
    {"analyzer": "char_wb", "ngram_range": (2, 5)},
)


def labels_to_array(labels: pd.Series) -> np.ndarray:
    if not labels.str.fullmatch(r"[01]{10}").all():
        raise ValueError("Every training label must be a 10-bit binary string")
    return np.array([[int(bit) for bit in value] for value in labels], dtype=np.int8)


def make_vectorizer(config: dict) -> TfidfVectorizer:
    return TfidfVectorizer(
        **config,
        min_df=2,
        max_features=300_000,
        sublinear_tf=True,
        dtype=np.float32,
    )


def make_model() -> OneVsRestClassifier:
    return OneVsRestClassifier(
        LinearSVC(
            C=1.2,
            class_weight="balanced",
            dual="auto",
            max_iter=10_000,
            random_state=SEED,
        ),
        n_jobs=-1,
    )


def optimal_positive_count(y_true: np.ndarray, scores: np.ndarray) -> int:
    """Return the top-k cutoff that maximizes F1 without a coarse threshold grid."""
    order = np.argsort(-scores, kind="stable")
    true_positives = np.cumsum(y_true[order])
    predicted_counts = np.arange(1, len(y_true) + 1)
    f1_values = 2 * true_positives / (predicted_counts + y_true.sum())
    return int(np.argmax(f1_values)) + 1


def oof_scores(text: pd.Series, y: np.ndarray, strata: pd.Series) -> np.ndarray:
    scores = np.zeros((len(text), N_LABELS), dtype=np.float64)
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for config_index, config in enumerate(FEATURE_CONFIGS, start=1):
        print(f"Generating OOF scores for feature model {config_index}/2: {config}")
        for fold, (train_index, valid_index) in enumerate(splitter.split(text, strata), start=1):
            vectorizer = make_vectorizer(config)
            x_train = vectorizer.fit_transform(text.iloc[train_index])
            x_valid = vectorizer.transform(text.iloc[valid_index])
            model = make_model()
            model.fit(x_train, y[train_index])
            scores[valid_index] += MODEL_WEIGHT * model.decision_function(x_valid)
            print(f"  fold {fold}/{N_FOLDS}: {x_train.shape[1]:,} features")
            del vectorizer, x_train, x_valid, model
            gc.collect()

    return scores


def full_train_scores(train_text: pd.Series, y: np.ndarray, test_text: pd.Series) -> np.ndarray:
    scores = np.zeros((len(test_text), N_LABELS), dtype=np.float64)
    for config_index, config in enumerate(FEATURE_CONFIGS, start=1):
        vectorizer = make_vectorizer(config)
        x_train = vectorizer.fit_transform(train_text)
        x_test = vectorizer.transform(test_text)
        model = make_model()
        model.fit(x_train, y)
        scores += MODEL_WEIGHT * model.decision_function(x_test)
        print(f"Full model {config_index}/2 trained with {x_train.shape[1]:,} features")
        del vectorizer, x_train, x_test, model
        gc.collect()
    return scores


def validate_submission(submission: pd.DataFrame, test: pd.DataFrame) -> None:
    if list(submission.columns) != ["id", "labels"]:
        raise ValueError("Submission columns must be exactly: id, labels")
    if len(submission) != len(test) or submission["id"].duplicated().any():
        raise ValueError("Submission must contain every test ID exactly once")
    if set(submission["id"]) != set(test["id"]):
        raise ValueError("Submission IDs do not match test IDs")
    if not submission["labels"].str.fullmatch(r"[01]{10}").all():
        raise ValueError("Every prediction must be a 10-bit binary string")


def main() -> None:
    train = pd.read_csv(ROOT / "train.csv", dtype=str)
    test = pd.read_csv(ROOT / "test.csv", dtype=str)
    y = labels_to_array(train["labels"])

    # Preserve common label combinations across folds; pool combinations too rare
    # for five-way stratification into one fallback stratum.
    combination_counts = train["labels"].value_counts()
    strata = train["labels"].where(train["labels"].map(combination_counts) >= N_FOLDS, "rare")
    train_oof_scores = oof_scores(train["sentence"], y, strata)

    positive_counts = np.array(
        [optimal_positive_count(y[:, label], train_oof_scores[:, label]) for label in range(N_LABELS)]
    )
    oof_prediction = np.zeros_like(y)
    for label, count in enumerate(positive_counts):
        top_indices = np.argsort(-train_oof_scores[:, label], kind="stable")[:count]
        oof_prediction[top_indices, label] = 1
    print(f"OOF macro F1: {f1_score(y, oof_prediction, average='macro'):.6f}")
    print("OOF selected positive counts:", positive_counts.tolist())

    test_scores = full_train_scores(train["sentence"], y, test["sentence"])
    test_prediction = np.zeros((len(test), N_LABELS), dtype=np.int8)
    # Test is an IID 20% split, so transfer the OOF-optimal predicted prevalence.
    test_positive_counts = np.rint(positive_counts / len(train) * len(test)).astype(int)
    for label, count in enumerate(test_positive_counts):
        top_indices = np.argsort(-test_scores[:, label], kind="stable")[:count]
        test_prediction[top_indices, label] = 1

    label_strings = ["".join(row.astype(str)) for row in test_prediction]
    submission = pd.DataFrame({"id": test["id"], "labels": label_strings})
    validate_submission(submission, test)
    output_path = ROOT / "outputs" / "submission.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print("Test selected positive counts:", test_prediction.sum(axis=0).tolist())
    print(f"Wrote {len(submission):,} predictions to {output_path}")


if __name__ == "__main__":
    main()

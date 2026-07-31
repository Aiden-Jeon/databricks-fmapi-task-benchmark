from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "train.csv"
TEST_PATH = ROOT / "test.csv"
SAMPLE_PATH = ROOT / "sample_submission.csv"
OUTPUT_PATH = ROOT / "outputs" / "submission.csv"


def build_model(analyzer: str, ngram_range: tuple[int, int]):
    return make_pipeline(
        TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            min_df=1,
            sublinear_tf=True,
        ),
        LinearSVC(C=1.0, dual="auto", random_state=20260731),
    )


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    sample = pd.read_csv(SAMPLE_PATH)

    if train[["id", "sentence", "label"]].isna().any().any():
        raise ValueError("train.csv contains missing values")
    if test[["id", "sentence"]].isna().any().any():
        raise ValueError("test.csv contains missing values")
    if train["id"].duplicated().any() or test["id"].duplicated().any():
        raise ValueError("Input IDs must be unique")
    if set(test["id"]) != set(sample["id"]):
        raise ValueError("test.csv and sample_submission.csv IDs differ")

    model_specs = [
        ("char", (2, 5), 0.25),
        ("char", (2, 4), 0.35),
        ("char_wb", (2, 6), 0.40),
    ]
    decision = np.zeros(len(test), dtype=np.float64)
    for analyzer, ngram_range, weight in model_specs:
        model = build_model(analyzer, ngram_range)
        model.fit(train["sentence"], train["label"])
        decision += weight * model.decision_function(test["sentence"])

    predictions = pd.DataFrame(
        {"id": test["id"], "label": (decision >= 0.0).astype(np.int8)}
    )
    submission = sample[["id"]].merge(
        predictions, on="id", how="left", validate="one_to_one", sort=False
    )

    if submission["label"].isna().any() or len(submission) != len(test):
        raise ValueError("Submission does not contain every test ID exactly once")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(submission)} predictions to {OUTPUT_PATH}")
    print(submission["label"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()

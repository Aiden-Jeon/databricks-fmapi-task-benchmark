"""Train the final sentiment classifier and create outputs/submission.csv."""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]


def build_model() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(1, 5),
                    min_df=2,
                    sublinear_tf=True,
                    max_features=180_000,
                ),
            ),
            ("classifier", LinearSVC(C=0.5, dual=True)),
        ]
    )


def main() -> None:
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    sample = pd.read_csv(ROOT / "sample_submission.csv")

    if list(train.columns) != ["id", "sentence", "label"]:
        raise ValueError("Unexpected train.csv columns")
    if list(test.columns) != ["id", "sentence"]:
        raise ValueError("Unexpected test.csv columns")
    if list(sample.columns) != ["id", "label"]:
        raise ValueError("Unexpected sample_submission.csv columns")
    if test["id"].duplicated().any() or set(test["id"]) != set(sample["id"]):
        raise ValueError("Test IDs must be unique and match sample_submission.csv")

    model = build_model()
    model.fit(train["sentence"], train["label"])
    predictions = model.predict(test["sentence"]).astype(int)

    submission = pd.DataFrame({"id": test["id"], "label": predictions})
    output_path = ROOT / "outputs" / "submission.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Wrote {len(submission)} predictions to {output_path}")
    print(f"Predicted label counts: {submission['label'].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()

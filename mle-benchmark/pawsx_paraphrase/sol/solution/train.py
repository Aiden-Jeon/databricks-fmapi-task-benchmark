"""Train the PAWS-X classifier and create outputs/submission.csv."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from experiment import add_graph_features, make_features


SEED = 2026


def main():
    root = Path(__file__).resolve().parents[1]
    train = pd.read_csv(root / "train.csv")
    test = pd.read_csv(root / "test.csv")
    sample = pd.read_csv(root / "sample_submission.csv")

    print("Extracting train features...")
    x_train = make_features(train)
    print("Extracting test features...")
    x_test = make_features(test)

    extra = ExtraTreesClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        max_features=0.9,
        class_weight="balanced",
        n_jobs=-1,
        random_state=SEED,
    )
    forest = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        max_features=0.9,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=SEED,
    )
    print("Training ensemble...")
    extra.fit(x_train, train.label)
    forest.fit(x_train, train.label)
    probability = 0.25 * extra.predict_proba(x_test)[:, 1] + 0.75 * forest.predict_proba(x_test)[:, 1]
    prediction = (probability >= 0.52).astype(np.int8)

    # Exact training pairs and positive transitive links are high-confidence signals.
    graph = add_graph_features(train, test)
    exact = graph[:, 0] == 1
    connected = (graph[:, 2] == 1) & (graph[:, 3] == 1)
    prediction[exact] = (graph[exact, 1] >= 0.5).astype(np.int8)
    prediction[connected] = 1
    nonempty_equal = test.sentence1.notna().to_numpy() & (
        test.sentence1.fillna("").to_numpy() == test.sentence2.fillna("").to_numpy()
    )
    prediction[nonempty_equal] = 1

    submission = pd.DataFrame({"id": test.id, "label": prediction})
    if list(submission.columns) != list(sample.columns):
        raise ValueError("Submission columns do not match sample_submission.csv")
    if len(submission) != len(test) or submission.id.nunique() != len(test):
        raise ValueError("Submission must contain every test id exactly once")
    if submission.id.tolist() != sample.id.tolist():
        raise ValueError("Submission id order does not match sample_submission.csv")
    if not set(submission.label.unique()).issubset({0, 1}):
        raise ValueError("Predictions must be binary")

    output_dir = root / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "submission.csv"
    submission.to_csv(output_path, index=False)
    print(f"Wrote {output_path} ({len(submission)} rows, positive rate={prediction.mean():.4f})")


if __name__ == "__main__":
    main()

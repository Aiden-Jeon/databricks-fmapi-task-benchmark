"""Train the final ranker and write outputs/submission.csv."""

from pathlib import Path

import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from experiment import flatten, handcrafted, similarities


ROOT = Path(__file__).resolve().parents[1]
SEED = 2026


def main():
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    sample = pd.read_csv(ROOT / "sample_submission.csv")

    train_contexts, train_endings, train_positions, train_targets = flatten(train)
    test_contexts, test_endings, test_positions, _ = flatten(test)

    char = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=2,
        max_features=120_000,
        sublinear_tf=True,
    )
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=80_000,
        sublinear_tf=True,
    )
    train_char = char.fit_transform(train_endings)
    test_char = char.transform(test_endings)
    train_word = word.fit_transform(train_endings)
    test_word = word.transform(test_endings)

    train_x = sparse.hstack([
        train_char,
        train_word * 0.3,
        handcrafted(train_contexts, train_endings, train_positions),
        similarities(char, train_contexts, train_endings),
        similarities(word, train_contexts, train_endings),
    ], format="csr")
    test_x = sparse.hstack([
        test_char,
        test_word * 0.3,
        handcrafted(test_contexts, test_endings, test_positions),
        similarities(char, test_contexts, test_endings),
        similarities(word, test_contexts, test_endings),
    ], format="csr")

    model = LogisticRegression(
        C=1.0,
        max_iter=1_000,
        class_weight="balanced",
        solver="liblinear",
        random_state=SEED,
    )
    model.fit(train_x, train_targets)
    predictions = model.decision_function(test_x).reshape(-1, 4).argmax(axis=1)

    submission = pd.DataFrame({"id": test["id"], "label": predictions})
    assert list(submission.columns) == list(sample.columns) == ["id", "label"]
    assert submission["id"].is_unique
    assert submission["id"].tolist() == sample["id"].tolist()
    assert set(submission["id"]) == set(test["id"])
    assert submission["label"].isin(range(4)).all()

    output_path = ROOT / "outputs" / "submission.csv"
    output_path.parent.mkdir(exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Wrote {len(submission)} predictions to {output_path}")
    print(submission["label"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()

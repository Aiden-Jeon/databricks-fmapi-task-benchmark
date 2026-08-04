#!/usr/bin/env python3
"""Train a KorNLI classifier and create outputs/submission.csv."""

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
LABELS = {"entailment", "neutral", "contradiction"}
NEGATION_CUES = ("않", "아니", "없", "못", "결코", "아무도", "절대")


def relational_features(data: pd.DataFrame) -> np.ndarray:
    """Extract language-independent and Korean-aware pair relation features."""
    rows = []
    for premise, hypothesis in zip(data["sentence1"], data["sentence2"]):
        p_words, h_words = premise.split(), hypothesis.split()
        p_word_set, h_word_set = set(p_words), set(h_words)
        p_chars, h_chars = set(premise), set(hypothesis)
        p_bigrams = {premise[i : i + 2] for i in range(len(premise) - 1)}
        h_bigrams = {hypothesis[i : i + 2] for i in range(len(hypothesis) - 1)}
        p_numbers = {w for w in p_words if any(c.isdigit() for c in w)}
        h_numbers = {w for w in h_words if any(c.isdigit() for c in w)}
        common_words = len(p_word_set & h_word_set)
        p_neg = sum(cue in premise for cue in NEGATION_CUES)
        h_neg = sum(cue in hypothesis for cue in NEGATION_CUES)

        rows.append(
            [
                len(premise),
                len(hypothesis),
                len(hypothesis) / (len(premise) + 1),
                abs(len(premise) - len(hypothesis)),
                len(p_words),
                len(h_words),
                common_words / (len(h_word_set) + 1e-3),
                common_words / (len(p_word_set) + 1e-3),
                common_words / (len(p_word_set | h_word_set) + 1e-3),
                len(p_chars & h_chars) / (len(h_chars) + 1e-3),
                len(p_bigrams & h_bigrams) / (len(h_bigrams) + 1e-3),
                hypothesis in premise,
                p_neg,
                h_neg,
                p_neg - h_neg,
                bool(p_numbers),
                bool(h_numbers),
                bool(p_numbers & h_numbers),
                bool(p_numbers and h_numbers and p_numbers != h_numbers),
                premise.endswith("?"),
                hypothesis.endswith("?"),
            ]
        )
    return np.asarray(rows, dtype=np.float64)


def observed_labels_by_premise(train: pd.DataFrame) -> dict[str, set[str]]:
    labels = defaultdict(set)
    for premise, label in zip(train["sentence1"], train["label"]):
        labels[premise].add(label)
    return labels


def exact_pair_labels(train: pd.DataFrame) -> dict[tuple[str, str], str]:
    counts = defaultdict(Counter)
    for premise, hypothesis, label in train[
        ["sentence1", "sentence2", "label"]
    ].itertuples(index=False, name=None):
        counts[(premise, hypothesis)][label] += 1
    return {pair: labels.most_common(1)[0][0] for pair, labels in counts.items()}


def main() -> None:
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    sample = pd.read_csv(ROOT / "sample_submission.csv")

    if set(train["label"]) != LABELS:
        raise ValueError("Unexpected labels in train.csv")
    if test["id"].duplicated().any() or set(test["id"]) != set(sample["id"]):
        raise ValueError("test.csv ids do not match sample_submission.csv")

    # Hypothesis wording carries strong NLI annotation cues. Character n-grams
    # also tolerate Korean particles and translated phrasing without a tokenizer.
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=2,
        max_df=0.995,
        max_features=300_000,
        sublinear_tf=True,
        dtype=np.float64,
    )
    train_text = vectorizer.fit_transform(train["sentence2"])
    test_text = vectorizer.transform(test["sentence2"])

    scaler = StandardScaler()
    train_rel = scaler.fit_transform(relational_features(train))
    test_rel = scaler.transform(relational_features(test))
    x_train = hstack([train_text, csr_matrix(train_rel * 4.0)], format="csr")
    x_test = hstack([test_text, csr_matrix(test_rel * 4.0)], format="csr")

    model = LinearSVC(C=0.5, dual=True, max_iter=1500, random_state=2026)
    model.fit(x_train, train["label"])
    scores = model.decision_function(x_test)

    # KorNLI contains repeated premises whose distinct hypotheses nearly always
    # cover distinct labels. Use that learned training-set pattern, while exact
    # repeated sentence pairs retain their directly observed label.
    premise_labels = observed_labels_by_premise(train)
    pair_labels = exact_pair_labels(train)
    exact_predictions = {}
    class_index = {label: i for i, label in enumerate(model.classes_)}
    for row_index, row in enumerate(test.itertuples(index=False)):
        pair = (row.sentence1, row.sentence2)
        if pair in pair_labels:
            exact_predictions[row_index] = pair_labels[pair]
            continue
        for label in premise_labels.get(row.sentence1, ()):
            scores[row_index, class_index[label]] -= 10.0

    predictions = model.classes_[scores.argmax(axis=1)]
    for row_index, label in exact_predictions.items():
        predictions[row_index] = label

    prediction_by_id = dict(zip(test["id"], predictions))
    submission = sample[["id"]].copy()
    submission["label"] = submission["id"].map(prediction_by_id)
    if submission["label"].isna().any() or not set(submission["label"]) <= LABELS:
        raise ValueError("Invalid predictions generated")

    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    submission.to_csv(output_dir / "submission.csv", index=False)
    print(f"Wrote {len(submission)} predictions to {output_dir / 'submission.csv'}")


if __name__ == "__main__":
    main()

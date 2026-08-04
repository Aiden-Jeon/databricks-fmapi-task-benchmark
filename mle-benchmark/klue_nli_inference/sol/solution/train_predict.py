#!/usr/bin/env python3
"""Train an offline KLUE-NLI model and create the submission."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


LABELS = np.array(["contradiction", "entailment", "neutral"])
TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9]+")
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
NEGATIONS = ("않", "아니", "없", "못", "금지", "반대", "불가", "제외", "실패")


def pair_features(frame: pd.DataFrame) -> np.ndarray:
    """Small dense features describing lexical relations within each pair."""
    rows = []
    for premise, hypothesis in zip(frame["premise"], frame["hypothesis"]):
        p_tokens = set(TOKEN_RE.findall(premise.lower()))
        h_tokens = set(TOKEN_RE.findall(hypothesis.lower()))
        common = len(p_tokens & h_tokens)
        p_num = set(NUMBER_RE.findall(premise))
        h_num = set(NUMBER_RE.findall(hypothesis))
        p_neg = sum(term in premise for term in NEGATIONS)
        h_neg = sum(term in hypothesis for term in NEGATIONS)
        rows.append(
            [
                len(premise),
                len(hypothesis),
                len(hypothesis) / max(len(premise), 1),
                common / max(len(h_tokens), 1),
                common / max(len(p_tokens | h_tokens), 1),
                float(hypothesis in premise),
                p_neg,
                h_neg,
                float(bool(p_neg) != bool(h_neg)),
                len(p_num & h_num) / max(len(h_num), 1),
                float(bool(h_num) and not h_num.issubset(p_num)),
            ]
        )
    return np.asarray(rows, dtype=np.float64)


class NLIModel:
    def __init__(self) -> None:
        self.word = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=2, max_df=0.995,
            sublinear_tf=True, max_features=180_000,
        )
        self.char = TfidfVectorizer(
            analyzer="char", ngram_range=(2, 5), min_df=2, max_df=0.995,
            sublinear_tf=True, max_features=220_000,
        )
        self.scaler = StandardScaler()
        self.classifier = LinearSVC(
            C=0.06, class_weight="balanced", dual=True, max_iter=10_000, tol=1e-3
        )

    @staticmethod
    def text(frame: pd.DataFrame) -> pd.Series:
        # Hypotheses carry useful annotation patterns; premise topic words mostly
        # add noise. Pair relations are represented by the dense features below.
        return frame["hypothesis"]

    def fit(self, frame: pd.DataFrame) -> "NLIModel":
        text = self.text(frame)
        word = self.word.fit_transform(text)
        char = self.char.fit_transform(text)
        dense = self.scaler.fit_transform(pair_features(frame))
        matrix = hstack([word, char, csr_matrix(dense)], format="csr")
        self.classifier.fit(matrix, frame["label"])
        return self

    def scores(self, frame: pd.DataFrame) -> np.ndarray:
        text = self.text(frame)
        dense = self.scaler.transform(pair_features(frame))
        matrix = hstack(
            [self.word.transform(text), self.char.transform(text), csr_matrix(dense)],
            format="csr",
        )
        raw = self.classifier.decision_function(matrix)
        order = [list(self.classifier.classes_).index(label) for label in LABELS]
        return raw[:, order]


def group_constrained_predictions(
    train: pd.DataFrame, test: pd.DataFrame, scores: np.ndarray
) -> np.ndarray:
    """Use the usual one-of-each-label structure for three-row premise groups."""
    predictions = LABELS[scores.argmax(axis=1)].copy()
    train_by_premise = train.groupby("premise")["label"].apply(list).to_dict()

    for premise, indices in test.groupby("premise", sort=False).groups.items():
        indices = np.asarray(list(indices), dtype=int)
        known = train_by_premise.get(premise, [])
        total_size = len(known) + len(indices)
        # Nearly all source groups have three rows. Only enforce the constraint
        # when observed labels are unique, so noisy duplicate-label groups fall
        # back to independent model predictions.
        if total_size != 3 or len(set(known)) != len(known):
            continue
        available = [i for i, label in enumerate(LABELS) if label not in known]
        if len(available) != len(indices):
            continue
        cost = -scores[np.ix_(indices, available)]
        row_ind, col_ind = linear_sum_assignment(cost)
        for row, col in zip(row_ind, col_ind):
            predictions[indices[row]] = LABELS[available[col]]
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    model = NLIModel().fit(train)
    scores = model.scores(test)
    labels = group_constrained_predictions(train, test, scores)

    output = args.output or args.data_dir / "outputs" / "submission.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": test["id"], "label": labels}).to_csv(output, index=False)
    print(f"Wrote {len(test):,} predictions to {output}")


if __name__ == "__main__":
    main()

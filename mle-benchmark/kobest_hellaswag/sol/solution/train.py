#!/usr/bin/env python3
"""Train a choice-ranking model and create the KoBEST HellaSwag submission."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold


N_CHOICES = 4
TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9]+")


def tokens(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text))


def last_sentence(text: str) -> str:
    parts = [part.strip() for part in re.split(r"[.!?]", text) if part.strip()]
    return parts[-1] if parts else text


def char_ngrams(text: str, n: int) -> set[str]:
    compact = re.sub(r"\s+", "", text)
    return {compact[i : i + n] for i in range(max(0, len(compact) - n + 1))}


def dense_features(frame: pd.DataFrame) -> np.ndarray:
    """Create candidate-level features without using labels."""
    rows: list[list[float]] = []
    for record in frame.itertuples(index=False):
        context = str(record.context)
        context_tokens = tokens(context)
        last_tokens = tokens(last_sentence(context))
        endings = [str(getattr(record, f"ending_{i}")) for i in range(1, 5)]
        ending_tokens = [tokens(text) for text in endings]
        lengths = np.asarray([len(text) for text in endings], dtype=float)
        length_mean = lengths.mean()
        length_std = lengths.std() + 1.0
        context_grams = {n: char_ngrams(context, n) for n in (1, 2, 3)}
        last_text = last_sentence(context)
        last_grams = {n: char_ngrams(last_text, n) for n in (1, 2, 3)}
        ending_grams = [{n: char_ngrams(text, n) for n in (1, 2, 3)} for text in endings]
        other_grams = [
            {n: set().union(*(ending_grams[j][n] for j in range(4) if j != i)) for n in (1, 2, 3)}
            for i in range(4)
        ]
        char_overlap = np.asarray(
            [
                len(ending_grams[i][2] & context_grams[2]) / (len(ending_grams[i][2]) + 1.0)
                for i in range(4)
            ]
        )
        last_char_overlap = np.asarray(
            [
                len(ending_grams[i][2] & last_grams[2]) / (len(ending_grams[i][2]) + 1.0)
                for i in range(4)
            ]
        )

        for i, (ending, current_tokens) in enumerate(zip(endings, ending_tokens)):
            other_tokens = set().union(*(ending_tokens[j] for j in range(4) if j != i))
            context_overlap = current_tokens & context_tokens
            last_overlap = current_tokens & last_tokens
            rows.append(
                [
                    1.0 if i == 0 else 0.0,
                    1.0 if i == 1 else 0.0,
                    1.0 if i == 2 else 0.0,
                    1.0 if i == 3 else 0.0,
                    len(ending) / 50.0,
                    len(current_tokens) / 12.0,
                    (len(ending) - length_mean) / length_std,
                    float(np.argsort(np.argsort(lengths))[i]) / 3.0,
                    len(context_overlap) / (len(current_tokens) + 1.0),
                    len(context_overlap) / (len(context_tokens | current_tokens) + 1.0),
                    len(last_overlap) / (len(current_tokens) + 1.0),
                    len(last_overlap) / (len(last_tokens | current_tokens) + 1.0),
                    len(current_tokens & other_tokens) / (len(current_tokens) + 1.0),
                    *(len(ending_grams[i][n] & context_grams[n]) / (len(ending_grams[i][n]) + 1.0) for n in (1, 2, 3)),
                    *(len(ending_grams[i][n] & last_grams[n]) / (len(ending_grams[i][n]) + 1.0) for n in (1, 2, 3)),
                    *(len(ending_grams[i][n] & context_grams[n]) / (len(ending_grams[i][n] | context_grams[n]) + 1.0) for n in (1, 2, 3)),
                    *(len(ending_grams[i][n] & last_grams[n]) / (len(ending_grams[i][n] | last_grams[n]) + 1.0) for n in (1, 2, 3)),
                    *(len(ending_grams[i][n] & other_grams[i][n]) / (len(ending_grams[i][n]) + 1.0) for n in (1, 2, 3)),
                    float(np.argsort(np.argsort(char_overlap))[i]) / 3.0,
                    float(np.argsort(np.argsort(last_char_overlap))[i]) / 3.0,
                    ending.count(",") / 3.0,
                    ending.count(".") / 2.0,
                    float(any(ending.startswith(x) for x in ("그리고", "그러나", "그래서", "그 후"))),
                    float(any(x in ending for x in ("다시", "마저", "계속", "이윽고", "결국"))),
                ]
            )
    return np.asarray(rows, dtype=np.float64)


def pair_texts(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    endings: list[str] = []
    transitions: list[str] = []
    for record in frame.itertuples(index=False):
        context_tail = last_sentence(str(record.context))[-100:]
        for i in range(1, N_CHOICES + 1):
            ending = str(getattr(record, f"ending_{i}"))
            endings.append(ending)
            transitions.append(context_tail + " " + ending)
    return endings, transitions


class ChoiceRanker:
    def __init__(self, c: float = 0.35):
        self.c = c
        self.word = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=2, max_df=0.995,
            sublinear_tf=True, max_features=50_000,
        )
        self.char = TfidfVectorizer(
            analyzer="char", ngram_range=(2, 5), min_df=2, max_df=0.995,
            sublinear_tf=True, max_features=100_000,
        )
        self.transition = TfidfVectorizer(
            analyzer="char", ngram_range=(2, 5), min_df=2, max_df=0.995,
            sublinear_tf=True, max_features=100_000,
        )
        self.model = LogisticRegression(
            C=c, solver="liblinear", max_iter=1000, random_state=2026,
        )

    def _fit_features(self, frame: pd.DataFrame):
        endings, transitions = pair_texts(frame)
        return hstack(
            [
                self.word.fit_transform(endings),
                self.char.fit_transform(endings),
                self.transition.fit_transform(transitions),
                csr_matrix(dense_features(frame)),
            ],
            format="csr",
        )

    def _transform_features(self, frame: pd.DataFrame):
        endings, transitions = pair_texts(frame)
        return hstack(
            [
                self.word.transform(endings),
                self.char.transform(endings),
                self.transition.transform(transitions),
                csr_matrix(dense_features(frame)),
            ],
            format="csr",
        )

    def fit(self, frame: pd.DataFrame, labels: np.ndarray) -> "ChoiceRanker":
        features = self._fit_features(frame)
        positive_differences = []
        for row, correct in enumerate(labels):
            correct_row = row * N_CHOICES + correct
            for choice in range(N_CHOICES):
                if choice != correct:
                    positive_differences.append(features[correct_row] - features[row * N_CHOICES + choice])
        positive = vstack(positive_differences, format="csr")
        pairwise = vstack([positive, -positive], format="csr")
        targets = np.concatenate([np.ones(positive.shape[0]), np.zeros(positive.shape[0])])
        self.model.fit(pairwise, targets)
        return self

    def scores(self, frame: pd.DataFrame) -> np.ndarray:
        scores = self.model.decision_function(self._transform_features(frame))
        return scores.reshape(-1, N_CHOICES)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.scores(frame).argmax(axis=1)


def cross_validate(train: pd.DataFrame, c: float, folds: int) -> float:
    labels = train["label"].to_numpy(dtype=int)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=2026)
    predictions = np.empty(len(train), dtype=int)
    for fold, (fit_idx, valid_idx) in enumerate(splitter.split(train, labels), start=1):
        ranker = ChoiceRanker(c=c).fit(train.iloc[fit_idx], labels[fit_idx])
        predictions[valid_idx] = ranker.predict(train.iloc[valid_idx])
        fold_score = accuracy_score(labels[valid_idx], predictions[valid_idx])
        print(f"fold {fold}: accuracy={fold_score:.5f}")
    score = accuracy_score(labels, predictions)
    print(f"OOF accuracy={score:.5f}")
    return score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--cv", action="store_true", help="run 5-fold cross-validation before fitting")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--c", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    output = args.output or args.data_dir / "outputs" / "submission.csv"

    if args.cv:
        cross_validate(train, args.c, args.folds)

    labels = train["label"].to_numpy(dtype=int)
    ranker = ChoiceRanker(c=args.c).fit(train, labels)
    predictions = ranker.predict(test)
    submission = pd.DataFrame({"id": test["id"], "label": predictions})
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    print(f"wrote {len(submission)} predictions to {output}")
    print(submission["label"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()

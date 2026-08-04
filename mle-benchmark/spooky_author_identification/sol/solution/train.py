#!/usr/bin/env python3
"""Train a reproducible authorship classifier and create the submission."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split


SEED = 2026
CLASSES = np.array(["EAP", "HPL", "MWS"])


def make_vectorizers() -> tuple[TfidfVectorizer, TfidfVectorizer]:
    word = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=100_000,
        sublinear_tf=True,
        dtype=np.float64,
    )
    char = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=160_000,
        sublinear_tf=True,
        dtype=np.float64,
    )
    return word, char


def make_model(c: float) -> LogisticRegression:
    return LogisticRegression(
        C=c,
        solver="liblinear",
        multi_class="ovr",
        max_iter=500,
        random_state=SEED,
    )


def normalized_geometric_blend(
    word_probability: np.ndarray,
    char_probability: np.ndarray,
    word_weight: float,
) -> np.ndarray:
    """Blend in log-probability space and normalize every row."""
    eps = 1e-15
    scores = word_weight * np.log(np.clip(word_probability, eps, 1.0))
    scores += (1.0 - word_weight) * np.log(np.clip(char_probability, eps, 1.0))
    return softmax(scores, axis=1)


def validate(train: pd.DataFrame) -> None:
    fit_idx, valid_idx = train_test_split(
        np.arange(len(train)),
        test_size=0.25,
        random_state=SEED,
        stratify=train["author"],
    )
    fit_text = train.loc[fit_idx, "text"]
    valid_text = train.loc[valid_idx, "text"]
    y_fit = train.loc[fit_idx, "author"]
    y_valid = train.loc[valid_idx, "author"]

    word, char = make_vectorizers()
    xw_fit = word.fit_transform(fit_text)
    xw_valid = word.transform(valid_text)
    xc_fit = char.fit_transform(fit_text)
    xc_valid = char.transform(valid_text)
    print(f"word features={xw_fit.shape[1]:,}, char features={xc_fit.shape[1]:,}")

    candidates = (13.0, 18.0, 25.0, 35.0)
    word_predictions: dict[float, np.ndarray] = {}
    char_predictions: dict[float, np.ndarray] = {}
    for c in candidates:
        word_model = make_model(c).fit(xw_fit, y_fit)
        char_model = make_model(c).fit(xc_fit, y_fit)
        word_predictions[c] = word_model.predict_proba(xw_valid)
        char_predictions[c] = char_model.predict_proba(xc_valid)
        print(
            f"C={c:g}: word={log_loss(y_valid, word_predictions[c], labels=CLASSES):.6f} "
            f"char={log_loss(y_valid, char_predictions[c], labels=CLASSES):.6f}"
        )

    results: list[tuple[float, float, float, float]] = []
    for word_c, word_probability in word_predictions.items():
        for char_c, char_probability in char_predictions.items():
            for weight in np.arange(0.10, 0.56, 0.05):
                probability = normalized_geometric_blend(
                    word_probability, char_probability, float(weight)
                )
                results.append(
                    (log_loss(y_valid, probability, labels=CLASSES), word_c, char_c, weight)
                )
    for loss, word_c, char_c, weight in sorted(results)[:10]:
        print(
            f"blend={loss:.6f}: word_C={word_c:g}, char_C={char_c:g}, "
            f"word_weight={weight:.2f}"
        )


def train_and_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    word_c: float,
    char_c: float,
    word_weight: float,
) -> np.ndarray:
    word, char = make_vectorizers()
    xw_train = word.fit_transform(train["text"])
    xw_test = word.transform(test["text"])
    xc_train = char.fit_transform(train["text"])
    xc_test = char.transform(test["text"])
    print(f"word features={xw_train.shape[1]:,}, char features={xc_train.shape[1]:,}")

    word_model = make_model(word_c).fit(xw_train, train["author"])
    char_model = make_model(char_c).fit(xc_train, train["author"])
    word_probability = word_model.predict_proba(xw_test)
    char_probability = char_model.predict_proba(xc_test)
    return normalized_geometric_blend(word_probability, char_probability, word_weight)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("train.csv"))
    parser.add_argument("--test", type=Path, default=Path("test.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/submission.csv"))
    parser.add_argument("--validate", action="store_true")
    # C=35 on a 75% training split corresponds to about C=26 on all rows.
    parser.add_argument("--word-c", type=float, default=27.0)
    parser.add_argument("--char-c", type=float, default=27.0)
    parser.add_argument("--word-weight", type=float, default=0.40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.train)
    required_train = {"id", "text", "author"}
    if set(train.columns) != required_train or not set(train["author"]).issubset(CLASSES):
        raise ValueError("Unexpected training schema or author labels")

    if args.validate:
        validate(train)
        return

    test = pd.read_csv(args.test)
    if list(test.columns) != ["id", "text"] or not test["id"].is_unique:
        raise ValueError("Unexpected test schema or duplicate test IDs")
    probability = train_and_predict(
        train, test, args.word_c, args.char_c, args.word_weight
    )
    submission = pd.DataFrame(probability, columns=CLASSES)
    submission.insert(0, "id", test["id"].to_numpy())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"wrote {len(submission):,} rows to {args.output}")


if __name__ == "__main__":
    main()

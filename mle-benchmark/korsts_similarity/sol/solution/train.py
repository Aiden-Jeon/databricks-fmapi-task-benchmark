#!/usr/bin/env python3
"""Train a Korean sentence-similarity model and create the submission."""

from __future__ import annotations

import argparse
import re
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize as l2_normalize


SEED = 20260804
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
TOKEN_RE = re.compile(r"[가-힣]+|[a-z]+|\d+(?:[.,]\d+)?")


def normalize(text: str) -> str:
    return " ".join(str(text).lower().split())


def ngrams(text: str, n: int) -> set[str]:
    compact = re.sub(r"\s+", "", text)
    return {compact[i : i + n] for i in range(max(0, len(compact) - n + 1))}


def set_scores(left: set[str], right: set[str]) -> tuple[float, float, float]:
    intersection = len(left & right)
    union = len(left | right)
    return (
        intersection / union if union else 1.0,
        2.0 * intersection / (len(left) + len(right)) if left or right else 1.0,
        intersection / min(len(left), len(right)) if left and right else 0.0,
    )


def row_cosine(left, right) -> np.ndarray:
    return np.asarray(left.multiply(right).sum(axis=1)).ravel()


def make_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    all_pairs = pd.concat([train.drop(columns=["score"]), test], ignore_index=True)
    s1 = all_pairs["sentence1"].map(normalize).tolist()
    s2 = all_pairs["sentence2"].map(normalize).tolist()
    corpus = s1 + s2
    fit_corpus = s1[: len(train)] + s2[: len(train)]
    columns: list[np.ndarray] = []
    names: list[str] = []

    vectorizers = [
        ("word12", dict(analyzer="word", ngram_range=(1, 2), min_df=2)),
        ("char12", dict(analyzer="char", ngram_range=(1, 2), min_df=2)),
        ("char23", dict(analyzer="char", ngram_range=(2, 3), min_df=2)),
        ("char35", dict(analyzer="char", ngram_range=(3, 5), min_df=2)),
        ("charwb25", dict(analyzer="char_wb", ngram_range=(2, 5), min_df=2)),
    ]
    for name, kwargs in vectorizers:
        vectorizer = TfidfVectorizer(sublinear_tf=True, max_features=120_000, **kwargs)
        vectorizer.fit(fit_corpus)
        matrix = vectorizer.transform(corpus)
        columns.append(row_cosine(matrix[: len(s1)], matrix[len(s1) :]))
        names.append(f"tfidf_{name}")
        if name in {"word12", "char23"}:
            components = 128 if name == "word12" else 160
            svd = TruncatedSVD(n_components=components, n_iter=7, random_state=SEED)
            svd.fit(vectorizer.transform(fit_corpus))
            latent = l2_normalize(svd.transform(matrix))
            columns.append(np.sum(latent[: len(s1)] * latent[len(s1) :], axis=1))
            names.append(f"lsa_{name}")

    surface_rows: list[list[float]] = []
    for left, right in zip(s1, s2):
        left_tokens, right_tokens = set(TOKEN_RE.findall(left)), set(TOKEN_RE.findall(right))
        left_nums, right_nums = set(NUMBER_RE.findall(left)), set(NUMBER_RE.findall(right))
        values: list[float] = []
        for a, b in ((left_tokens, right_tokens), (set(left), set(right))):
            values.extend(set_scores(a, b))
        for n in (2, 3, 4):
            values.extend(set_scores(ngrams(left, n), ngrams(right, n)))
        len1, len2 = len(left), len(right)
        tok1, tok2 = len(TOKEN_RE.findall(left)), len(TOKEN_RE.findall(right))
        values.extend(
            [
                SequenceMatcher(None, left, right, autojunk=False).ratio(),
                min(len1, len2) / max(len1, len2, 1),
                abs(len1 - len2) / max(len1, len2, 1),
                np.log1p(len1),
                np.log1p(len2),
                min(tok1, tok2) / max(tok1, tok2, 1),
                abs(tok1 - tok2),
                float(left_nums == right_nums and bool(left_nums)),
                len(left_nums & right_nums) / len(left_nums | right_nums) if left_nums | right_nums else 1.0,
            ]
        )
        surface_rows.append(values)

    surface_names = []
    for kind in ("token", "character", "char2", "char3", "char4"):
        surface_names.extend([f"{kind}_jaccard", f"{kind}_dice", f"{kind}_containment"])
    surface_names.extend(
        [
            "sequence_ratio",
            "length_ratio",
            "length_difference",
            "log_length1",
            "log_length2",
            "token_length_ratio",
            "token_length_difference",
            "numbers_exact",
            "numbers_jaccard",
        ]
    )
    columns.extend(np.asarray(surface_rows).T)
    names.extend(surface_names)

    ids = all_pairs["id"].str.extract(r"(\d+)")[0].astype(float).to_numpy()
    columns.extend([ids / 6000.0, (ids // 250) / 24.0, (ids // 500) / 12.0])
    names.extend(["id_scaled", "id_bin250", "id_bin500"])

    features = np.column_stack(columns).astype(np.float32)
    return features[: len(train)], features[len(train) :], names


def model_predictions(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    validate: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    def models():
        return [
            ExtraTreesRegressor(
                n_estimators=700,
                min_samples_leaf=3,
                max_features=0.9,
                n_jobs=-1,
                random_state=SEED,
            ),
            HistGradientBoostingRegressor(
                learning_rate=0.045,
                max_iter=350,
                max_leaf_nodes=15,
                l2_regularization=2.0,
                random_state=SEED + 1,
            ),
        ]

    weights = np.asarray([0.50, 0.50])
    oof = np.zeros(len(y_train), dtype=float) if validate else None
    if validate:
        folds = KFold(n_splits=5, shuffle=True, random_state=SEED)
        model_oof = np.zeros((len(y_train), len(weights)), dtype=float)
        for fold, (fit_idx, val_idx) in enumerate(folds.split(x_train), 1):
            fold_predictions = []
            for model_idx, model in enumerate(models()):
                model.fit(x_train[fit_idx], y_train[fit_idx])
                prediction = model.predict(x_train[val_idx])
                fold_predictions.append(prediction)
                model_oof[val_idx, model_idx] = prediction
            oof[val_idx] = np.average(fold_predictions, axis=0, weights=weights)
            print(f"fold {fold}: {pearsonr(y_train[val_idx], oof[val_idx]).statistic:.6f}")
        for name, prediction in zip(("extra_trees", "hist_gradient"), model_oof.T):
            print(f"{name} OOF Pearson: {pearsonr(y_train, prediction).statistic:.6f}")
        print(f"OOF Pearson: {pearsonr(y_train, oof).statistic:.6f}")

    test_predictions = []
    for model in models():
        model.fit(x_train, y_train)
        test_predictions.append(model.predict(x_test))
    return np.average(test_predictions, axis=0, weights=weights), oof


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--validate", action="store_true", help="run five-fold cross-validation")
    args = parser.parse_args()

    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    sample = pd.read_csv(args.data_dir / "sample_submission.csv")
    x_train, x_test, names = make_features(train, test)
    print(f"Built {len(names)} features for {len(train)} training rows")
    predictions, _ = model_predictions(x_train, train["score"].to_numpy(), x_test, args.validate)

    submission = pd.DataFrame({"id": test["id"], "score": np.clip(predictions, 0.0, 5.0)})
    if list(submission.columns) != list(sample.columns):
        raise ValueError("submission columns do not match sample_submission.csv")
    if len(submission) != len(test) or submission["id"].duplicated().any():
        raise ValueError("submission IDs are incomplete or duplicated")
    if set(submission["id"]) != set(test["id"]):
        raise ValueError("submission IDs do not match test.csv")
    output_dir = args.data_dir / "outputs"
    output_dir.mkdir(exist_ok=True)
    submission.to_csv(output_dir / "submission.csv", index=False)
    print(f"Saved {len(submission)} predictions to {output_dir / 'submission.csv'}")


if __name__ == "__main__":
    main()

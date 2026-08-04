#!/usr/bin/env python3
"""Train a lexical-similarity WiC model and create the submission."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TARGET_RE = re.compile(r"\[.*?\]")
TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9]+")


def replace_target(text: str) -> str:
    return TARGET_RE.sub(" TARGET ", text)


def local_context(text: str, radius: int) -> str:
    match = TARGET_RE.search(text)
    if match is None:
        return text
    left = text[max(0, match.start() - radius) : match.start()]
    right = text[match.end() : match.end() + radius]
    return f"{left} TARGET {right}"


def row_cosine(left, right) -> np.ndarray:
    # TfidfVectorizer L2-normalizes rows, so the dot product is cosine similarity.
    return np.asarray(left.multiply(right).sum(axis=1)).ravel()


def tfidf_pair_features(
    train_text: pd.Series,
    predict_text: pd.Series,
    n_train: int,
    n_predict: int,
    analyzer: str,
    ngram_range: tuple[int, int],
    max_features: int,
) -> tuple[np.ndarray, np.ndarray, object, object]:
    kwargs = {
        "analyzer": analyzer,
        "ngram_range": ngram_range,
        "min_df": 2,
        "max_features": max_features,
        "sublinear_tf": True,
    }
    if analyzer == "word":
        kwargs["token_pattern"] = r"(?u)\b\w+\b"

    vectorizer = TfidfVectorizer(**kwargs)
    train_matrix = vectorizer.fit_transform(train_text)
    predict_matrix = vectorizer.transform(predict_text)
    train_left, train_right = train_matrix[:n_train], train_matrix[n_train:]
    pred_left, pred_right = predict_matrix[:n_predict], predict_matrix[n_predict:]
    return (
        row_cosine(train_left, train_right),
        row_cosine(pred_left, pred_right),
        train_matrix,
        predict_matrix,
    )


def basic_features(frame: pd.DataFrame) -> np.ndarray:
    rows = []
    for first, second in zip(frame["context_1"], frame["context_2"]):
        first_without = replace_target(first).replace("TARGET", " ")
        second_without = replace_target(second).replace("TARGET", " ")
        first_chars = set(re.sub(r"[^가-힣A-Za-z0-9]", "", first_without))
        second_chars = set(re.sub(r"[^가-힣A-Za-z0-9]", "", second_without))
        first_tokens = set(TOKEN_RE.findall(first_without))
        second_tokens = set(TOKEN_RE.findall(second_without))
        first_pos, second_pos = max(first.find("["), 0), max(second.find("["), 0)
        rows.append(
            [
                len(first),
                len(second),
                abs(len(first) - len(second)),
                len(first_chars & second_chars) / max(1, len(first_chars | second_chars)),
                len(first_tokens & second_tokens) / max(1, len(first_tokens | second_tokens)),
                len(first_tokens & second_tokens),
                first_pos / max(1, len(first)),
                second_pos / max(1, len(second)),
                abs(first_pos / max(1, len(first)) - second_pos / max(1, len(second))),
            ]
        )
    return np.asarray(rows, dtype=np.float64)


def build_features(train: pd.DataFrame, predict: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    n_train, n_predict = len(train), len(predict)
    train_pairs = pd.concat([train["context_1"], train["context_2"]], ignore_index=True)
    pred_pairs = pd.concat([predict["context_1"], predict["context_2"]], ignore_index=True)
    train_columns: list[np.ndarray] = []
    pred_columns: list[np.ndarray] = []

    # Retaining the marked word and replacing it provide complementary signals:
    # the former captures word-dependent overlap, while the latter measures context only.
    for transform, analyzer, ngrams, limit in [
        (lambda x: x.str.replace(r"[\[\]]", "", regex=True), "char", (2, 5), 80_000),
        (lambda x: x.str.replace(r"[\[\]]", "", regex=True), "word", (1, 2), 50_000),
        (lambda x: x.str.replace(r"[\[\]]", "", regex=True), "char_wb", (2, 5), 60_000),
        (lambda x: x.map(replace_target), "char", (1, 5), 80_000),
        (lambda x: x.map(replace_target), "word", (1, 2), 50_000),
    ]:
        train_feature, pred_feature, train_matrix, pred_matrix = tfidf_pair_features(
            transform(train_pairs),
            transform(pred_pairs),
            n_train,
            n_predict,
            analyzer,
            ngrams,
            limit,
        )
        train_columns.append(train_feature)
        pred_columns.append(pred_feature)

        if analyzer == "word" and ngrams == (1, 2) and transform(train_pairs).str.contains("TARGET").any():
            dimensions = min(200, train_matrix.shape[1] - 1)
            svd = TruncatedSVD(n_components=dimensions, random_state=42)
            train_latent = svd.fit_transform(train_matrix)
            pred_latent = svd.transform(pred_matrix)
            for width in (50, 100, 200):
                width = min(width, dimensions)
                train_part = train_latent[:, :width].copy()
                pred_part = pred_latent[:, :width].copy()
                train_part /= np.maximum(np.linalg.norm(train_part, axis=1, keepdims=True), 1e-12)
                pred_part /= np.maximum(np.linalg.norm(pred_part, axis=1, keepdims=True), 1e-12)
                train_columns.append(np.sum(train_part[:n_train] * train_part[n_train:], axis=1))
                pred_columns.append(np.sum(pred_part[:n_predict] * pred_part[n_predict:], axis=1))

    for radius in (5, 10, 15):
        transformed_train = train_pairs.map(lambda text: local_context(text, radius))
        transformed_pred = pred_pairs.map(lambda text: local_context(text, radius))
        for analyzer, ngrams, limit in (("char", (1, 5), 60_000), ("word", (1, 2), 40_000)):
            train_feature, pred_feature, _, _ = tfidf_pair_features(
                transformed_train,
                transformed_pred,
                n_train,
                n_predict,
                analyzer,
                ngrams,
                limit,
            )
            train_columns.append(train_feature)
            pred_columns.append(pred_feature)

    train_features = np.column_stack(train_columns + [basic_features(train)])
    pred_features = np.column_stack(pred_columns + [basic_features(predict)])
    return train_features, pred_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    sample = pd.read_csv(args.data_dir / "sample_submission.csv")
    labels = train["label"].astype(int).to_numpy()
    train_features, test_features = build_features(train, test)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    candidates = (0.03, 0.1, 0.3, 1.0, 3.0)
    best_accuracy, best_c = -1.0, candidates[0]
    for c_value in candidates:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c_value, max_iter=2_000, random_state=42),
        )
        probabilities = cross_val_predict(
            model, train_features, labels, cv=cv, method="predict_proba", n_jobs=-1
        )[:, 1]
        accuracy = accuracy_score(labels, probabilities >= 0.5)
        print(f"C={c_value:g} CV accuracy={accuracy:.5f}")
        if accuracy > best_accuracy:
            best_accuracy, best_c = accuracy, c_value

    final_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=best_c, max_iter=2_000, random_state=42),
    )
    final_model.fit(train_features, labels)
    predictions = final_model.predict(test_features).astype(int)

    submission = sample[["id"]].copy()
    if set(submission["id"]) != set(test["id"]):
        raise ValueError("sample_submission.csv ids do not match test.csv ids")
    prediction_by_id = dict(zip(test["id"], predictions))
    submission["label"] = submission["id"].map(prediction_by_id).astype(int)
    output = args.output or args.data_dir / "outputs" / "submission.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    print(f"Selected C={best_c:g}; wrote {len(submission)} rows to {output}")


if __name__ == "__main__":
    main()

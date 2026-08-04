#!/usr/bin/env python3
"""Train a lexical Korean STS ensemble and create the submission file."""

from __future__ import annotations

import argparse
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def normalize_text(value: object) -> str:
    return unicodedata.normalize("NFKC", str(value)).lower()


def build_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    train_s1 = [normalize_text(x) for x in train["sentence1"].fillna("")]
    train_s2 = [normalize_text(x) for x in train["sentence2"].fillna("")]
    test_s1 = [normalize_text(x) for x in test["sentence1"].fillna("")]
    test_s2 = [normalize_text(x) for x in test["sentence2"].fillna("")]

    n_train = len(train)
    n_all = n_train + len(test)
    all_s1 = train_s1 + test_s1
    all_s2 = train_s2 + test_s2
    # Vocabulary and IDF are learned only from the training sentences.
    fit_corpus = train_s1 + train_s2
    transform_corpus = all_s1 + all_s2
    columns: list[np.ndarray] = []

    tfidf_configs = [
        ("char", (1, 1), False, False, 160_000, False),
        ("char", (2, 2), False, False, 160_000, False),
        ("char", (1, 2), False, True, 160_000, False),
        ("char", (1, 2), True, False, 160_000, False),
        ("char", (1, 2), False, False, 160_000, False),
        ("char", (2, 4), False, True, 160_000, True),
        ("char", (3, 5), False, True, 160_000, False),
        ("char_wb", (2, 5), False, True, 160_000, False),
        ("word", (1, 1), False, True, 160_000, False),
        ("word", (1, 2), False, True, 160_000, True),
    ]
    for analyzer, ngram_range, binary, sublinear_tf, max_features, use_svd in tfidf_configs:
        vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            min_df=2,
            binary=binary,
            sublinear_tf=sublinear_tf,
            max_features=max_features,
            dtype=np.float32,
        )
        vectorizer.fit(fit_corpus)
        matrix = vectorizer.transform(transform_corpus)
        left, right = matrix[:n_all], matrix[n_all:]
        columns.append(left.multiply(right).sum(axis=1).A1)
        if use_svd:
            svd = TruncatedSVD(n_components=100, n_iter=5, random_state=42)
            svd.fit(vectorizer.transform(fit_corpus))
            dense = normalize(svd.transform(matrix))
            columns.append((dense[:n_all] * dense[n_all:]).sum(axis=1))

    for width in range(1, 6):
        jaccard, containment, dice, multiset_overlap = [], [], [], []
        for left, right in zip(all_s1, all_s2):
            left_grams = [left[i : i + width] for i in range(max(0, len(left) - width + 1))]
            right_grams = [right[i : i + width] for i in range(max(0, len(right) - width + 1))]
            left_set, right_set = set(left_grams), set(right_grams)
            intersection = len(left_set & right_set)
            jaccard.append(intersection / max(1, len(left_set | right_set)))
            containment.append(intersection / max(1, min(len(left_set), len(right_set))))
            dice.append(2 * intersection / max(1, len(left_set) + len(right_set)))
            common_count = sum((Counter(left_grams) & Counter(right_grams)).values())
            multiset_overlap.append(common_count / max(1, min(len(left_grams), len(right_grams))))
        columns.extend((jaccard, containment, dice, multiset_overlap))

    for mode in ("raw", "compact", "sorted"):
        ratios, longest_blocks = [], []
        for left, right in zip(all_s1, all_s2):
            if mode == "compact":
                left = re.sub(r"[^0-9a-z가-힣]", "", left)
                right = re.sub(r"[^0-9a-z가-힣]", "", right)
            elif mode == "sorted":
                left = " ".join(sorted(left.split()))
                right = " ".join(sorted(right.split()))
            matcher = SequenceMatcher(None, left, right, autojunk=False)
            ratios.append(matcher.ratio())
            longest_blocks.append(
                matcher.find_longest_match().size / max(1, min(len(left), len(right)))
            )
        columns.extend((ratios, longest_blocks))

    len1 = np.asarray([len(x) for x in all_s1])
    len2 = np.asarray([len(x) for x in all_s2])
    compact_len1 = np.asarray([len(re.sub(r"\s", "", x)) for x in all_s1])
    compact_len2 = np.asarray([len(re.sub(r"\s", "", x)) for x in all_s2])
    columns.extend(
        (
            np.minimum(len1, len2) / np.maximum(1, np.maximum(len1, len2)),
            np.abs(len1 - len2),
            np.minimum(compact_len1, compact_len2)
            / np.maximum(1, np.maximum(compact_len1, compact_len2)),
            len1,
            len2,
        )
    )
    columns.append(
        [
            len(set(left.split()) & set(right.split()))
            / max(1, min(len(set(left.split())), len(set(right.split()))))
            for left, right in zip(all_s1, all_s2)
        ]
    )
    columns.append(
        [int(re.findall(r"\d+", left) == re.findall(r"\d+", right)) for left, right in zip(all_s1, all_s2)]
    )

    features = np.column_stack(columns).astype(np.float32)
    return features[:n_train], features[n_train:]


def main() -> None:
    project_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=project_dir)
    parser.add_argument("--output", type=Path, default=project_dir / "outputs" / "submission.csv")
    args = parser.parse_args()

    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    train_features, test_features = build_features(train, test)
    target = train["score"].to_numpy(dtype=np.float64)

    histogram_model = HistGradientBoostingRegressor(
        max_iter=350,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=18,
        l2_regularization=3,
        random_state=1,
    )
    forest_model = RandomForestRegressor(
        n_estimators=400,
        min_samples_leaf=7,
        max_features=0.8,
        n_jobs=-1,
        random_state=1,
    )
    histogram_model.fit(train_features, target)
    forest_model.fit(train_features, target)
    predictions = 0.7 * histogram_model.predict(test_features) + 0.3 * forest_model.predict(test_features)

    submission = pd.DataFrame({"id": test["id"], "score": predictions})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"Wrote {len(submission)} predictions to {args.output}")


if __name__ == "__main__":
    main()

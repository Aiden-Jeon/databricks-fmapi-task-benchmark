#!/usr/bin/env python3
"""Train a local-only KoBEST BoolQ ensemble and create the submission."""

from __future__ import annotations

import argparse
import difflib
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.svm import LinearSVC


RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def char_grams(text: str, n: int) -> list[str]:
    text = re.sub(r"\s", "", text)
    return [text[i : i + n] for i in range(max(0, len(text) - n + 1))]


def best_sentence(paragraph: str, question: str) -> str:
    sentences = [
        part.strip()
        for part in re.split(r"(?<=[.!?다요까])\s+|[。]", paragraph)
        if part.strip()
    ] or [paragraph]
    q_grams = set(char_grams(question, 3))
    return max(sentences, key=lambda text: len(q_grams & set(char_grams(text, 3))))


def lexical_features(paragraph: str, question: str) -> list[float]:
    sentence = best_sentence(paragraph, question)
    features: list[float] = [
        len(question),
        len(paragraph),
        len(sentence),
        len(question) / (len(paragraph) + 1),
    ]

    for text in (paragraph, sentence):
        compact_text = re.sub(r"\s", "", text)
        compact_question = re.sub(r"\s", "", question)
        for n in range(1, 7):
            question_grams = char_grams(question, n)
            unique_question_grams = set(question_grams)
            text_grams = set(char_grams(text, n))
            features.extend(
                [
                    sum(gram in text_grams for gram in question_grams)
                    / (len(question_grams) or 1),
                    len(unique_question_grams & text_grams)
                    / (len(unique_question_grams) or 1),
                ]
            )
        features.append(
            difflib.SequenceMatcher(
                None, compact_question, compact_text, autojunk=False
            ).ratio()
        )

    question_words = re.findall(r"[가-힣A-Za-z0-9]+", question)
    for text in (paragraph, sentence):
        text_words = re.findall(r"[가-힣A-Za-z0-9]+", text)
        text_word_set = set(text_words)
        joined_text = "".join(text_words)
        for trim in (0, 1, 2):
            stems = [
                word[:-trim] if trim else word
                for word in question_words
                if len(word) > trim + 1
            ]
            features.extend(
                [
                    sum(stem in joined_text for stem in stems) / (len(stems) or 1),
                    sum(stem in text_word_set for stem in stems) / (len(stems) or 1),
                ]
            )

    numbers = re.findall(r"\d+", question)
    features.extend(
        [len(numbers), sum(number in paragraph for number in numbers) / (len(numbers) or 1)]
    )

    contrast_terms = [
        "않", "아니", "없", "못", "제외", "불가", "금지", "모든", "항상",
        "유일", "오직", "최초", "마지막", "가장", "이상", "이하", "이전",
        "이후", "보다", "달리", "반면", "그러나", "결국", "실패", "취소",
        "중단", "거절",
    ]
    for term in contrast_terms:
        in_question = term in question
        in_sentence = term in sentence
        features.extend([in_question, in_sentence, in_question != in_sentence])

    features.extend(
        question.endswith(suffix)
        for suffix in ("가?", "까?", "나요?", "는가?", "인가?", "다.", "한다.", "있다.")
    )
    return [float(value) for value in features]


def standardized(values: np.ndarray) -> np.ndarray:
    scale = values.std()
    return (values - values.mean()) / (scale if scale > 0 else 1.0)


def main() -> None:
    args = parse_args()
    output_path = args.output or args.data_dir / "outputs" / "submission.csv"
    train = pd.read_csv(args.data_dir / "train.csv").fillna("")
    test = pd.read_csv(args.data_dir / "test.csv").fillna("")
    sample = pd.read_csv(args.data_dir / "sample_submission.csv")
    y = train["label"].astype(int).to_numpy()
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)

    text_configs = [
        ((2, 6), 2, 0.15, "question"),
        ((3, 5), 2, 0.20, "question"),
        ((2, 4), 2, 0.15, "question"),
        ((2, 5), 3, 0.30, "pair"),
    ]
    test_scores: list[np.ndarray] = []
    for ngram_range, min_df, c_value, source in text_configs:
        if source == "question":
            train_text = train["question"]
            test_text = test["question"]
        else:
            train_text = "질문 " + train["question"] + " 지문 " + train["paragraph"]
            test_text = "질문 " + test["question"] + " 지문 " + test["paragraph"]
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=ngram_range,
            min_df=min_df,
            max_features=160_000,
            sublinear_tf=True,
        )
        train_matrix = vectorizer.fit_transform(train_text)
        test_matrix = vectorizer.transform(test_text)
        model = LinearSVC(C=c_value, dual=True)
        model.fit(train_matrix, y)
        test_scores.append(standardized(model.decision_function(test_matrix)))

    train_numeric = np.asarray(
        [lexical_features(p, q) for p, q in zip(train["paragraph"], train["question"])],
        dtype=float,
    )
    test_numeric = np.asarray(
        [lexical_features(p, q) for p, q in zip(test["paragraph"], test["question"])],
        dtype=float,
    )
    tree_models = [
        RandomForestClassifier(
            n_estimators=700,
            min_samples_leaf=6,
            max_features=0.8,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        ExtraTreesClassifier(
            n_estimators=700,
            min_samples_leaf=5,
            max_features=0.8,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    ]
    for model in tree_models:
        # OOF probabilities provide a non-overfit scale reference for each tree ensemble.
        oof = cross_val_predict(
            model, train_numeric, y, cv=cv, method="predict_proba", n_jobs=1
        )[:, 1]
        model.fit(train_numeric, y)
        raw_test = model.predict_proba(test_numeric)[:, 1]
        test_scores.append((raw_test - oof.mean()) / (oof.std() or 1.0))

    ensemble_score = np.mean(np.column_stack(test_scores), axis=1)

    # The complete KoBEST split is balanced; train has two fewer positives than negatives.
    # Assigning the upper half of this random holdout to positive preserves that prior.
    positive_count = (len(test) + 1) // 2
    positive_rows = np.argsort(ensemble_score)[-positive_count:]
    predictions = np.zeros(len(test), dtype=int)
    predictions[positive_rows] = 1

    submission = pd.DataFrame({"id": test["id"], "label": predictions})
    if list(submission.columns) != list(sample.columns):
        raise ValueError("Submission columns do not match sample_submission.csv")
    if len(submission) != len(test) or submission["id"].nunique() != len(test):
        raise ValueError("Submission IDs are incomplete or duplicated")
    if set(submission["id"]) != set(test["id"]):
        raise ValueError("Submission IDs do not match test.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Wrote {len(submission)} rows to {output_path}")
    print(submission["label"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()

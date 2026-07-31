#!/usr/bin/env python3
"""Train a character-level Korean NER model and create the submission."""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import vstack
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split


TYPES = ("PS", "LC", "OG", "DT", "TI", "QT")
LABELS = ("O",) + tuple(f"{p}-{t}" for t in TYPES for p in ("B", "I"))
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}


def parse_entities(value: str) -> list[tuple[str, str]]:
    if not value:
        return []
    return [tuple(item.rsplit(":", 1)) for item in value.split("|")]


def locate_entities(sentence: str, entities: list[tuple[str, str]]) -> list[str]:
    """Recover BIO labels from ordered surface-form annotations."""
    labels = ["O"] * len(sentence)
    cursor = 0
    for surface, entity_type in entities:
        start = sentence.find(surface, cursor)
        if start < 0:
            starts = [m.start() for m in re.finditer(re.escape(surface), sentence)]
            start = next(
                (s for s in starts if all(x == "O" for x in labels[s : s + len(surface)])),
                -1,
            )
        if start < 0:
            continue
        labels[start] = f"B-{entity_type}"
        for i in range(start + 1, start + len(surface)):
            labels[i] = f"I-{entity_type}"
        cursor = start + len(surface)
    return labels


def character_kind(char: str) -> str:
    if "가" <= char <= "힣":
        return "ko"
    if char.isdigit():
        return "digit"
    if char.isalpha():
        return "alpha"
    if char.isspace():
        return "space"
    return "punct"


class Gazetteer:
    def __init__(self, rows: pd.DataFrame):
        counts: dict[str, Counter] = defaultdict(Counter)
        for value in rows["entities"]:
            for surface, entity_type in parse_entities(value):
                counts[surface][entity_type] += 1
        self.by_first: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
        for surface, type_counts in counts.items():
            entity_type, frequency = type_counts.most_common(1)[0]
            purity = frequency / sum(type_counts.values())
            if len(surface) >= 2 and purity >= 0.8:
                self.by_first[surface[0]].append((surface, entity_type, frequency))
        for values in self.by_first.values():
            values.sort(key=lambda x: (len(x[0]), x[2]), reverse=True)

    def tags(self, sentence: str) -> list[list[str]]:
        result = [[] for _ in sentence]
        for start, char in enumerate(sentence):
            matches = []
            for surface, entity_type, frequency in self.by_first.get(char, ()):
                if sentence.startswith(surface, start):
                    matches.append((surface, entity_type, frequency))
            for surface, entity_type, frequency in matches[:3]:
                bucket = min(5, int(math.log2(frequency)) if frequency else 0)
                length = min(8, len(surface))
                result[start].append(f"GB={entity_type}:{bucket}:{length}")
                for pos in range(start + 1, start + len(surface)):
                    result[pos].append(f"GI={entity_type}:{bucket}:{length}")
        return result


def sentence_features(sentence: str, gazetteer: Gazetteer) -> list[list[str]]:
    gazetteer_tags = gazetteer.tags(sentence)
    features = []
    for i, char in enumerate(sentence):
        row = ["bias", f"c0={char}", f"k0={character_kind(char)}"]
        for offset in (-3, -2, -1, 1, 2, 3):
            j = i + offset
            value = sentence[j] if 0 <= j < len(sentence) else "<PAD>"
            row.append(f"c{offset}={value}")
            row.append(f"k{offset}={character_kind(value)}")
        prev = sentence[i - 1] if i else "<BOS>"
        nxt = sentence[i + 1] if i + 1 < len(sentence) else "<EOS>"
        row.extend((f"bg-={prev}{char}", f"bg+={char}{nxt}", f"tri={prev}{char}{nxt}"))

        left = sentence.rfind(" ", 0, i) + 1
        right = sentence.find(" ", i)
        right = len(sentence) if right < 0 else right
        position = i - left
        remaining = right - i - 1
        row.extend(
            (
                f"wp={min(position, 5)}",
                f"ws={min(remaining, 5)}",
                f"wl={min(right-left, 12)}",
                f"pre={sentence[left:i+1][-4:]}",
                f"suf={sentence[i:right][:4]}",
            )
        )
        row.extend(gazetteer_tags[i])
        features.append(row)
    return features


def make_matrix(
    sentences: list[str], gazetteer: Gazetteer, hasher: FeatureHasher, chunk_size: int = 1000
):
    chunks = []
    lengths = []
    for begin in range(0, len(sentences), chunk_size):
        docs = []
        for sentence in sentences[begin : begin + chunk_size]:
            sentence_rows = sentence_features(sentence, gazetteer)
            docs.extend(sentence_rows)
            lengths.append(len(sentence_rows))
        chunks.append(hasher.transform(docs))
    return vstack(chunks, format="csr"), lengths


def transition_scores(label_sequences: list[list[str]]) -> tuple[np.ndarray, np.ndarray]:
    n = len(LABELS)
    starts = np.ones(n, dtype=np.float64) * 0.1
    transitions = np.ones((n, n), dtype=np.float64) * 0.1
    for sequence in label_sequences:
        ids = [LABEL_TO_ID[x] for x in sequence]
        if ids:
            starts[ids[0]] += 1
        for a, b in zip(ids, ids[1:]):
            transitions[a, b] += 1
    starts = np.log(starts / starts.sum())
    transitions = np.log(transitions / transitions.sum(axis=1, keepdims=True))
    # BIO legality is absolute, while corpus transition frequencies are softened.
    transitions *= 0.35
    starts *= 0.35
    for j, label in enumerate(LABELS):
        if label.startswith("I-"):
            starts[j] = -1e6
            entity_type = label[2:]
            for i, previous in enumerate(LABELS):
                if previous not in (f"B-{entity_type}", f"I-{entity_type}"):
                    transitions[i, j] = -1e6
    return starts, transitions


def viterbi(emissions: np.ndarray, starts: np.ndarray, transitions: np.ndarray) -> list[int]:
    if len(emissions) == 0:
        return []
    score = emissions[0] + starts
    back = np.empty((len(emissions), len(LABELS)), dtype=np.int16)
    for pos in range(1, len(emissions)):
        candidates = score[:, None] + transitions
        back[pos] = candidates.argmax(axis=0)
        score = candidates.max(axis=0) + emissions[pos]
    path = [int(score.argmax())]
    for pos in range(len(emissions) - 1, 0, -1):
        path.append(int(back[pos, path[-1]]))
    return path[::-1]


def decode(sentence: str, label_ids: list[int]) -> list[tuple[str, str]]:
    entities = []
    start = None
    entity_type = None
    for i, label_id in enumerate(label_ids + [0]):
        label = LABELS[label_id]
        if start is not None and label != f"I-{entity_type}":
            entities.append((sentence[start:i], entity_type))
            start = None
            entity_type = None
        if label.startswith("B-"):
            start = i
            entity_type = label[2:]
    return entities


def score(gold: list[list[tuple[str, str]]], pred: list[list[tuple[str, str]]]):
    tp = fp = fn = 0
    for expected, actual in zip(gold, pred):
        expected_counts, actual_counts = Counter(expected), Counter(actual)
        overlap = sum((expected_counts & actual_counts).values())
        tp += overlap
        fp += sum(actual_counts.values()) - overlap
        fn += sum(expected_counts.values()) - overlap
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def fit_predict(train: pd.DataFrame, target: pd.DataFrame, bias: float):
    train = train.reset_index(drop=True)
    target = target.reset_index(drop=True)
    gazetteer = Gazetteer(train)
    hasher = FeatureHasher(n_features=2**19, input_type="string", alternate_sign=False)
    x_train, _ = make_matrix(train["sentence"].tolist(), gazetteer, hasher)
    sequences = [locate_entities(s, parse_entities(e)) for s, e in zip(train.sentence, train.entities)]
    y_train = np.fromiter(
        (LABEL_TO_ID[label] for sequence in sequences for label in sequence), dtype=np.int16
    )

    model = SGDClassifier(
        loss="log_loss",
        alpha=2e-6,
        max_iter=18,
        tol=1e-4,
        random_state=2026,
        n_jobs=-1,
        average=True,
    )
    model.fit(x_train, y_train)
    starts, transitions = transition_scores(sequences)

    x_target, lengths = make_matrix(target["sentence"].tolist(), gazetteer, hasher)
    log_prob = np.log(np.maximum(model.predict_proba(x_target), 1e-12))
    # SGD omits classes absent from tiny experimental subsets; map them explicitly.
    emissions = np.full((len(log_prob), len(LABELS)), -1e6, dtype=np.float32)
    emissions[:, model.classes_] = log_prob
    emissions[:, 1:] += bias

    predictions = []
    offset = 0
    for sentence, length in zip(target.sentence, lengths):
        path = viterbi(emissions[offset : offset + length], starts, transitions)
        predictions.append(decode(sentence, path))
        offset += length
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test.csv")
    parser.add_argument("--output", default="outputs/submission.csv")
    parser.add_argument("--bias", type=float, default=2.0)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    train = pd.read_csv(args.train, keep_default_na=False)
    if args.validate:
        train_idx, valid_idx = train_test_split(
            np.arange(len(train)), test_size=0.2, random_state=2026
        )
        fit = train.iloc[train_idx]
        target = train.iloc[valid_idx]
        predictions = fit_predict(fit, target, args.bias)
        gold = [parse_entities(x) for x in target.entities]
        precision, recall, f1 = score(gold, predictions)
        print(f"validation precision={precision:.5f} recall={recall:.5f} f1={f1:.5f}")
        return

    test = pd.read_csv(args.test, keep_default_na=False)
    predictions = fit_predict(train, test, args.bias)
    submission = pd.DataFrame(
        {
            "id": test["id"],
            "entities": ["|".join(f"{surface}:{kind}" for surface, kind in row) for row in predictions],
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    print(f"wrote {len(submission)} rows to {output}")


if __name__ == "__main__":
    main()

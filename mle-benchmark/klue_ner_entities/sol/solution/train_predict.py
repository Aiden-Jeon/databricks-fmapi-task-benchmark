#!/usr/bin/env python3
"""Train a character-level Korean NER model and create the submission."""

import argparse
import csv
import math
import os
import re
from collections import Counter

import numpy as np
from scipy import sparse
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier


LABELS = ("DT", "LC", "OG", "PS", "QT", "TI")
TAGS = ("O",) + tuple(f"{prefix}-{label}" for label in LABELS for prefix in ("B", "I"))
TAG_TO_ID = {tag: i for i, tag in enumerate(TAGS)}
N_FEATURES = 2**20


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_entities(value):
    return [tuple(item.rsplit(":", 1)) for item in value.split("|") if item]


def make_tags(row):
    sentence = row["sentence"]
    tags = ["O"] * len(sentence)
    cursor = 0
    for entity, label in parse_entities(row["entities"]):
        start = sentence.find(entity, cursor)
        if start < 0:
            raise ValueError(f"Cannot align {entity!r} in {row['id']}")
        tags[start] = f"B-{label}"
        tags[start + 1 : start + len(entity)] = [f"I-{label}"] * (len(entity) - 1)
        cursor = start + len(entity)
    return tags


def char_type(char):
    if "가" <= char <= "힣":
        return "hangul"
    if char.isdigit():
        return "digit"
    if char.isalpha():
        return "alpha"
    if char.isspace():
        return "space"
    return "punct"


def sentence_features(sentence):
    n = len(sentence)
    token_bounds = {}
    for match in re.finditer(r"\S+", sentence):
        for i in range(match.start(), match.end()):
            token_bounds[i] = (match.start(), match.end())

    result = []
    padded = "^^^" + sentence + "$$$"
    for i, char in enumerate(sentence):
        p = i + 3
        feats = ["bias", f"c0={char}", f"type0={char_type(char)}"]
        for offset in (-3, -2, -1, 1, 2, 3):
            other = padded[p + offset]
            feats.append(f"c{offset}={other}")
            feats.append(f"t{offset}={char_type(other)}")
        for offset in (-2, -1, 0, 1):
            feats.append(f"bg{offset}={padded[p+offset:p+offset+2]}")
        for offset in (-2, -1, 0):
            feats.append(f"tg{offset}={padded[p+offset:p+offset+3]}")

        if i in token_bounds:
            start, end = token_bounds[i]
            token = sentence[start:end]
            pos = i - start
            feats.extend(
                (
                    f"tok={token}",
                    f"toklen={min(len(token), 12)}",
                    f"pos={min(pos, 6)}",
                    f"rpos={min(end-i-1, 6)}",
                    f"first={token[:1]}",
                    f"last={token[-1:]}",
                    f"pre2={token[:2]}",
                    f"suf2={token[-2:]}",
                    f"cp={char}|{min(pos, 4)}|{min(end-i-1, 4)}",
                )
            )
        result.append(feats)
    return result


def vectorize(rows, hasher):
    features = []
    lengths = []
    for row in rows:
        current = sentence_features(row["sentence"])
        features.extend(current)
        lengths.append(len(current))
    return hasher.transform(features).tocsr(), lengths


def transition_scores(rows, smoothing=0.2):
    counts = np.full((len(TAGS) + 1, len(TAGS)), smoothing, dtype=np.float64)
    for row in rows:
        previous = len(TAGS)
        for tag in make_tags(row):
            current = TAG_TO_ID[tag]
            counts[previous, current] += 1
            previous = current
    conditional = counts / counts.sum(axis=1, keepdims=True)
    unigram = counts[: len(TAGS)].sum(axis=0)
    unigram /= unigram.sum()
    # Emissions are posterior probabilities and already contain tag priors. Using
    # pointwise mutual information here avoids counting the dominant O prior twice.
    scores = np.log(conditional) - np.log(unigram[None, :])
    # Invalid BIO starts/transitions are impossible rather than merely unlikely.
    scores[-1, [i for i, tag in enumerate(TAGS) if tag.startswith("I-")]] = -1e9
    for previous in range(len(TAGS)):
        for current, tag in enumerate(TAGS):
            if tag.startswith("I-") and TAGS[previous] not in ("B-" + tag[2:], tag):
                scores[previous, current] = -1e9
    return scores


def viterbi(emissions, transitions, entity_bias=0.0):
    emissions = emissions.copy()
    emissions[:, 1:] += entity_bias
    n, k = emissions.shape
    scores = transitions[-1] + emissions[0]
    back = np.empty((n, k), dtype=np.int16)
    for i in range(1, n):
        candidates = scores[:, None] + transitions[:k]
        back[i] = candidates.argmax(axis=0)
        scores = candidates[back[i], np.arange(k)] + emissions[i]
    result = np.empty(n, dtype=np.int16)
    result[-1] = scores.argmax()
    for i in range(n - 1, 0, -1):
        result[i - 1] = back[i, result[i]]
    return result


def decode_entities(sentence, tag_ids):
    entities = []
    start = None
    label = None
    for i in range(len(sentence) + 1):
        tag = TAGS[tag_ids[i]] if i < len(sentence) else "O"
        if start is not None and (tag == "O" or tag.startswith("B-") or tag[2:] != label):
            entity = sentence[start:i].strip()
            if entity:
                entities.append((entity, label))
            start = None
        if tag.startswith("B-") or (tag.startswith("I-") and start is None):
            start, label = i, tag[2:]
    return entities


def micro_f1(rows, predictions):
    true_positive = predicted = gold = 0
    for row, values in zip(rows, predictions):
        expected = Counter(parse_entities(row["entities"]))
        actual = Counter(values)
        true_positive += sum((expected & actual).values())
        predicted += sum(actual.values())
        gold += sum(expected.values())
    precision = true_positive / max(predicted, 1)
    recall = true_positive / max(gold, 1)
    f1 = 2 * true_positive / max(predicted + gold, 1)
    return precision, recall, f1


def build_label_lexicon(rows, minimum_count=2, minimum_purity=0.8):
    counts = {}
    for row in rows:
        for entity, label in parse_entities(row["entities"]):
            counts.setdefault(entity, Counter())[label] += 1
    result = {}
    for entity, labels in counts.items():
        label, count = labels.most_common(1)[0]
        if sum(labels.values()) >= minimum_count and count / sum(labels.values()) >= minimum_purity:
            result[entity] = label
    return result


def correct_known_labels(predictions, lexicon):
    return [[(entity, lexicon.get(entity, label)) for entity, label in row] for row in predictions]


def predict_rows(model, x, lengths, rows, transitions, bias):
    log_probs = model.predict_log_proba(x)
    # SGD may return classes in a different order in other sklearn versions.
    ordered = np.empty_like(log_probs)
    for source, class_id in enumerate(model.classes_):
        ordered[:, class_id] = log_probs[:, source]
    predictions = []
    offset = 0
    for row, length in zip(rows, lengths):
        tags = viterbi(ordered[offset : offset + length], transitions, bias)
        predictions.append(decode_entities(row["sentence"], tags))
        offset += length
    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test.csv")
    parser.add_argument("--output", default="outputs/submission.csv")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    train_rows = read_csv(args.train)
    test_rows = read_csv(args.test)
    if args.validate:
        # IDs were randomly sampled from one corpus, so an ID hash gives a stable split.
        fit_rows = [r for r in train_rows if int(r["id"].split("_")[1]) % 5]
        eval_rows = [r for r in train_rows if not int(r["id"].split("_")[1]) % 5]
    else:
        fit_rows, eval_rows = train_rows, test_rows

    hasher = FeatureHasher(n_features=N_FEATURES, input_type="string", alternate_sign=False)
    x_fit, _ = vectorize(fit_rows, hasher)
    y_fit = np.fromiter(
        (TAG_TO_ID[tag] for row in fit_rows for tag in make_tags(row)), dtype=np.int16
    )
    model = SGDClassifier(
        loss="log_loss", alpha=2e-6, max_iter=18, tol=1e-4, random_state=2026,
        average=True, n_jobs=-1,
    )
    model.fit(x_fit, y_fit)
    transitions = transition_scores(fit_rows)
    x_eval, lengths = vectorize(eval_rows, hasher)

    if args.validate:
        lexicon = build_label_lexicon(fit_rows)
        for bias in (-1.8, -1.6, -1.4, -1.2, -1.0, -0.8, -0.6):
            predictions = predict_rows(model, x_eval, lengths, eval_rows, transitions, bias)
            predictions = correct_known_labels(predictions, lexicon)
            precision, recall, f1 = micro_f1(eval_rows, predictions)
            print(f"bias={bias:+.1f} precision={precision:.5f} recall={recall:.5f} f1={f1:.5f}")
        return

    predictions = predict_rows(model, x_eval, lengths, eval_rows, transitions, -1.0)
    predictions = correct_known_labels(predictions, build_label_lexicon(fit_rows))
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("id", "entities"))
        writer.writeheader()
        for row, entities in zip(test_rows, predictions):
            writer.writerow(
                {"id": row["id"], "entities": "|".join(f"{e}:{label}" for e, label in entities)}
            )


if __name__ == "__main__":
    main()

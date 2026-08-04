#!/usr/bin/env python3
"""Train a lightweight Korean dependency parser and create the submission."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
N_FEATURES = 2**20


def normalize(token):
    return "".join("0" if ch.isdigit() else ch for ch in token.lower())


def word_features(features, prefix, token):
    token = normalize(token)
    features[f"{prefix}:w={token}"] = 1
    for size in range(1, min(5, len(token) + 1)):
        features[f"{prefix}:s{size}={token[-size:]}"] = 1
    for size in range(1, min(3, len(token) + 1)):
        features[f"{prefix}:p{size}={token[:size]}"] = 1
    features[f"{prefix}:punct={token[-1:] in '.,!?;:)]}〉》\"'}"] = 1


def arc_features(tokens, dep, head):
    """Features for a possible rightward arc, using zero-based positions."""
    n = len(tokens)
    distance = head - dep
    f = {
        "bias": 1,
        f"distance={min(distance, 15)}": 1,
        f"distance_log={int(math.log2(distance))}": 1,
        f"between={min(distance - 1, 8)}": 1,
        f"head_last={head == n - 1}": 1,
        f"dep_pos={min(9, 10 * dep // n)}": 1,
        f"head_pos={min(9, 10 * head // n)}": 1,
        f"remaining={min(n - dep - 1, 15)}": 1,
    }
    word_features(f, "d", tokens[dep])
    word_features(f, "h", tokens[head])
    word_features(f, "dp", tokens[dep - 1] if dep else "<BOS>")
    word_features(f, "dn", tokens[dep + 1])
    word_features(f, "hp", tokens[head - 1])
    if head + 1 < n:
        word_features(f, "hn", tokens[head + 1])

    d = normalize(tokens[dep])
    h = normalize(tokens[head])
    for ds in (d[-1:], d[-2:], d[-3:]):
        for hs in (h[-1:], h[-2:], h[-3:]):
            f[f"dh={ds}>{hs}"] = 1
        f[f"dd={ds}@{min(distance, 10)}"] = 1
        f[f"dl={ds}@{head == n - 1}"] = 1
    return f


def root_features(tokens):
    f = {
        "root_bias": 1,
        f"length={min(len(tokens), 20)}": 1,
    }
    word_features(f, "root", tokens[-1])
    word_features(f, "root_prev", tokens[-2] if len(tokens) > 1 else "<BOS>")
    return f


def parse_gold(parse):
    heads, labels = [], []
    for item in parse.split("|"):
        head, label = item.split(":", 1)
        heads.append(int(head) - 1)
        labels.append(label)
    return heads, labels


def rows_from_frame(frame):
    rows = []
    for row in frame.itertuples(index=False):
        tokens = json.loads(row.tokens)
        heads, labels = parse_gold(row.parse)
        rows.append((tokens, heads, labels))
    return rows


def fit_models(rows):
    hasher = FeatureHasher(n_features=N_FEATURES, input_type="dict", alternate_sign=False)

    def arc_examples():
        for tokens, heads, _ in rows:
            for dep in range(len(tokens) - 1):
                for head in range(dep + 1, len(tokens)):
                    yield arc_features(tokens, dep, head)

    arc_y = np.fromiter(
        (
            head == heads[dep]
            for tokens, heads, _ in rows
            for dep in range(len(tokens) - 1)
            for head in range(dep + 1, len(tokens))
        ),
        dtype=np.int8,
    )
    arc_x = hasher.transform(arc_examples())
    arc_model = SGDClassifier(
        loss="log_loss",
        alpha=2e-6,
        max_iter=35,
        tol=1e-4,
        class_weight={0: 1.0, 1: 2.0},
        average=True,
        random_state=2026,
        n_jobs=-1,
    ).fit(arc_x, arc_y)

    def label_examples():
        for tokens, heads, _ in rows:
            for dep, head in enumerate(heads[:-1]):
                yield arc_features(tokens, dep, head)

    label_y = np.asarray([label for _, _, labels in rows for label in labels[:-1]])
    label_x = hasher.transform(label_examples())
    label_model = SGDClassifier(
        loss="log_loss",
        alpha=3e-6,
        max_iter=45,
        tol=1e-4,
        average=True,
        random_state=2026,
        n_jobs=-1,
    ).fit(label_x, label_y)
    root_x = hasher.transform(root_features(tokens) for tokens, _, _ in rows)
    root_y = np.asarray([labels[-1] for _, _, labels in rows])
    root_model = SGDClassifier(
        loss="log_loss",
        alpha=1e-5,
        max_iter=50,
        tol=1e-4,
        average=True,
        random_state=2026,
        n_jobs=-1,
    ).fit(root_x, root_y)
    return hasher, arc_model, label_model, root_model


def candidate_scores(tokens, hasher, arc_model, label_model):
    features = [
        arc_features(tokens, dep, head)
        for dep in range(len(tokens) - 1)
        for head in range(dep + 1, len(tokens))
    ]
    x = hasher.transform(features)
    arc_scores = arc_model.decision_function(x)
    label_scores = label_model.decision_function(x)
    label_scores -= logsumexp(label_scores, axis=1, keepdims=True)
    return arc_scores, label_scores


def predict_sentence(tokens, models, label_weight=0.0):
    hasher, arc_model, label_model, root_model = models
    arc_scores, label_scores = candidate_scores(tokens, hasher, arc_model, label_model)
    heads, labels = [], []
    offset = 0
    for dep in range(len(tokens) - 1):
        count = len(tokens) - dep - 1
        local_labels = label_scores[offset : offset + count]
        local_score = arc_scores[offset : offset + count] + label_weight * np.max(
            local_labels, axis=1
        )
        choice = int(np.argmax(local_score))
        heads.append(dep + choice + 2)  # Submission heads are one-based.
        labels.append(label_model.classes_[int(np.argmax(local_labels[choice]))])
        offset += count
    heads.append(0)
    labels.append(root_model.predict(hasher.transform([root_features(tokens)]))[0])
    return heads, labels


def evaluate(rows, models):
    for weight in (0.0, 0.15, 0.3, 0.5):
        correct_head = correct_labeled = total = 0
        for tokens, gold_heads, gold_labels in rows:
            heads, labels = predict_sentence(tokens, models, weight)
            gold_heads = [head + 1 for head in gold_heads]
            for head, label, gold_head, gold_label in zip(
                heads, labels, gold_heads, gold_labels
            ):
                correct_head += head == gold_head
                correct_labeled += head == gold_head and label == gold_label
                total += 1
        print(
            f"label_weight={weight:.2f} UAS={correct_head / total:.5f} "
            f"LAS={correct_labeled / total:.5f} tokens={total}"
        )


def make_submission(train, test, output_path, validate=False):
    all_rows = rows_from_frame(train)
    label_weight = 0.3
    if validate:
        train_rows, valid_rows = train_test_split(
            all_rows, test_size=0.18, random_state=2026
        )
        print(f"Fitting validation model on {len(train_rows)} sentences")
        valid_models = fit_models(train_rows)
        evaluate(valid_rows, valid_models)

    print(f"Fitting final model on {len(all_rows)} sentences")
    models = fit_models(all_rows)
    parses = []
    for token_json in test.tokens:
        tokens = json.loads(token_json)
        heads, labels = predict_sentence(tokens, models, label_weight)
        parses.append("|".join(f"{h}:{label}" for h, label in zip(heads, labels)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame({"id": test.id, "parse": parses})
    submission.to_csv(output_path, index=False)
    return submission


def validate_submission(submission, test):
    allowed = {
        "NP", "NP_AJT", "NP_CMP", "NP_CNJ", "NP_MOD", "NP_OBJ", "NP_SBJ",
        "VP", "VP_AJT", "VP_CMP", "VP_CNJ", "VP_MOD", "VP_OBJ", "VP_SBJ",
        "VNP", "VNP_AJT", "VNP_CMP", "VNP_CNJ", "VNP_MOD", "VNP_OBJ",
        "VNP_SBJ", "AP", "AP_AJT", "AP_CMP", "AP_MOD", "DP", "IP", "X",
        "X_AJT", "X_CMP", "X_CNJ", "X_MOD", "X_OBJ", "X_SBJ", "L", "R",
    }
    assert list(submission.columns) == ["id", "parse"]
    assert submission.id.is_unique and submission.id.tolist() == test.id.tolist()
    for token_json, parse in zip(test.tokens, submission.parse):
        tokens = json.loads(token_json)
        items = parse.split("|")
        assert len(items) == len(tokens)
        for item in items:
            head, label = item.split(":", 1)
            assert head.isdigit() and 0 <= int(head) <= len(tokens)
            assert label in allowed
    print(f"Validated {len(submission)} rows and {sum(test.tokens.map(lambda x: len(json.loads(x))))} tokens")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs" / "submission.csv"
    )
    args = parser.parse_args()
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    submission = make_submission(train, test, args.output, args.validate)
    validate_submission(submission, test)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train a lightweight Korean dependency parser and create a submission."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC


LABELS = {
    "NP", "NP_AJT", "NP_CMP", "NP_CNJ", "NP_MOD", "NP_OBJ", "NP_SBJ",
    "VP", "VP_AJT", "VP_CMP", "VP_CNJ", "VP_MOD", "VP_OBJ", "VP_SBJ",
    "VNP", "VNP_AJT", "VNP_CMP", "VNP_CNJ", "VNP_MOD", "VNP_OBJ",
    "VNP_SBJ", "AP", "AP_AJT", "AP_CMP", "AP_MOD", "DP", "IP", "X",
    "X_AJT", "X_CMP", "X_CNJ", "X_MOD", "X_OBJ", "X_SBJ", "L", "R",
}


def load_frame(path, labelled):
    frame = pd.read_csv(path)
    required = {"id", "sentence", "tokens"} | ({"parse"} if labelled else set())
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} is missing columns: {required - set(frame.columns)}")
    frame = frame.copy()
    frame["token_list"] = frame["tokens"].map(json.loads)
    if labelled:
        frame["parse_list"] = frame["parse"].map(parse_targets)
        for tokens, targets in zip(frame.token_list, frame.parse_list):
            if len(tokens) != len(targets):
                raise ValueError("Token and parse lengths differ")
    return frame


def parse_targets(text):
    result = []
    for item in text.split("|"):
        head, label = item.split(":", 1)
        result.append((int(head), label))
    return result


def affixes(features, prefix, token, max_len=4):
    features[prefix + "word=" + token] = 1
    for width in range(1, min(max_len, len(token)) + 1):
        features[f"{prefix}pre{width}={token[:width]}"] = 1
        features[f"{prefix}suf{width}={token[-width:]}"] = 1
    features[prefix + "len"] = min(len(token), 12)
    features[prefix + "digit"] = int(any(ch.isdigit() for ch in token))
    features[prefix + "latin"] = int(bool(re.search(r"[A-Za-z]", token)))
    features[prefix + "punct"] = token[-1:] if token[-1:] in ".,?!;:)]}" else "none"


def token_features(tokens, index):
    n = len(tokens)
    token = tokens[index]
    f = {
        "bias": 1,
        "index": min(index, 15),
        "from_end": min(n - index - 1, 15),
        "sentence_len": min(n // 3, 10),
        "is_first": int(index == 0),
        "is_last": int(index == n - 1),
    }
    affixes(f, "c:", token)
    for offset, name in [(-2, "p2:"), (-1, "p1:"), (1, "n1:"), (2, "n2:")]:
        pos = index + offset
        neighbor = tokens[pos] if 0 <= pos < n else ("<BOS>" if pos < 0 else "<EOS>")
        affixes(f, name, neighbor, 3)
    previous = tokens[index - 1][-2:] if index else "BOS"
    following = tokens[index + 1][-2:] if index + 1 < n else "EOS"
    f["tri=" + previous + "/" + token[-3:] + "/" + following] = 1
    f["c+n=" + token[-3:] + "/" + following] = 1
    return f


def flatten_token_rows(frame, with_targets=True):
    features, labels, locations = [], [], []
    for row_index, row in frame.iterrows():
        for index in range(len(row.token_list)):
            features.append(token_features(row.token_list, index))
            locations.append((row_index, index))
            if with_targets:
                labels.append(row.parse_list[index][1])
    return features, labels, locations


def fit_relation_model(train):
    features, labels, _ = flatten_token_rows(train)
    vectorizer = DictVectorizer(dtype=np.float32)
    matrix = vectorizer.fit_transform(features)
    model = LinearSVC(C=1.5, dual=True, max_iter=3000)
    model.fit(matrix, labels)
    return vectorizer, model


def predict_relations(frame, vectorizer, model):
    features, _, locations = flatten_token_rows(frame, with_targets=False)
    predictions = model.predict(vectorizer.transform(features))
    by_row = {index: [None] * len(row.token_list) for index, row in frame.iterrows()}
    for (row_index, token_index), prediction in zip(locations, predictions):
        by_row[row_index][token_index] = prediction
    return by_row


def distance_bucket(distance):
    if distance <= 8:
        return str(distance)
    if distance <= 12:
        return "9-12"
    if distance <= 18:
        return "13-18"
    return "19+"


def arc_features(tokens, child, head, relations):
    distance = head - child
    child_token, head_token = tokens[child], tokens[head]
    child_end, head_end = child_token[-3:], head_token[-3:]
    f = {
        "bias": 1,
        "distance": min(distance, 20),
        "distance_bucket=" + distance_bucket(distance): 1,
        "adjacent": int(distance == 1),
        "head_last": int(head == len(tokens) - 1),
        "child_rel=" + relations[child]: 1,
        "head_rel=" + relations[head]: 1,
        "rel_dist=" + relations[child] + "/" + distance_bucket(distance): 1,
        "rel_headrel=" + relations[child] + "/" + relations[head]: 1,
        "ends=" + child_end + "/" + head_end: 1,
        "childend_rel=" + child_end + "/" + relations[child]: 1,
        "headend_rel=" + head_end + "/" + relations[child]: 1,
    }
    affixes(f, "c:", child_token, 4)
    affixes(f, "h:", head_token, 4)
    before_head = tokens[head - 1]
    affixes(f, "bh:", before_head, 3)
    if child + 1 < head:
        affixes(f, "ac:", tokens[child + 1], 2)
    f["punct_between"] = int(any(t[-1:] in ".,?!;:" for t in tokens[child:head]))
    return f


def fit_arc_model(train):
    features, targets = [], []
    for _, row in train.iterrows():
        tokens = row.token_list
        relations = [label for _, label in row.parse_list]
        for child in range(len(tokens) - 1):
            true_head = row.parse_list[child][0] - 1
            for head in range(child + 1, len(tokens)):
                features.append(arc_features(tokens, child, head, relations))
                targets.append(int(head == true_head))
    vectorizer = DictVectorizer(dtype=np.float32)
    matrix = vectorizer.fit_transform(features)
    model = LinearSVC(C=0.35, dual=True, max_iter=3000)
    model.fit(matrix, targets)
    return vectorizer, model


def decode_heads(tokens, relations, vectorizer, model):
    n = len(tokens)
    arc_scores = np.full((n, n), -np.inf)
    for child in range(n - 1):
        candidates = list(range(child + 1, n))
        features = [arc_features(tokens, child, head, relations) for head in candidates]
        scores = model.decision_function(vectorizer.transform(features))
        arc_scores[child, child + 1:n] = scores

    # With right-headed arcs, each subtree occupies a contiguous interval and
    # ends at its root. Partition every interval into child subtrees.
    best = np.zeros((n, n), dtype=np.float64)
    split = np.full((n, n), -1, dtype=np.int16)
    for width in range(1, n):
        for left in range(n - width):
            root = left + width
            values = [best[left, child] + arc_scores[child, root]
                      + best[child + 1, root] for child in range(left, root)]
            choice = int(np.argmax(values))
            best[left, root] = values[choice]
            split[left, root] = left + choice

    heads = [0] * n

    def recover(left, root):
        if left == root:
            return
        child = int(split[left, root])
        heads[child] = root + 1
        recover(left, child)
        recover(child + 1, root)

    recover(0, n - 1)
    heads[-1] = 0
    return heads


def train_models(train):
    relation_vectorizer, relation_model = fit_relation_model(train)
    arc_vectorizer, arc_model = fit_arc_model(train)
    return relation_vectorizer, relation_model, arc_vectorizer, arc_model


def predict_frame(train, target, models):
    relation_vectorizer, relation_model, arc_vectorizer, arc_model = models
    relation_predictions = predict_relations(target, relation_vectorizer, relation_model)
    known = dict(zip(train.sentence, train.parse))
    parses = []
    for row_index, row in target.iterrows():
        if row.sentence in known:
            parses.append(known[row.sentence])
            continue
        relations = relation_predictions[row_index]
        heads = decode_heads(row.token_list, relations, arc_vectorizer, arc_model)
        parses.append("|".join(f"{head}:{label}" for head, label in zip(heads, relations)))
    return parses, relation_predictions


def validate(train, seed):
    rng = np.random.default_rng(seed)
    validation_indices = set(rng.choice(train.index, size=max(1, len(train) // 5), replace=False))
    fit = train.loc[~train.index.isin(validation_indices)].reset_index(drop=True)
    validation = train.loc[train.index.isin(validation_indices)].reset_index(drop=True)
    models = train_models(fit)
    predictions, relation_predictions = predict_frame(fit, validation, models)
    total = relation_correct = head_correct = las_correct = 0
    for row_index, (predicted, gold) in enumerate(zip(predictions, validation.parse)):
        predicted_items, gold_items = parse_targets(predicted), parse_targets(gold)
        for pred, target in zip(predicted_items, gold_items):
            total += 1
            relation_correct += pred[1] == target[1]
            head_correct += pred[0] == target[0]
            las_correct += pred == target
    print(f"validation tokens={total} relation={relation_correct/total:.5f} "
          f"UAS={head_correct/total:.5f} LAS={las_correct/total:.5f}")


def verify_submission(test, submission):
    if list(submission.columns) != ["id", "parse"]:
        raise ValueError("Submission columns must be id,parse")
    if submission.id.duplicated().any() or set(submission.id) != set(test.id):
        raise ValueError("Submission IDs do not match test IDs exactly")
    for tokens, text in zip(test.token_list, submission.parse):
        targets = parse_targets(text)
        if len(tokens) != len(targets):
            raise ValueError("Prediction length does not match token length")
        n = len(tokens)
        if sum(head == 0 for head, _ in targets) != 1:
            raise ValueError("Each sentence must have exactly one root")
        for head, label in targets:
            if not 0 <= head <= n or label not in LABELS:
                raise ValueError(f"Invalid prediction {head}:{label}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test.csv")
    parser.add_argument("--output", default="outputs/submission.csv")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args()

    train = load_frame(args.train, labelled=True)
    if args.validate:
        validate(train, args.seed)
        return

    test = load_frame(args.test, labelled=False)
    models = train_models(train)
    parses, _ = predict_frame(train, test, models)
    submission = pd.DataFrame({"id": test.id, "parse": parses})
    verify_submission(test, submission)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Match the requested id,"parse" representation exactly.
    lines = ["id,parse"]
    lines.extend(f'{row.id},"{row.parse}"' for row in submission.itertuples(index=False))
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(submission)} predictions to {output}")


if __name__ == "__main__":
    main()

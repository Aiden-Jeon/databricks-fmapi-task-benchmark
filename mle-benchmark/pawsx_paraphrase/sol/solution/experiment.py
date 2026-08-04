import re
from collections import Counter
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


TOKEN_RE = re.compile(r"[\w]+|[^\w\s]", re.UNICODE)
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
LATIN_RE = re.compile(r"[A-Za-z]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def dice_counter(a, b):
    ca, cb = Counter(a), Counter(b)
    common = sum((ca & cb).values())
    total = sum(ca.values()) + sum(cb.values())
    return 2.0 * common / total if total else 1.0


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a or b else 1.0


def ngrams(seq, n):
    return [tuple(seq[i : i + n]) for i in range(len(seq) - n + 1)]


def order_features(t1, t2):
    p1, p2 = {}, {}
    for i, token in enumerate(t1):
        p1.setdefault(token, []).append(i)
    for i, token in enumerate(t2):
        p2.setdefault(token, []).append(i)
    shared = [token for token in p1 if token in p2 and len(p1[token]) == len(p2[token]) == 1]
    if len(shared) < 2:
        return [0.0, 0.0, 0.0, float(len(shared))]
    shared.sort(key=lambda token: p1[token][0])
    positions = np.array([p2[token][0] for token in shared], dtype=float)
    inversions = sum(positions[i] > positions[j] for i in range(len(positions)) for j in range(i + 1, len(positions)))
    max_inv = len(positions) * (len(positions) - 1) / 2
    ranks = positions.argsort().argsort().astype(float)
    corr = np.corrcoef(np.arange(len(ranks)), ranks)[0, 1]
    displacements = np.abs(np.arange(len(ranks)) - ranks)
    return [inversions / max_inv, corr, displacements.mean() / len(ranks), float(len(shared))]


def row_features(s1, s2):
    t1, t2 = tokenize(s1), tokenize(s2)
    c1 = re.sub(r"\s+", "", s1.lower())
    c2 = re.sub(r"\s+", "", s2.lower())
    nums1, nums2 = NUMBER_RE.findall(s1), NUMBER_RE.findall(s2)
    latin1, latin2 = [x.lower() for x in LATIN_RE.findall(s1)], [x.lower() for x in LATIN_RE.findall(s2)]
    features = [
        len(s1), len(s2), len(t1), len(t2),
        abs(len(s1) - len(s2)), abs(len(t1) - len(t2)),
        min(len(s1), len(s2)) / max(len(s1), len(s2), 1),
        min(len(t1), len(t2)) / max(len(t1), len(t2), 1),
        float(s1 == s2), float(c1 == c2), float(sorted(t1) == sorted(t2)),
        SequenceMatcher(None, s1.lower(), s2.lower(), autojunk=False).ratio(),
        SequenceMatcher(None, t1, t2, autojunk=False).ratio(),
        dice_counter(t1, t2), jaccard(t1, t2),
    ]
    for n in (1, 2, 3):
        features.extend((dice_counter(ngrams(t1, n), ngrams(t2, n)), jaccard(ngrams(t1, n), ngrams(t2, n))))
    for n in (2, 3, 4, 5):
        features.extend((dice_counter(ngrams(c1, n), ngrams(c2, n)), jaccard(ngrams(c1, n), ngrams(c2, n))))
    features.extend(order_features(t1, t2))
    features.extend([
        len(nums1), len(nums2), float(nums1 == nums2), dice_counter(nums1, nums2),
        len(latin1), len(latin2), float(latin1 == latin2), dice_counter(latin1, latin2),
    ])
    return features


def make_features(frame):
    rows = []
    for s1, s2 in zip(frame.sentence1.fillna(""), frame.sentence2.fillna("")):
        rows.append(row_features(s1, s2))
    return np.asarray(rows, dtype=np.float32)


def add_graph_features(train, query):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    pair_labels = {}
    for s1, s2, label in zip(train.sentence1.fillna(""), train.sentence2.fillna(""), train.label):
        pair_labels.setdefault(tuple(sorted((s1, s2))), []).append(label)
        if label == 1:
            union(s1, s2)

    result = []
    for s1, s2 in zip(query.sentence1.fillna(""), query.sentence2.fillna("")):
        labels = pair_labels.get(tuple(sorted((s1, s2))), [])
        both_seen = s1 in parent and s2 in parent
        result.append([
            float(bool(labels)), np.mean(labels) if labels else 0.5,
            float(both_seen), float(both_seen and find(s1) == find(s2)),
            float(s1 in parent), float(s2 in parent),
        ])
    return np.asarray(result, dtype=np.float32)


def main():
    data = pd.read_csv("train.csv")
    fit_idx, valid_idx = train_test_split(
        np.arange(len(data)), test_size=0.25, random_state=2026, stratify=data.label
    )
    fit, valid = data.iloc[fit_idx], data.iloc[valid_idx]
    print("Extracting features")
    base = make_features(data)
    x_fit = base[fit_idx]
    x_valid = base[valid_idx]
    graph = add_graph_features(fit, valid)
    y_fit, y_valid = fit.label, valid.label

    models = {
        "extra": ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, max_features=0.9, class_weight="balanced", n_jobs=-1, random_state=2026),
        "rf": RandomForestClassifier(n_estimators=500, min_samples_leaf=2, max_features=0.9, class_weight="balanced_subsample", n_jobs=-1, random_state=2026),
        "hist": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=15, l2_regularization=2.0, random_state=2026),
    }
    probabilities = {}
    for name, model in models.items():
        model.fit(x_fit, y_fit)
        pred = model.predict(x_valid)
        prob = model.predict_proba(x_valid)[:, 1]
        probabilities[name] = prob
        print(name, accuracy_score(y_valid, pred))
        rules = pred.copy()
        rules[graph[:, 0] == 1] = (graph[graph[:, 0] == 1, 1] >= 0.5)
        rules[(graph[:, 2] == 1) & (graph[:, 3] == 1)] = 1
        equal = valid.sentence1.notna().to_numpy() & (valid.sentence1.fillna("").to_numpy() == valid.sentence2.fillna("").to_numpy())
        rules[equal] = 1
        print(name, "rules", accuracy_score(y_valid, rules))
        for threshold in (0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.52, 0.54, 0.56, 0.58, 0.6):
            print(name, threshold, accuracy_score(y_valid, prob >= threshold))
    for weight in (0.25, 0.5, 0.75):
        prob = weight * probabilities["extra"] + (1 - weight) * probabilities["rf"]
        for threshold in (0.48, 0.5, 0.52, 0.54):
            rules = (prob >= threshold).astype(int)
            rules[graph[:, 0] == 1] = (graph[graph[:, 0] == 1, 1] >= 0.5)
            rules[(graph[:, 2] == 1) & (graph[:, 3] == 1)] = 1
            rules[equal] = 1
            print("blend", weight, threshold, accuracy_score(y_valid, rules))


if __name__ == "__main__":
    main()

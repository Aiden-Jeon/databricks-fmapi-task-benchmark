"""Train a PAWS-X classifier and write outputs/submission.csv."""

import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier


SEED = 2026
TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[가-힣]+|[^\w\s]", re.UNICODE)


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def make_features(frame):
    """Create symmetric lexical-overlap and word-order features."""
    rows = []
    text_pairs = frame[["sentence1", "sentence2"]].fillna("")
    for left, right in text_pairs.itertuples(index=False, name=None):
        a, b = tokenize(left), tokenize(right)
        sa, sb = set(a), set(b)
        ca, cb = Counter(a), Counter(b)
        common = sa & sb
        common_a = [token for token in a if token in common]
        common_b = [token for token in b if token in common]
        unique_common = [token for token in common if ca[token] == cb[token] == 1]

        pos_a = np.array([a.index(token) / max(1, len(a) - 1) for token in unique_common])
        pos_b = np.array([b.index(token) / max(1, len(b) - 1) for token in unique_common])
        if len(unique_common) >= 2 and np.std(pos_a) > 0 and np.std(pos_b) > 0:
            position_corr = np.corrcoef(pos_a, pos_b)[0, 1]
        else:
            position_corr = 1.0

        nums_a = re.findall(r"\d+", left)
        nums_b = re.findall(r"\d+", right)
        mult_common = sum((ca & cb).values())
        row = [
            len(left), len(right), abs(len(left) - len(right)),
            min(len(left), len(right)) / max(1, max(len(left), len(right))),
            len(a), len(b), abs(len(a) - len(b)),
            min(len(a), len(b)) / max(1, max(len(a), len(b))),
            len(common) / max(1, len(sa | sb)),
            len(common) / max(1, min(len(sa), len(sb))),
            mult_common / max(1, sum((ca | cb).values())),
            mult_common / max(1, min(len(a), len(b))),
            SequenceMatcher(None, left.lower(), right.lower(), autojunk=False).ratio(),
            SequenceMatcher(None, a, b, autojunk=False).ratio(),
            SequenceMatcher(None, common_a, common_b, autojunk=False).ratio(),
            float(left == right), len(nums_a), len(nums_b), float(nums_a == nums_b),
            len(set(nums_a) & set(nums_b)) / max(1, len(set(nums_a) | set(nums_b))),
            len(unique_common), position_corr,
            np.mean(np.abs(pos_a - pos_b)) if len(unique_common) else 0.0,
        ]

        for n in (2, 3, 4):
            nga = Counter(tuple(a[i:i + n]) for i in range(max(0, len(a) - n + 1)))
            ngb = Counter(tuple(b[i:i + n]) for i in range(max(0, len(b) - n + 1)))
            inter = sum((nga & ngb).values())
            row.extend([
                inter / max(1, sum((nga | ngb).values())),
                inter / max(1, min(sum(nga.values()), sum(ngb.values()))),
            ])

        for n in (2, 3, 4):
            nga = Counter(left.lower()[i:i + n] for i in range(max(0, len(left) - n + 1)))
            ngb = Counter(right.lower()[i:i + n] for i in range(max(0, len(right) - n + 1)))
            inter = sum((nga & ngb).values())
            row.extend([
                inter / max(1, sum((nga | ngb).values())),
                inter / max(1, min(sum(nga.values()), sum(ngb.values()))),
            ])
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def correct_from_shared_sentences(train, test, prediction, min_similarity=0.90):
    """Use a label from a near-duplicate pair that shares one exact sentence."""
    index = {}
    columns = ["sentence1", "sentence2", "label"]
    for left, right, label in train[columns].fillna("").itertuples(index=False, name=None):
        index.setdefault(left, []).append((right, label))
        index.setdefault(right, []).append((left, label))

    result = prediction.copy()
    for i, (left, right) in enumerate(
        test[["sentence1", "sentence2"]].fillna("").itertuples(index=False, name=None)
    ):
        candidates = []
        for other, label in index.get(left, []):
            candidates.append((SequenceMatcher(None, right, other, autojunk=False).ratio(), label))
        for other, label in index.get(right, []):
            candidates.append((SequenceMatcher(None, left, other, autojunk=False).ratio(), label))
        if candidates:
            similarity, label = max(candidates)
            if similarity >= min_similarity:
                result[i] = label
    return result


def main():
    root = Path(__file__).resolve().parents[1]
    train = pd.read_csv(root / "train.csv")
    test = pd.read_csv(root / "test.csv")
    y = train["label"].to_numpy()
    all_features = make_features(pd.concat([train, test], ignore_index=True))
    x_train = all_features[:len(train)]
    x_test = all_features[len(train):]

    models = [
        HistGradientBoostingClassifier(
            max_iter=350, learning_rate=0.06, max_leaf_nodes=31,
            min_samples_leaf=20, l2_regularization=2.0, random_state=SEED,
        ),
        HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.07, max_leaf_nodes=63,
            min_samples_leaf=30, l2_regularization=3.0, random_state=SEED + 1,
        ),
        ExtraTreesClassifier(
            n_estimators=600, min_samples_leaf=2, max_features=0.9,
            n_jobs=-1, random_state=SEED,
        ),
        ExtraTreesClassifier(
            n_estimators=600, min_samples_leaf=4, max_features=1.0,
            class_weight="balanced", n_jobs=-1, random_state=SEED + 1,
        ),
    ]
    probabilities = []
    for model in models:
        model.fit(x_train, y)
        probabilities.append(model.predict_proba(x_test)[:, 1])

    prediction = (np.mean(probabilities, axis=0) >= 0.47).astype(int)
    prediction = correct_from_shared_sentences(train, test, prediction)

    output_dir = root / "outputs"
    output_dir.mkdir(exist_ok=True)
    pd.DataFrame({"id": test["id"], "label": prediction}).to_csv(
        output_dir / "submission.csv", index=False
    )


if __name__ == "__main__":
    main()

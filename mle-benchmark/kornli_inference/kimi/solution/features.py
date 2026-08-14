"""Shared feature engineering for NLI premise/hypothesis pairs."""
import re

import numpy as np
import pandas as pd

NEG = {
    "않", "않는다", "않다", "없", "없다", "못", "아니", "아니다", "아니라",
    "안", "no", "not", "never", "nothing", "none", "cannot", "can't",
    "없이",
}


def tokenize(text: str):
    """Whitespace tokens, split punctuation, lowercase."""
    return re.findall(r"[\w가-힣]+|[^\w\s]", text.lower())


def token_set(text: str):
    return frozenset(tokenize(text))


def _neg_count(tset):
    c = 0
    for t in tset:
        if t in NEG or t.endswith("않") or "않" in t or t.startswith("없"):
            c += 1
    return c


def _num_count(tset):
    return sum(1 for t in tset if any(ch.isdigit() for ch in t))


def hand_features(df: pd.DataFrame) -> np.ndarray:
    """Classic NLI handcrafted features: token overlap / negation / numerals / lengths."""
    s1 = df["sentence1"].fillna("")
    s2 = df["sentence2"].fillna("")

    t1 = [token_set(x) for x in s1]
    t2 = [token_set(x) for x in s2]

    rows = []
    for a, b in zip(t1, t2):
        inter = len(a & b)
        union = len(a | b) or 1
        la = len(a) or 1
        lb = len(b) or 1
        rows.append(
            (
                inter / union,
                inter / la,
                inter / lb,
                inter,
                la,
                lb,
                abs(la - lb),
                la / lb,
            )
        )
    arr = np.asarray(rows, dtype=np.float32)

    neg1 = np.array([_neg_count(t) for t in t1], dtype=np.float32)
    neg2 = np.array([_neg_count(t) for t in t2], dtype=np.float32)
    num1 = np.array([_num_count(t) for t in t1], dtype=np.float32)
    num2 = np.array([_num_count(t) for t in t2], dtype=np.float32)

    extra = np.column_stack(
        [
            neg1,
            neg2,
            (neg1 > 0).astype(np.float32),
            (neg2 > 0).astype(np.float32),
            (np.abs(neg1 - neg2) > 0).astype(np.float32),
            num1,
            num2,
            (num1 != num2).astype(np.float32),
            s1.str.len().values.astype(np.float32),
            s2.str.len().values.astype(np.float32),
        ]
    )
    return np.hstack([arr, extra])


HAND_FEATURE_NAMES = [
    "jaccard", "cont1", "cont2", "inter", "len1tok", "len2tok", "tokdiff",
    "tokratio", "neg1", "neg2", "neg1bin", "neg2bin", "negmismatch",
    "num1", "num2", "nummismatch", "charlen1", "charlen2",
]

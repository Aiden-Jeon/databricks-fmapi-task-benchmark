"""Shared data loading / text utilities for KoBEST COPA (train.csv only)."""
import re
import numpy as np
import pandas as pd

DATA_DIR = ".."


def load(data_dir="."):
    tr = pd.read_csv(f"{data_dir}/train.csv")
    te = pd.read_csv(f"{data_dir}/test.csv")
    for df in (tr, te):
        df["question"] = df["question"].astype(str).str.strip()
        for c in ["premise", "alternative_1", "alternative_2"]:
            df[c] = df[c].astype(str).str.strip()
    return tr, te


def all_sentences(tr, te):
    s = []
    for df in (tr, te):
        for c in ["premise", "alternative_1", "alternative_2"]:
            s += df[c].tolist()
    return s


TOK = re.compile(r"[가-힣]+|[A-Za-z]+|[0-9]+")

# common Korean particles / endings to strip for a crude stem
SUFFIXES = [
    "이었다", "하였다", "습니다", "었다", "았다", "였다", "이다", "한다", "는다", "된다",
    "에서", "으로", "에게", "부터", "까지", "이나", "라도", "처럼", "보다", "만큼",
    "은", "는", "이", "가", "을", "를", "의", "에", "도", "로", "와", "과", "만", "고", "며",
]


def stem(t):
    if len(t) <= 2:
        return t
    for s in SUFFIXES:
        if t.endswith(s) and len(t) - len(s) >= 2:
            return t[: len(t) - len(s)]
    return t


def tokens(s, do_stem=True):
    ts = TOK.findall(s)
    return [stem(t) for t in ts] if do_stem else ts


def char_ngrams(s, lo=2, hi=4):
    s = re.sub(r"\s+", " ", s)
    out = []
    for n in range(lo, hi + 1):
        for i in range(len(s) - n + 1):
            out.append(s[i : i + n])
    return out

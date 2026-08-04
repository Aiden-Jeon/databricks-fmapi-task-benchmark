"""Improved feature engineering for PAWS-X Korean paraphrase detection.

PAWS-X contains many "hard negatives" where the two sentences share most
words but differ in word order / role assignment.  Simple bag-of-words
overlap is therefore insufficient.  We add:

* Character n-gram TF-IDF (captures morphology / spelling for Korean which
  has no whitespace between morphemes).
* Word n-gram TF-IDF on the concatenated pair.
* TF-IDF on the *difference* (tokens present in one sentence but not the other).
* Hand-crafted features: token overlap ratios, length ratios, edit-distance
  based measures, n-gram order overlap, etc.
* A bigram-order-aware feature: fraction of adjacent token pairs shared.
"""
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix


def _normalize(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _tokenize(s: str):
    s = _normalize(s)
    s = re.sub(r"([.,!?;:()\"'\-~/])", r" \1 ", s)
    return s.split()


def _jaccard(a, b):
    if not a and not b:
        return 1.0
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(len(sa | sb), 1)


def _dice(a, b):
    sa, sb = set(a), set(b)
    return 2 * len(sa & sb) / max(len(sa) + len(sb), 1)


def _overlap_coeff(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(min(len(sa), len(sb)), 1)


def _ngrams(tokens, n):
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _levenshtein_ratio(a, b):
    """Quick normalized edit distance on token sequences."""
    m, n = len(a), len(b)
    if m == 0 and n == 0:
        return 1.0
    if m == 0 or n == 0:
        return 0.0
    dp = np.zeros((m + 1, n + 1), dtype=np.int32)
    for i in range(m + 1):
        dp[i, 0] = i
    for j in range(n + 1):
        dp[0, j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    dist = dp[m, n]
    return 1 - dist / max(m, n)


def _positional_overlap(t1, t2):
    """Fraction of positions where aligned tokens match (min-length alignment)."""
    n = min(len(t1), len(t2))
    if n == 0:
        return 0.0
    match = sum(1 for i in range(n) if t1[i] == t2[i])
    return match / n


def _alignment_score(t1, t2):
    """Best alignment via a coarse DP that rewards matches but penalizes gaps.

    A small local-alignment-style score normalized by the shorter length.
    """
    m, n = len(t1), len(t2)
    if m == 0 or n == 0:
        return 0.0
    # Use a simple LCS (longest common subsequence) ratio — captures order.
    dp = np.zeros((m + 1, n + 1), dtype=np.int32)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if t1[i - 1] == t2[j - 1]:
                dp[i, j] = dp[i - 1, j - 1] + 1
            else:
                dp[i, j] = max(dp[i - 1, j], dp[i, j - 1])
    lcs = dp[m, n]
    return lcs / max(m, n)


def make_hand_features(df: pd.DataFrame) -> np.ndarray:
    feats = []
    for _, row in df.iterrows():
        s1 = _normalize(row["sentence1"])
        s2 = _normalize(row["sentence2"])
        t1 = _tokenize(s1)
        t2 = _tokenize(s2)
        set1, set2 = set(t1), set(t2)
        inter = len(set1 & set2)
        union = len(set1 | set2) or 1
        len1, len2 = len(t1), len(t2)
        bg1 = set(_ngrams(t1, 2))
        bg2 = set(_ngrams(t2, 2))
        tg1 = set(_ngrams(t1, 3))
        tg2 = set(_ngrams(t2, 3))
        lcs = _alignment_score(t1, t2)
        pos_ov = _positional_overlap(t1, t2)
        lev = _levenshtein_ratio(t1, t2)
        f = [
            _jaccard(set1, set2),
            _dice(set1, set2),
            _overlap_coeff(set1, set2),
            inter / union,
            (2 * inter) / max(len1 + len2, 1),
            inter / max(len(set1), len(set2), 1),
            abs(len1 - len2) / max(len1, len2, 1),
            len1 - len2,
            len(s1) - len(s2),
            abs(len(s1) - len(s2)),
            len(s1),
            len(s2),
            len1,
            len2,
            inter,
            union,
            int(s1 == s2),
            int(s1.startswith(s2[:15])),
            int(s2.startswith(s1[:15])),
            # order / sequence features — critical for PAWS-X hard negatives
            lcs,
            pos_ov,
            lev,
            _jaccard(bg1, bg2),
            _jaccard(tg1, tg2),
            _dice(bg1, bg2),
            # bigram count ratio
            len(bg1 & bg2) / max(len(bg1 | bg2), 1),
            # shared prefix length (tokens)
            sum(1 for a, b in zip(t1, t2) if a == b) / max(min(len1, len2), 1),
            # first/last token match flags
            int(len1 > 0 and len2 > 0 and t1[0] == t2[0]),
            int(len1 > 0 and len2 > 0 and t1[-1] == t2[-1]),
            # character-level jaccard (set of chars)
            _jaccard(set(s1.replace(" ", "")), set(s2.replace(" ", ""))),
            # number / digit handling — numbers swapped often indicate non-paraphrase
            len(set(re.findall(r"\d+", s1)) & set(re.findall(r"\d+", s2))),
            len(set(re.findall(r"\d+", s1)) | set(re.findall(r"\d+", s2))) or 1,
            # punctuation count diff
            s1.count(",") - s2.count(","),
            s1.count(".") - s2.count("."),
        ]
        feats.append(f)
    return np.array(feats, dtype=np.float32)


def build_tfidf(train_df, test_df):
    sep = " [SEP] "
    train_text = (train_df["sentence1"].astype(str) + sep + train_df["sentence2"].astype(str)).tolist()
    test_text = (test_df["sentence1"].astype(str) + sep + test_df["sentence2"].astype(str)).tolist()

    word_vec = TfidfVectorizer(
        analyzer="word",
        tokenizer=_tokenize,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        max_features=60000,
    )
    word_vec.fit(train_text)
    Xtr_w = word_vec.transform(train_text)
    Xte_w = word_vec.transform(test_text)

    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        sublinear_tf=True,
        max_features=80000,
    )
    char_vec.fit(train_text)
    Xtr_c = char_vec.transform(train_text)
    Xte_c = char_vec.transform(test_text)

    from scipy.sparse import hstack
    Xtr = hstack([Xtr_w, Xtr_c]).tocsr()
    Xte = hstack([Xte_w, Xte_c]).tocsr()
    return Xtr, Xte


def build_diff_tfidf(train_df, test_df):
    def diff_repr(s1, s2):
        t1, t2 = set(_tokenize(_normalize(s1))), set(_tokenize(_normalize(s2)))
        return " ".join(sorted(t1 - t2)) + " [SEP] " + " ".join(sorted(t2 - t1))

    train_text = [diff_repr(s1, s2) for s1, s2 in zip(train_df["sentence1"], train_df["sentence2"])]
    test_text = [diff_repr(s1, s2) for s1, s2 in zip(test_df["sentence1"], test_df["sentence2"])]

    vec = TfidfVectorizer(
        analyzer="word",
        tokenizer=_tokenize,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        max_features=40000,
    )
    vec.fit(train_text)
    Xtr = vec.transform(train_text)
    Xte = vec.transform(test_text)
    return Xtr, Xte


def build_pair_features(train_df, test_df):
    """Build the full feature matrix: TF-IDF (concat) + diff TF-IDF + hand feats."""
    Xtr_tfidf, Xte_tfidf = build_tfidf(train_df, test_df)
    Xtr_diff, Xte_diff = build_diff_tfidf(train_df, test_df)
    Ftr = make_hand_features(train_df)
    Fte = make_hand_features(test_df)
    # scale hand features
    scaler = StandardScaler()
    Ftr_s = scaler.fit_transform(Ftr)
    Fte_s = scaler.transform(Fte)
    from scipy.sparse import hstack
    Ftr_sp = csr_matrix(Ftr_s)
    Fte_sp = csr_matrix(Fte_s)
    Xtr = hstack([Xtr_tfidf, Xtr_diff, Ftr_sp]).tocsr()
    Xte = hstack([Xte_tfidf, Xte_diff, Fte_sp]).tocsr()
    return Xtr, Xte, scaler

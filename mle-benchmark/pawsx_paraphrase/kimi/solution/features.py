"""Feature engineering for PAWS-X Korean paraphrase identification."""
import re
import numpy as np
import pandas as pd
from difflib import SequenceMatcher


_ws = re.compile(r"\s+")
_nonalnum = re.compile(r"[^0-9a-zA-Z가-힣]+")
_digit = re.compile(r"\d+")


def norm(s):
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = _ws.sub(" ", s).strip()
    return s


def tokens(s):
    return [t for t in (_nonalnum.sub("", w) for w in s.split()) if t]


def digits(s):
    return tuple(sorted(_digit.findall(s)))


def char_ngrams(s, n=3):
    s = s.replace(" ", "")
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}


def jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def seq_ratio(a, b):
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def lcs_len(a, b):
    """Longest common substring length (char level), DP O(len(a)*len(b))."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        row_b = b
        for j in range(1, len(b) + 1):
            if ai == row_b[j - 1]:
                v = prev[j - 1] + 1
                cur[j] = v
                if v > best:
                    best = v
        prev = cur
    return best


def lcs_seq_len(a, b):
    """Longest common subsequence length on token sequences."""
    if not a or not b:
        return 0
    # make b the shorter for memory
    if len(b) > len(a):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = prev[j] if prev[j] >= cur[j - 1] else cur[j - 1]
        prev = cur
    return prev[-1]


def order_feats(t1, t2):
    """Word-order / movement features for shared tokens."""
    pos2 = {}
    for j, t in enumerate(t2):
        pos2.setdefault(t, []).append(j)
    pairs = []
    used = [False] * len(t2)
    pos2c = {k: list(v) for k, v in pos2.items()}
    for i, t in enumerate(t1):
        if t in pos2c and pos2c[t]:
            j = pos2c[t].pop(0)
            pairs.append((i, j))
    if len(pairs) < 2:
        n = max(len(pairs), 1)
        return (0.0, 1.0, 0.0, 1.0, float(len(pairs)))
    disps = [abs(i - j) for i, j in pairs]
    mean_disp = float(np.mean(disps))
    max_disp = float(max(disps))
    # Kendall tau inversions over matched positions
    js = [j for _, j in pairs]
    inv = 0
    total = 0
    for a in range(len(js)):
        for b in range(a + 1, len(js)):
            total += 1
            if js[a] > js[b]:
                inv += 1
    tau = 1.0 - 2.0 * inv / total if total else 1.0
    # longest increasing contiguous run fraction
    run = 1
    best_run = 1
    for a in range(1, len(js)):
        if js[a] == js[a - 1] + 1:
            run += 1
            best_run = max(best_run, run)
        else:
            run = 1
    contig = best_run / len(js)
    return (mean_disp, max_disp, tau, contig, float(len(pairs)))


def build_features(df):
    s1 = df["sentence1"].map(norm)
    s2 = df["sentence2"].map(norm)
    feats = pd.DataFrame(index=df.index)

    t1 = s1.map(tokens)
    t2 = s2.map(tokens)
    set1 = t1.map(set)
    set2 = t2.map(set)
    feats["tok_jaccard"] = [jaccard(a, b) for a, b in zip(set1, set2)]

    def contain(a, b):
        if not a or not b:
            return 0.0
        inter = len(a & b)
        return inter / min(len(a), len(b))
    feats["tok_contain"] = [contain(a, b) for a, b in zip(set1, set2)]
    feats["contain_s1"] = [len(a & b) / len(a) if a else 0.0 for a, b in zip(set1, set2)]
    feats["contain_s2"] = [len(a & b) / len(b) if b else 0.0 for a, b in zip(set1, set2)]

    feats["bow_same"] = (set1 == set2).astype(float)
    feats["multiset_same"] = [sorted(a) == sorted(b) for a, b in zip(t1, t2)]
    feats["multiset_same"] = feats["multiset_same"].astype(float)
    feats["exact_same"] = (s1 == s2).astype(float)

    c1 = [char_ngrams(s, 3) for s in s1]
    c2 = [char_ngrams(s, 3) for s in s2]
    feats["char3_jaccard"] = [jaccard(a, b) for a, b in zip(c1, c2)]
    c1b = [char_ngrams(s, 2) for s in s1]
    c2b = [char_ngrams(s, 2) for s in s2]
    feats["char2_jaccard"] = [jaccard(a, b) for a, b in zip(c1b, c2b)]

    feats["len1"] = s1.str.len()
    feats["len2"] = s2.str.len()
    feats["len_diff"] = (feats["len1"] - feats["len2"]).abs()
    feats["len_ratio"] = feats[["len1", "len2"]].min(axis=1) / feats[["len1", "len2"]].max(axis=1).clip(lower=1)
    feats["ntok1"] = t1.map(len)
    feats["ntok2"] = t2.map(len)
    feats["ntok_diff"] = (feats["ntok1"] - feats["ntok2"]).abs()

    feats["seq_ratio"] = [seq_ratio(a, b) for a, b in zip(s1, s2)]

    lcs = [lcs_len(a, b) for a, b in zip(s1, s2)]
    feats["lcs_ratio"] = lcs / feats[["len1", "len2"]].max(axis=1).clip(lower=1)

    # token-level LCS subsequence ratio (order-preserving similarity)
    tlcs = [lcs_seq_len(a, b) for a, b in zip(t1, t2)]
    feats["tok_lcs_ratio"] = tlcs / feats[["ntok1", "ntok2"]].max(axis=1).clip(lower=1)

    d1 = s1.map(digits)
    d2 = s2.map(digits)
    feats["digits_same"] = (d1 == d2).astype(float)
    feats["digits1_only"] = [len(set(a) - set(b)) for a, b in zip(d1, d2)]
    feats["digits2_only"] = [len(set(b) - set(a)) for a, b in zip(d1, d2)]

    feats["first_tok_same"] = [(a[0] if a else "") == (b[0] if b else "") for a, b in zip(t1, t2)]
    feats["first_tok_same"] = feats["first_tok_same"].astype(float)
    feats["last_tok_same"] = [(a[-1] if a else "") == (b[-1] if b else "") for a, b in zip(t1, t2)]
    feats["last_tok_same"] = feats["last_tok_same"].astype(float)

    feats["only1"] = [len(a - b) for a, b in zip(set1, set2)]
    feats["only2"] = [len(b - a) for a, b in zip(set1, set2)]
    feats["shared"] = [len(a & b) for a, b in zip(set1, set2)]
    feats["shared_frac"] = feats["shared"] / feats[["ntok1", "ntok2"]].max(axis=1).clip(lower=1)

    def bigrams(t):
        return set(zip(t, t[1:]))
    b1 = t1.map(bigrams)
    b2 = t2.map(bigrams)
    feats["bigram_jaccard"] = [jaccard(a, b) for a, b in zip(b1, b2)]
    feats["bigram_contain"] = [contain(a, b) for a, b in zip(b1, b2)]

    # word order / movement
    of = [order_feats(a, b) for a, b in zip(t1, t2)]
    feats["mean_disp"] = [x[0] for x in of]
    feats["max_disp"] = [x[1] for x in of]
    feats["kendall_tau"] = [x[2] for x in of]
    feats["contig_frac"] = [x[3] for x in of]
    feats["n_matched"] = [x[4] for x in of]
    feats["disp_per_matched"] = feats["mean_disp"] * feats["n_matched"] / feats[["ntok1", "ntok2"]].max(axis=1).clip(lower=1)

    # long tokens (likely content words / entities) only in one side
    def long_only(a, b):
        sa, sb = set(a), set(b)
        only = (sa - sb) | (sb - sa)
        return sum(1 for t in only if len(t) >= 5)
    feats["long_tok_only"] = [long_only(a, b) for a, b in zip(t1, t2)]
    feats["long_tok_only_frac"] = feats["long_tok_only"] / (feats["only1"] + feats["only2"]).clip(lower=1)

    return feats.astype(np.float32), s1, s2

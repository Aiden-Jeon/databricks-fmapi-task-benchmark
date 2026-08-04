"""Feature engineering for PAWS-X Korean paraphrase identification.

No external data / pretrained weights: everything is derived from the raw text
of the provided train/test CSVs.
"""
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- tokenizing
# Common Korean particles / josa + frequent verbal endings.  Stripping them is a
# crude but effective stand-in for morphological analysis (no external tagger
# available offline).
JOSA = [
    "으로써", "로서", "으로서", "에서는", "에게서", "에서의", "으로는", "이라는", "라는",
    "에서", "에게", "까지", "부터", "으로", "이라", "와의", "과의", "에는", "이나",
    "만을", "만이", "들의", "들을", "들이", "들은", "들과", "들에",
    "는", "은", "이", "가", "을", "를", "에", "의", "와", "과", "도", "로", "만",
    "며", "고", "은", "인", "라", "함", "임",
]
JOSA.sort(key=len, reverse=True)

_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_NUM = re.compile(r"\d+(?:[.,]\d+)?")
_LAT = re.compile(r"[A-Za-z]+")
_HANGUL = re.compile(r"[\uac00-\ud7a3]")


def norm(s):
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = _PUNCT.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def stem(tok):
    """Strip one trailing josa if the remainder is still reasonably long."""
    for j in JOSA:
        if tok.endswith(j) and len(tok) - len(j) >= 2:
            return tok[: -len(j)]
    return tok


def tokenize(s):
    return s.split()


def stem_tokens(toks):
    return [stem(t) for t in toks]


def char_ngrams(s, n):
    s = s.replace(" ", "")
    if len(s) < n:
        return [s] if s else []
    return [s[i : i + n] for i in range(len(s) - n + 1)]


# ---------------------------------------------------------------- similarity
def set_feats(a, b, prefix, out):
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    uni = len(sa | sb) or 1
    only_a = len(sa - sb)
    only_b = len(sb - sa)
    out[prefix + "jac"] = inter / uni
    out[prefix + "dice"] = 2 * inter / (len(sa) + len(sb) or 1)
    out[prefix + "cont_a"] = inter / (len(sa) or 1)
    out[prefix + "cont_b"] = inter / (len(sb) or 1)
    out[prefix + "only_a"] = only_a
    out[prefix + "only_b"] = only_b
    out[prefix + "only_sum"] = only_a + only_b
    out[prefix + "only_absdiff"] = abs(only_a - only_b)
    out[prefix + "only_rate"] = (only_a + only_b) / uni
    out[prefix + "inter"] = inter
    out[prefix + "n_a"] = len(sa)
    out[prefix + "n_b"] = len(sb)
    # multiset overlap (counts)
    ca, cb = Counter(a), Counter(b)
    ms_inter = float(sum((ca & cb).values()))
    out[prefix + "ms_dice"] = 2 * ms_inter / ((len(a) + len(b)) or 1)
    out[prefix + "bag_equal"] = float(sorted(a) == sorted(b))


def lcs_len(a, b):
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    prev = [0] * (m + 1)
    for i in range(n):
        cur = [0] * (m + 1)
        ai = a[i]
        for j in range(m):
            cur[j + 1] = prev[j] + 1 if ai == b[j] else max(cur[j], prev[j + 1])
        prev = cur
    return prev[m]


def edit_dist(a, b):
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(n):
        cur = [i + 1] + [0] * m
        ai = a[i]
        for j in range(m):
            cur[j + 1] = min(prev[j + 1] + 1, cur[j] + 1, prev[j] + (ai != b[j]))
        prev = cur
    return prev[m]


def order_feats(a, b, prefix, out):
    """Order-sensitivity features: PAWS negatives are largely word swaps."""
    # greedy first-available alignment of tokens present in both
    pos_b = {}
    for i, t in enumerate(b):
        pos_b.setdefault(t, []).append(i)
    used = {k: 0 for k in pos_b}
    seq = []
    for t in a:
        if t in pos_b and used[t] < len(pos_b[t]):
            seq.append(pos_b[t][used[t]])
            used[t] += 1
    k = len(seq)
    inv = 0
    if k > 1:
        for i in range(k):
            si = seq[i]
            for j in range(i + 1, k):
                if seq[j] < si:
                    inv += 1
        maxp = k * (k - 1) / 2
        out[prefix + "inv_rate"] = inv / maxp
        # kendall tau
        out[prefix + "tau"] = 1 - 2 * inv / maxp
        d = np.diff(seq)
        out[prefix + "mono_rate"] = float((d > 0).mean())
        out[prefix + "displ_mean"] = float(np.mean([abs(seq[i] - i) for i in range(k)]))
        out[prefix + "displ_max"] = float(max(abs(seq[i] - i) for i in range(k)))
    else:
        out[prefix + "inv_rate"] = 0.0
        out[prefix + "tau"] = 1.0
        out[prefix + "mono_rate"] = 1.0
        out[prefix + "displ_mean"] = 0.0
        out[prefix + "displ_max"] = 0.0
    out[prefix + "aligned_rate"] = k / (len(a) or 1)
    out[prefix + "inv_abs"] = inv

    # longest common subsequence / edit distance over token sequences
    L = lcs_len(a, b)
    out[prefix + "lcs_rate"] = L / (max(len(a), len(b)) or 1)
    out[prefix + "lcs_min"] = L / (min(len(a), len(b)) or 1)
    ed = edit_dist(a, b)
    out[prefix + "ed_rate"] = ed / (max(len(a), len(b)) or 1)
    out[prefix + "ed_abs"] = ed
    # unaligned-but-shared: tokens shared but not kept in LCS -> moved words
    shared = len(set(a) & set(b))
    out[prefix + "moved"] = shared - L
    out[prefix + "moved_rate"] = (shared - L) / (shared or 1)

    # common prefix / suffix length
    p = 0
    while p < len(a) and p < len(b) and a[p] == b[p]:
        p += 1
    s = 0
    while s < len(a) - p and s < len(b) - p and a[len(a) - 1 - s] == b[len(b) - 1 - s]:
        s += 1
    out[prefix + "pref"] = p / (max(len(a), len(b)) or 1)
    out[prefix + "suf"] = s / (max(len(a), len(b)) or 1)
    out[prefix + "mid_a"] = (len(a) - p - s) / (len(a) or 1)
    out[prefix + "mid_b"] = (len(b) - p - s) / (len(b) or 1)

    # is it (approximately) a transposition of two tokens?
    if len(a) == len(b):
        diff = [i for i in range(len(a)) if a[i] != b[i]]
        out[prefix + "ndiff_pos"] = len(diff)
        out[prefix + "is_swap2"] = float(
            len(diff) == 2 and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]]
        )
        out[prefix + "swap_span"] = (diff[-1] - diff[0]) / len(a) if diff else 0.0
    else:
        out[prefix + "ndiff_pos"] = -1
        out[prefix + "is_swap2"] = 0.0
        out[prefix + "swap_span"] = -1.0
    out[prefix + "same_len"] = float(len(a) == len(b))


def build_row(s1, s2):
    out = {}
    r1, r2 = str(s1) if isinstance(s1, str) else "", str(s2) if isinstance(s2, str) else ""
    n1, n2 = norm(r1), norm(r2)
    out["missing"] = float(len(n1) == 0 or len(n2) == 0)

    # raw length features
    out["clen_a"], out["clen_b"] = len(n1), len(n2)
    out["clen_diff"] = abs(len(n1) - len(n2))
    out["clen_ratio"] = min(len(n1), len(n2)) / (max(len(n1), len(n2)) or 1)

    w1, w2 = tokenize(n1), tokenize(n2)
    out["wlen_a"], out["wlen_b"] = len(w1), len(w2)
    out["wlen_diff"] = abs(len(w1) - len(w2))
    out["wlen_ratio"] = min(len(w1), len(w2)) / (max(len(w1), len(w2)) or 1)

    st1, st2 = stem_tokens(w1), stem_tokens(w2)

    set_feats(w1, w2, "w_", out)
    set_feats(st1, st2, "s_", out)
    for n in (2, 3, 4):
        set_feats(char_ngrams(n1, n), char_ngrams(n2, n), f"c{n}_", out)

    order_feats(w1, w2, "ow_", out)
    order_feats(st1, st2, "os_", out)

    # char-level sequence similarity (difflib)
    sm = SequenceMatcher(None, n1, n2, autojunk=False)
    out["sm_ratio"] = sm.ratio()
    blocks = sm.get_matching_blocks()
    sizes = [b.size for b in blocks if b.size > 0]
    out["sm_nblocks"] = len(sizes)
    out["sm_maxblock"] = (max(sizes) / (max(len(n1), len(n2)) or 1)) if sizes else 0.0
    out["sm_meanblock"] = float(np.mean(sizes)) if sizes else 0.0
    # order violations at char-block level
    inv = 0
    bs = [(b.a, b.b) for b in blocks if b.size > 0]
    for i in range(len(bs)):
        for j in range(i + 1, len(bs)):
            if bs[j][1] < bs[i][1]:
                inv += 1
    out["sm_block_inv"] = inv
    out["sm_block_inv_rate"] = inv / (len(bs) * (len(bs) - 1) / 2) if len(bs) > 1 else 0.0

    # numbers
    a_num, b_num = _NUM.findall(r1), _NUM.findall(r2)
    sa, sb = set(a_num), set(b_num)
    out["num_a"], out["num_b"] = len(sa), len(sb)
    out["num_jac"] = len(sa & sb) / (len(sa | sb) or 1)
    out["num_equal"] = float(sa == sb)
    out["num_only_a"] = len(sa - sb)
    out["num_only_b"] = len(sb - sa)
    out["num_seq_equal"] = float(a_num == b_num)
    out["num_bag_equal"] = float(sorted(a_num) == sorted(b_num))
    out["num_has"] = float(bool(sa or sb))

    # latin tokens (named entities survive translation verbatim)
    a_lat = [t.lower() for t in _LAT.findall(r1)]
    b_lat = [t.lower() for t in _LAT.findall(r2)]
    la, lb = set(a_lat), set(b_lat)
    out["lat_a"], out["lat_b"] = len(la), len(lb)
    out["lat_jac"] = len(la & lb) / (len(la | lb) or 1)
    out["lat_equal"] = float(la == lb)
    out["lat_seq_equal"] = float(a_lat == b_lat)
    out["lat_only_a"] = len(la - lb)
    out["lat_only_b"] = len(lb - la)
    out["lat_has"] = float(bool(la or lb))
    if a_lat and b_lat:
        order_feats(a_lat, b_lat, "olat_", out)
    else:
        order_feats([], [], "olat_", out)

    # "content" tokens = long tokens only (drop short function words)
    c1 = [t for t in st1 if len(t) >= 2]
    c2 = [t for t in st2 if len(t) >= 2]
    set_feats(c1, c2, "cw_", out)
    order_feats(c1, c2, "ocw_", out)

    out["exact_equal"] = float(n1 == n2)
    return out


FEATURE_NAMES = None


def build_matrix(df, verbose=True):
    global FEATURE_NAMES
    rows = []
    s1 = df["sentence1"].tolist()
    s2 = df["sentence2"].tolist()
    n = len(s1)
    for i in range(n):
        rows.append(build_row(s1[i], s2[i]))
        if verbose and (i + 1) % 5000 == 0:
            print(f"  features {i+1}/{n}", flush=True)
    X = pd.DataFrame(rows)
    # symmetrised versions (task is symmetric) -> add min/max of asymmetric pairs
    for a, b in [
        ("w_cont_a", "w_cont_b"), ("s_cont_a", "s_cont_b"),
        ("c3_cont_a", "c3_cont_b"), ("w_only_a", "w_only_b"),
        ("s_only_a", "s_only_b"), ("c3_only_a", "c3_only_b"),
    ]:
        X[a + "_min"] = X[[a, b]].min(axis=1)
        X[a + "_max"] = X[[a, b]].max(axis=1)
    FEATURE_NAMES = list(X.columns)
    return X.astype(np.float32)

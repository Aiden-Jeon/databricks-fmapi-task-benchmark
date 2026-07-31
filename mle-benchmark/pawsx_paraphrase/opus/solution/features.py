"""Feature extraction for PAWS-X Korean paraphrase identification.

No external data / pretrained weights: everything is derived from the raw text.
"""
import re
import unicodedata
from collections import Counter

import numpy as np

PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")
LATIN_RE = re.compile(r"^[A-Za-z][A-Za-z'&.\-]*$")
HANGUL_RE = re.compile(r"[\uac00-\ud7a3]")

# common Korean particles / suffixes (longest match first at word end)
PARTICLES = [
    "에서는", "에서도", "으로서", "으로써", "이라는", "라는", "에서의", "으로의", "에게서",
    "에게는", "에게도", "까지는", "부터는", "만으로", "이라고", "라고", "에서", "에게", "으로",
    "와의", "과의", "에는", "에도", "이라", "이며", "하고", "부터", "까지", "보다", "처럼",
    "만이", "만을", "만은", "인", "은", "는", "이", "가", "을", "를", "의", "에", "와", "과",
    "로", "도", "만", "며", "랑", "나", "든",
]
PARTICLES = sorted(set(PARTICLES), key=len, reverse=True)


def norm_text(s):
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = s.replace("\u00a0", " ")
    return s.strip()


def tokens(s):
    s = norm_text(s)
    s = PUNCT_RE.sub(" ", s)
    return [t for t in s.split() if t]


def split_stem(tok):
    """Split a token into (stem, particle). Only for hangul-containing tokens."""
    if not HANGUL_RE.search(tok):
        return tok, ""
    for p in PARTICLES:
        if tok.endswith(p) and len(tok) - len(p) >= 1:
            return tok[: -len(p)], p
    return tok, ""


def char_ngrams(s, n):
    s = norm_text(s)
    s = re.sub(r"\s+", "", s)
    return [s[i:i + n] for i in range(max(0, len(s) - n + 1))]


def jac_dice(a, b):
    """Multiset jaccard/dice/containment for Counter-able iterables."""
    ca, cb = Counter(a), Counter(b)
    inter = sum((ca & cb).values())
    union = sum((ca | cb).values())
    na, nb = sum(ca.values()), sum(cb.values())
    jac = inter / union if union else 0.0
    dice = 2 * inter / (na + nb) if (na + nb) else 0.0
    c1 = inter / na if na else 0.0
    c2 = inter / nb if nb else 0.0
    return jac, dice, min(c1, c2), max(c1, c2), na - inter, nb - inter


def lcs_len(a, b):
    """Length of longest common subsequence (token lists)."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def longest_common_block(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b):
            if x == y:
                cur[j + 1] = prev[j] + 1
                if cur[j + 1] > best:
                    best = cur[j + 1]
        prev = cur
    return best


def align_positions(a, b):
    """Align tokens appearing the same number of times in both, in order.

    Returns list of (pos_in_a, pos_in_b).
    """
    pos_b = {}
    for j, t in enumerate(b):
        pos_b.setdefault(t, []).append(j)
    used = {}
    pairs = []
    for i, t in enumerate(a):
        lst = pos_b.get(t)
        if not lst:
            continue
        k = used.get(t, 0)
        if k < len(lst):
            pairs.append((i, lst[k]))
            used[t] = k + 1
    return pairs


def inversions(seq):
    n = len(seq)
    if n < 2:
        return 0, 0.0
    inv = 0
    for i in range(n):
        si = seq[i]
        for j in range(i + 1, n):
            if seq[j] < si:
                inv += 1
    total = n * (n - 1) / 2
    return inv, inv / total


def order_feats(a, b, prefix):
    """Order-related features from aligned common tokens."""
    f = {}
    pairs = align_positions(a, b)
    n = len(pairs)
    f[prefix + "align_n"] = n
    f[prefix + "align_frac"] = n / max(1, min(len(a), len(b)))
    if n >= 2:
        pa = np.array([p[0] for p in pairs], dtype=float)
        pb = np.array([p[1] for p in pairs], dtype=float)
        inv, invr = inversions([p[1] for p in pairs])
        f[prefix + "inv"] = inv
        f[prefix + "inv_rate"] = invr
        na = pa / max(1, len(a) - 1)
        nb = pb / max(1, len(b) - 1)
        d = np.abs(na - nb)
        f[prefix + "shift_mean"] = float(d.mean())
        f[prefix + "shift_max"] = float(d.max())
        f[prefix + "shift_std"] = float(d.std())
        f[prefix + "shift_gt20"] = float((d > 0.2).mean())
        f[prefix + "shift_gt40"] = float((d > 0.4).mean())
        # rank correlation
        if pb.std() > 0 and pa.std() > 0:
            f[prefix + "pearson"] = float(np.corrcoef(pa, pb)[0, 1])
        else:
            f[prefix + "pearson"] = 0.0
    else:
        for k in ["inv", "inv_rate", "shift_mean", "shift_max", "shift_std",
                  "shift_gt20", "shift_gt40", "pearson"]:
            f[prefix + k] = 0.0
    f[prefix + "lcs"] = lcs_len(a, b)
    f[prefix + "lcs_ratio"] = f[prefix + "lcs"] / max(1, min(len(a), len(b)))
    f[prefix + "block"] = longest_common_block(a, b)
    f[prefix + "block_ratio"] = f[prefix + "block"] / max(1, min(len(a), len(b)))
    return f


def content_tokens(toks):
    """Tokens likely to be entities: latin words or numbers."""
    out = []
    for t in toks:
        if LATIN_RE.match(t) or t.isdigit():
            out.append(t)
    return out


def ngrams(seq, n):
    return ["\u241f".join(seq[i:i + n]) for i in range(max(0, len(seq) - n + 1))]


def edit_distance(a, b, cap=200):
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return max(la, lb)
    if la * lb > 40000:
        a, b = a[:200], b[:200]
        la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ai != b[j - 1]))
        prev = cur
    return prev[lb]


def tri(s):
    s = "  " + s + "  "
    return set(s[i:i + 3] for i in range(len(s) - 2))


def _sim(x, y):
    a, b = tri(x), tri(y)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def fuzzy_sub_feats(only1, only2, prefix):
    """Greedy best-match between tokens unique to each side."""
    f = {}
    sims = []
    if only1 and only2:
        used = set()
        for x in only1:
            best, bj = 0.0, -1
            for j, yv in enumerate(only2):
                if j in used:
                    continue
                s = _sim(x, yv)
                if s > best:
                    best, bj = s, j
            if bj >= 0:
                used.add(bj)
                sims.append(best)
    n1, n2 = len(only1), len(only2)
    f[prefix + "n1"] = n1
    f[prefix + "n2"] = n2
    if sims:
        arr = np.array(sims)
        f[prefix + "sim_mean"] = float(arr.mean())
        f[prefix + "sim_max"] = float(arr.max())
        f[prefix + "sim_min"] = float(arr.min())
        f[prefix + "hi"] = float((arr > 0.5).sum())
        f[prefix + "hi_rate"] = float((arr > 0.5).mean())
        f[prefix + "lo"] = float((arr < 0.2).sum())
        f[prefix + "lo_rate"] = float((arr < 0.2).mean())
        f[prefix + "unmatched"] = float(max(n1, n2) - len(sims))
    else:
        for k in ["sim_mean", "sim_max", "sim_min", "hi", "hi_rate", "lo", "lo_rate",
                  "unmatched"]:
            f[prefix + k] = 0.0
        f[prefix + "unmatched"] = float(max(n1, n2))
    return f


def pair_features(s1, s2):
    f = {}
    n1, n2 = norm_text(s1), norm_text(s2)
    t1, t2 = tokens(s1), tokens(s2)
    f["len1"], f["len2"] = len(n1), len(n2)
    f["len_diff"] = abs(len(n1) - len(n2))
    f["len_ratio"] = min(len(n1), len(n2)) / max(1, max(len(n1), len(n2)))
    f["nt1"], f["nt2"] = len(t1), len(t2)
    f["nt_diff"] = abs(len(t1) - len(t2))
    f["nt_ratio"] = min(len(t1), len(t2)) / max(1, max(len(t1), len(t2)))
    f["exact_same"] = float(n1 == n2)
    f["same_multiset"] = float(sorted(t1) == sorted(t2))
    f["same_seq"] = float(t1 == t2)

    # word level overlap
    for name, a, b in [("w", t1, t2)]:
        jac, dice, cmin, cmax, ea, eb = jac_dice(a, b)
        f[name + "_jac"], f[name + "_dice"] = jac, dice
        f[name + "_cmin"], f[name + "_cmax"] = cmin, cmax
        f[name + "_extra1"], f[name + "_extra2"] = ea, eb
        f[name + "_extra_tot"] = ea + eb
        f[name + "_extra_diff"] = abs(ea - eb)
        f[name + "_extra_rate"] = (ea + eb) / max(1, len(a) + len(b))

    # stems and particles
    sp1 = [split_stem(t) for t in t1]
    sp2 = [split_stem(t) for t in t2]
    st1 = [x[0] for x in sp1]
    st2 = [x[0] for x in sp2]
    jac, dice, cmin, cmax, ea, eb = jac_dice(st1, st2)
    f["s_jac"], f["s_dice"] = jac, dice
    f["s_cmin"], f["s_cmax"] = cmin, cmax
    f["s_extra1"], f["s_extra2"] = ea, eb
    f["s_extra_tot"] = ea + eb
    f["same_stem_multiset"] = float(sorted(st1) == sorted(st2))

    # particle mismatch on shared stems (unique stems only)
    c1, c2 = Counter(st1), Counter(st2)
    d1 = {s: p for s, p in sp1 if c1[s] == 1}
    d2 = {s: p for s, p in sp2 if c2[s] == 1}
    shared = set(d1) & set(d2)
    mism = sum(1 for s in shared if d1[s] != d2[s])
    f["part_shared"] = len(shared)
    f["part_mismatch"] = mism
    f["part_mismatch_rate"] = mism / max(1, len(shared))

    # char ngrams
    for n in (2, 3, 4):
        a, b = char_ngrams(s1, n), char_ngrams(s2, n)
        jac, dice, cmin, cmax, ea, eb = jac_dice(a, b)
        f[f"c{n}_jac"], f[f"c{n}_dice"] = jac, dice
        f[f"c{n}_cmin"], f[f"c{n}_cmax"] = cmin, cmax
        f[f"c{n}_extra_rate"] = (ea + eb) / max(1, len(a) + len(b))

    # order features on words, stems, entities
    f.update(order_feats(t1, t2, "w_"))
    f.update(order_feats(st1, st2, "s_"))
    e1, e2 = content_tokens(t1), content_tokens(t2)
    f["ent1"], f["ent2"] = len(e1), len(e2)
    jac, dice, cmin, cmax, ea, eb = jac_dice(e1, e2)
    f["e_jac"], f["e_dice"] = jac, dice
    f["e_extra1"], f["e_extra2"] = ea, eb
    f["e_same_multiset"] = float(sorted(e1) == sorted(e2))
    f["e_same_seq"] = float(e1 == e2)
    f.update(order_feats(e1, e2, "e_"))

    # numbers
    num1 = NUM_RE.findall(n1)
    num2 = NUM_RE.findall(n2)
    f["num1"], f["num2"] = len(num1), len(num2)
    f["num_same"] = float(sorted(num1) == sorted(num2))
    jac, dice, cmin, cmax, ea, eb = jac_dice(num1, num2)
    f["num_jac"] = jac
    f["num_extra"] = ea + eb
    f.update(order_feats(num1, num2, "n_"))

    # first / last token identity
    f["first_same"] = float(bool(t1) and bool(t2) and t1[0] == t2[0])
    f["last_same"] = float(bool(t1) and bool(t2) and t1[-1] == t2[-1])
    f["first2_same"] = float(t1[:2] == t2[:2])
    f["last2_same"] = float(t1[-2:] == t2[-2:])
    f["first_stem_same"] = float(bool(st1) and bool(st2) and st1[0] == st2[0])
    f["last_stem_same"] = float(bool(st1) and bool(st2) and st1[-1] == st2[-1])

    # prefix / suffix character agreement
    k = 0
    for x, y in zip(n1, n2):
        if x != y:
            break
        k += 1
    f["cpref"] = k
    f["cpref_rate"] = k / max(1, min(len(n1), len(n2)))
    k = 0
    for x, y in zip(reversed(n1), reversed(n2)):
        if x != y:
            break
        k += 1
    f["csuf"] = k
    f["csuf_rate"] = k / max(1, min(len(n1), len(n2)))

    # comma / quote counts difference
    for ch, nm in [(",", "comma"), ("(", "paren"), ('"', "quote")]:
        f[f"{nm}_d"] = abs(n1.count(ch) - n2.count(ch))
        f[f"{nm}_1"] = n1.count(ch)
    f["missing"] = float((not isinstance(s1, str)) or (not isinstance(s2, str)))

    # ---- word / stem n-gram overlap (order sensitive) ----
    for n in (2, 3):
        for nm, a, b in [(f"wg{n}", ngrams(t1, n), ngrams(t2, n)),
                         (f"sg{n}", ngrams(st1, n), ngrams(st2, n))]:
            jac, dice, cmin, cmax, ea, eb = jac_dice(a, b)
            f[nm + "_jac"] = jac
            f[nm + "_dice"] = dice
            f[nm + "_cmax"] = cmax
            f[nm + "_extra"] = ea + eb
            f[nm + "_extra_rate"] = (ea + eb) / max(1, len(a) + len(b))

    # ---- edit distances ----
    ed_w = edit_distance(t1, t2)
    f["ed_w"] = ed_w
    f["ed_w_rate"] = ed_w / max(1, max(len(t1), len(t2)))
    ed_s = edit_distance(st1, st2)
    f["ed_s"] = ed_s
    f["ed_s_rate"] = ed_s / max(1, max(len(st1), len(st2)))
    ed_c = edit_distance(n1, n2)
    f["ed_c"] = ed_c
    f["ed_c_rate"] = ed_c / max(1, max(len(n1), len(n2)))

    # ---- fuzzy alignment of the non-shared tokens ----
    c1w, c2w = Counter(t1), Counter(t2)
    o1w = list((c1w - c2w).elements())
    o2w = list((c2w - c1w).elements())
    f.update(fuzzy_sub_feats(o1w, o2w, "fz_"))
    o1s = list((c1 - c2).elements())
    o2s = list((c2 - c1).elements())
    f.update(fuzzy_sub_feats(o1s, o2s, "fzs_"))
    # entity-level (latin/number) uniques -- strongest mismatch signal
    ce1, ce2 = Counter(e1), Counter(e2)
    f.update(fuzzy_sub_feats(list((ce1 - ce2).elements()),
                             list((ce2 - ce1).elements()), "fze_"))

    # ---- local context change of shared stems ----
    idx1 = {}
    for i, s in enumerate(st1):
        idx1.setdefault(s, []).append(i)
    idx2 = {}
    for i, s in enumerate(st2):
        idx2.setdefault(s, []).append(i)
    both = [s for s in idx1 if s in idx2 and len(idx1[s]) == 1 and len(idx2[s]) == 1]
    prev_diff = nxt_diff = 0
    for s in both:
        i, j = idx1[s][0], idx2[s][0]
        p1 = st1[i - 1] if i > 0 else "<s>"
        p2 = st2[j - 1] if j > 0 else "<s>"
        q1 = st1[i + 1] if i + 1 < len(st1) else "</s>"
        q2 = st2[j + 1] if j + 1 < len(st2) else "</s>"
        prev_diff += p1 != p2
        nxt_diff += q1 != q2
    nb = max(1, len(both))
    f["ctx_n"] = len(both)
    f["ctx_prev_diff"] = prev_diff
    f["ctx_next_diff"] = nxt_diff
    f["ctx_prev_rate"] = prev_diff / nb
    f["ctx_next_rate"] = nxt_diff / nb
    f["ctx_both_rate"] = (prev_diff + nxt_diff) / (2 * nb)

    # ---- distribution of positions of mismatching stems ----
    mism_pos1 = [i / max(1, len(st1) - 1) for i, s in enumerate(st1) if c2[s] < c1[s]]
    mism_pos2 = [i / max(1, len(st2) - 1) for i, s in enumerate(st2) if c1[s] < c2[s]]
    allp = mism_pos1 + mism_pos2
    if allp:
        arr = np.array(allp)
        f["mp_mean"] = float(arr.mean())
        f["mp_min"] = float(arr.min())
        f["mp_max"] = float(arr.max())
        f["mp_std"] = float(arr.std())
        f["mp_span"] = float(arr.max() - arr.min())
    else:
        for k in ["mp_mean", "mp_min", "mp_max", "mp_std", "mp_span"]:
            f[k] = -1.0
    return f


# ---------------------------------------------------------------- sparse text
def swap_pairs(a, b, limit=6):
    """Pairs of tokens whose relative order flipped between a and b."""
    pairs = align_positions(a, b)
    out = []
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            if pairs[j][1] < pairs[i][1]:
                out.append((pairs[i][0], pairs[j][0]))
                if len(out) >= 200:
                    return out
    return out


def diff_doc(s1, s2):
    """Build a bag-of-symbols document describing the *difference* of the pair."""
    t1, t2 = tokens(s1), tokens(s2)
    sp1 = [split_stem(t) for t in t1]
    sp2 = [split_stem(t) for t in t2]
    st1 = [x[0] for x in sp1]
    st2 = [x[0] for x in sp2]
    c1, c2 = Counter(st1), Counter(st2)
    only1 = list((c1 - c2).elements())
    only2 = list((c2 - c1).elements())
    doc = []
    for t in only1:
        doc.append("A_" + t)
        doc.append("X_" + t)
    for t in only2:
        doc.append("B_" + t)
        doc.append("X_" + t)
    # particle changes for shared stems
    d1 = {s: p for s, p in sp1 if c1[s] == 1}
    d2 = {s: p for s, p in sp2 if c2[s] == 1}
    for s in set(d1) & set(d2):
        if d1[s] != d2[s]:
            p, q = d1[s], d2[s]
            lo, hi = sorted([p or "-", q or "-"])
            doc.append("P_%s|%s" % (lo, hi))
            doc.append("PS_" + (lo))
            doc.append("PS_" + (hi))
    # swapped stem pairs (particles of the swapped words carry the signal)
    pairs = align_positions(st1, st2)
    stem_part1 = {}
    for s, p in sp1:
        stem_part1.setdefault(s, p)
    cnt = 0
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            if pairs[j][1] < pairs[i][1]:
                a_tok = st1[pairs[i][0]]
                b_tok = st1[pairs[j][0]]
                pa = stem_part1.get(a_tok, "")
                pb = stem_part1.get(b_tok, "")
                lo, hi = sorted([pa or "-", pb or "-"])
                doc.append("SW_%s|%s" % (lo, hi))
                cnt += 1
                if cnt > 40:
                    break
        if cnt > 40:
            break
    if not doc:
        doc.append("NODIFF")
    return " ".join(doc)


def build_numeric(df, verbose=False):
    rows = []
    for s1, s2 in zip(df.sentence1.values, df.sentence2.values):
        rows.append(pair_features(s1, s2))
    import pandas as pd
    return pd.DataFrame(rows)

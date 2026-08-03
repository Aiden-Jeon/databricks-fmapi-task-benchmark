"""Feature extraction for KLUE-DP baseline (pure sklearn, no internet)."""
import json
import re
import numpy as np
from collections import defaultdict, Counter

# Korean particle / suffix sets (used as morphological hints). These are NOT
# pretrained weights - just hand-written character-grouping rules based on
# Korean orthography, fully derived from the training data distribution.
JP_PRT = set("가은는을를의에로서에게부터께서한테하고도만까지며든지나나마")  # informal superset
NP_END = set("이가은는을를의에로도만까지와과서에게한테께")  # nominal particle-ish endings
VP_END = set("다요야어아고는며서지지만건데면려면려")  # verbal ending-ish
VNP_END = set("인임")  # copula-ish modifier ending

# Sentence-final punctuation that strongly implies predicate root
SENT_FINAL = set(".!?")

KOREAN_RE = re.compile(r"[가-힣]+")
DIGIT_RE = re.compile(r"[0-9０-９]+")
ENGLISH_RE = re.compile(r"[A-Za-z]+")


def is_korean(s):
    return bool(KOREAN_RE.fullmatch(s))


def has_korean(s):
    return bool(KOREAN_RE.search(s))


def is_digit_ish(s):
    return bool(DIGIT_RE.search(s)) and not has_korean(s)


def is_english_ish(s):
    return bool(ENGLISH_RE.fullmatch(s.replace(".", "").replace(",", "")))


def suffix(s, n):
    return s[-n:] if len(s) >= n else s


def char_class(ch):
    # Hangul syllable block basic class hint
    code = ord(ch)
    if 0xAC00 <= code <= 0xD7A3:
        return "H"  # Hangul syllable
    if 0x3131 <= code <= 0x318F:
        return "J"  # Jamo
    if ch.isdigit() or 0xFF10 <= ord(ch) <= 0xFF19:
        return "D"
    if ch.isascii() and ch.isalpha():
        return "E"
    if ch in SENT_FINAL:
        return "F"
    if ch in ",;:":
        return "C"
    if ch in "()[]''\"":
        return "B"
    return "O"


def token_features(tok, n_sent, pos1):
    """Features for a single token (position-aware, 1-indexed pos1)."""
    feats = {}
    L = len(tok)
    feats["len"] = L
    feats["len1"] = 1 if L == 1 else 0
    feats["len2"] = 1 if L == 2 else 0
    feats["len3"] = 1 if L == 3 else 0
    feats["len4"] = 1 if L == 4 else 0
    feats["len5p"] = 1 if L >= 5 else 0

    # full token (high cardinality - we hash via dict)
    feats["tok"] = tok
    # last 1..4 chars
    for n in (1, 2, 3, 4):
        feats[f"suf{n}"] = suffix(tok, n)
    # first 1..3 chars
    for n in (1, 2, 3):
        feats[f"pre{n}"] = tok[:n]

    feats["last"] = tok[-1]
    feats["first"] = tok[0]
    feats["cls_last"] = char_class(tok[-1])
    feats["cls_first"] = char_class(tok[0])

    feats["is_final"] = 1 if tok[-1] in SENT_FINAL else 0
    feats["has_comma"] = 1 if "," in tok else 0
    feats["has_dot"] = 1 if "." in tok else 0
    feats["has_bang"] = 1 if "!" in tok else 0
    feats["has_q"] = 1 if "?" in tok else 0
    feats["has_quote"] = 1 if ("'" in tok or '"' in tok) else 0
    feats["has_paren"] = 1 if ("(" in tok or ")" in tok) else 0
    feats["has_slash"] = 1 if "/" in tok else 0
    feats["has_digit"] = 1 if any(c.isdigit() or 0xFF10 <= ord(c) <= 0xFF19 for c in tok) else 0
    feats["all_korean"] = 1 if is_korean(tok) else 0
    feats["has_korean"] = 1 if has_korean(tok) else 0
    feats["all_digit"] = 1 if is_digit_ish(tok) else 0
    feats["all_english"] = 1 if is_english_ish(tok) else 0

    # particle-ish endings (nominal)
    feats["np_end"] = 1 if tok[-1] in NP_END else 0
    feats["vp_end"] = 1 if tok[-1] in VP_END else 0
    feats["vnp_end"] = 1 if tok[-1] in VNP_END else 0
    feats["final_punct"] = 1 if tok[-1] in SENT_FINAL else 0

    # specific high-signal suffixes
    feats["end_da"] = 1 if tok.endswith("다.") or tok.endswith("다!") or tok.endswith("다?") else 0
    feats["end_yo"] = 1 if tok.endswith("요.") or tok.endswith("요!") or tok.endswith("요?") or tok.endswith("요") else 0
    feats["end_eo"] = 1 if tok[-1] in "어아" else 0
    feats["end_neun"] = 1 if tok.endswith("는") else 0
    feats["end_eun"] = 1 if tok.endswith("은") else 0
    feats["end_ga"] = 1 if tok.endswith("가") else 0
    feats["end_i"] = 1 if tok.endswith("이") else 0
    feats["end_eul"] = 1 if tok.endswith("을") else 0
    feats["end_leul"] = 1 if tok.endswith("를") else 0
    feats["end_e"] = 1 if tok.endswith("에") else 0
    feats["end_ro"] = 1 if tok.endswith("로") else 0
    feats["end_eui"] = 1 if tok.endswith("의") else 0
    feats["end_han"] = 1 if tok.endswith("한") else 0
    feats["end_go"] = 1 if tok.endswith("고") else 0
    feats["end_seo"] = 1 if tok.endswith("서") else 0
    feats["end_in"] = 1 if tok.endswith("인") else 0
    feats["end_ji"] = 1 if tok.endswith("지") else 0
    feats["end_myun"] = 1 if tok.endswith("며") or tok.endswith("면") else 0

    # position features
    feats["pos"] = pos1
    feats["pos_norm"] = round(pos1 / max(1, n_sent), 2)
    feats["is_first"] = 1 if pos1 == 1 else 0
    feats["is_last"] = 1 if pos1 == n_sent else 0
    feats["rel_last"] = n_sent - pos1
    feats["rel_last1"] = 1 if n_sent - pos1 == 0 else 0
    feats["rel_last2"] = 1 if n_sent - pos1 == 1 else 0
    feats["rel_last3"] = 1 if n_sent - pos1 >= 2 else 0
    feats["pos2"] = 1 if pos1 == 2 else 0
    feats["pos3"] = 1 if pos1 == 3 else 0
    feats["pos_from_end1"] = 1 if pos1 == n_sent else 0
    feats["pos_from_end2"] = 1 if pos1 == n_sent - 1 else 0
    feats["pos_from_end3"] = 1 if pos1 == n_sent - 2 else 0

    # quadratic-ish normalized position bins
    feats["pos_bin5"] = min(4, int(5 * pos1 / max(1, n_sent)))
    return feats


def pair_features(dep_tok, head_tok, dep_pos1, head_pos1, n_sent, dep_deprel=None,
                  tokens=None):
    """Features for a (dep -> head) candidate pair. dep_pos1 < head_pos1 (head-final)."""
    f = {}
    dist = head_pos1 - dep_pos1
    f["dist"] = dist
    f["dist1"] = 1 if dist == 1 else 0
    f["dist2"] = 1 if dist == 2 else 0
    f["dist3"] = 1 if dist == 3 else 0
    f["dist4"] = 1 if dist == 4 else 0
    f["dist5p"] = 1 if dist >= 5 else 0
    f["dist_log"] = round(np.log1p(dist), 3)
    f["dist_norm"] = round(dist / max(1, n_sent), 3)

    f["dep_tok"] = dep_tok
    f["head_tok"] = head_tok
    for n in (1, 2, 3):
        f[f"dep_suf{n}"] = suffix(dep_tok, n)
        f[f"head_suf{n}"] = suffix(head_tok, n)
    f["dep_last"] = dep_tok[-1]
    f["head_last"] = head_tok[-1]
    f["dep_first"] = dep_tok[0]
    f["head_first"] = head_tok[0]

    f["dep_cls"] = char_class(dep_tok[-1])
    f["head_cls"] = char_class(head_tok[-1])

    f["same_suf2"] = 1 if suffix(dep_tok, 2) == suffix(head_tok, 2) else 0
    f["head_final"] = 1 if head_tok[-1] in SENT_FINAL else 0
    f["head_da"] = 1 if head_tok.endswith("다.") or head_tok.endswith("다!") or head_tok.endswith("다?") else 0
    f["head_yo"] = 1 if head_tok.endswith("요") else 0
    f["head_is_last"] = 1 if head_pos1 == n_sent else 0
    f["head_near_last"] = 1 if head_pos1 >= n_sent - 1 else 0
    f["dep_is_first"] = 1 if dep_pos1 == 1 else 0

    # head token morphological class hints (what kind of phrase head it is)
    f["head_np_end"] = 1 if head_tok[-1] in NP_END else 0
    f["head_vp_end"] = 1 if head_tok[-1] in VP_END else 0
    f["head_vnp_end"] = 1 if head_tok[-1] in VNP_END else 0
    f["head_has_comma"] = 1 if "," in head_tok else 0
    f["head_len5p"] = 1 if len(head_tok) >= 5 else 0
    f["dep_len5p"] = 1 if len(dep_tok) >= 5 else 0

    # context: tokens between dep and head (the intervening span)
    if tokens is not None:
        span = tokens[dep_pos1:head_pos1 - 1]  # strictly between
        f["span_len"] = len(span)
        f["span_len0"] = 1 if len(span) == 0 else 0
        f["span_len1"] = 1 if len(span) == 1 else 0
        f["span_len2p"] = 1 if len(span) >= 2 else 0
        if span:
            f["span_mid_last"] = span[len(span) // 2][-1]
            f["span_first_last"] = span[0][-1]
            f["span_last_last"] = span[-1][-1]
            # any comma in between -> suggests conjunct boundary
            f["span_has_comma"] = 1 if any("," in s for s in span) else 0
            f["span_has_conj"] = 1 if any(s[-1] in "고와과나" for s in span) else 0
        else:
            f["span_mid_last"] = "<NA>"
            f["span_first_last"] = "<NA>"
            f["span_last_last"] = "<NA>"
            f["span_has_comma"] = 0
            f["span_has_conj"] = 0
        # tokens adjacent to dep (left neighbor) and head (right neighbor)
        if dep_pos1 - 2 >= 0:
            f["dep_left_last"] = tokens[dep_pos1 - 2][-1]
        else:
            f["dep_left_last"] = "<BOS>"
        if head_pos1 < n_sent:
            f["head_right_last"] = tokens[head_pos1][-1]
        else:
            f["head_right_last"] = "<EOS>"

    # deprel hint (if provided) -> very strong
    if dep_deprel is not None:
        f["deprel"] = dep_deprel
        # coarse deprel prefix
        f["deprel_coarse"] = dep_deprel.split("_")[0] if "_" in dep_deprel else dep_deprel
        # deprel x distance interaction (some relations prefer close heads)
        f["deprel_dist1"] = f"{dep_deprel}_d{1 if dist == 1 else 0}"
        f["deprel_head_is_last"] = f"{dep_deprel}_hl{f['head_is_last']}"

    # relative position of head in sentence
    f["head_pos_norm"] = round(head_pos1 / max(1, n_sent), 2)
    f["dep_pos_norm"] = round(dep_pos1 / max(1, n_sent), 2)
    # head in the predicate region (last third)?
    f["head_pred_region"] = 1 if head_pos1 >= (2 * n_sent + 2) // 3 else 0
    return f


def featurize_tokens(tokens):
    """Return list of token-feature dicts for a sentence."""
    n = len(tokens)
    return [token_features(t, n, i + 1) for i, t in enumerate(tokens)]


def parse_string_to_pairs(parse_str):
    """parse_str like '2:NP|0:VP' -> list of (head, deprel)."""
    out = []
    for p in str(parse_str).split("|"):
        if p == "" or p == "nan":
            continue
        h, d = p.split(":")
        out.append((int(h), d))
    return out


def pairs_to_parse(pairs):
    return "|".join(f"{h}:{d}" for h, d in pairs)

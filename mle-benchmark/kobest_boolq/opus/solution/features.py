"""Feature engineering for KoBEST BoolQ (sklearn-only, no pretrained models)."""
import re
import numpy as np
import pandas as pd

NEG_PATTERNS = ["않", "없", "아니", "못", "불가", "무관", "반대", "틀리", "거짓", "아닌", "안 "]
HEDGE = ["모든", "항상", "유일", "오직", "만", "전혀", "절대", "최초", "최고", "가장"]
QWORDS = ["나요", "인가", "습니까", "ㄴ가", "인지", "일까", "은가", "는가", "?"]

_sent_split = re.compile(r"(?<=[.!?])\s+|\n+")
_han = re.compile(r"[가-힣]+")
_num = re.compile(r"\d+")


def sentences(p):
    s = [x.strip() for x in _sent_split.split(str(p)) if len(x.strip()) > 1]
    return s if s else [str(p)]


def stems(text, minlen=2):
    """Crude Korean content-token extraction: eojeol prefixes (drop trailing particles)."""
    out = []
    for w in _han.findall(str(text)):
        if len(w) < minlen:
            continue
        out.append(w)
        if len(w) > 2:
            out.append(w[:-1])
        if len(w) > 3:
            out.append(w[:-2])
    return out


def head_stems(text, minlen=2):
    """One representative stem per eojeol (longest prefix >=2)."""
    res = []
    for w in _han.findall(str(text)):
        if len(w) < minlen:
            continue
        res.append(w if len(w) <= 3 else w[:-1])
    return res


def ngrams(text, n):
    t = re.sub(r"\s+", " ", str(text))
    return set(t[i:i + n] for i in range(max(0, len(t) - n + 1)))


def contain_ratio(qtoks, target):
    """fraction of question tokens appearing as substring in target"""
    if not qtoks:
        return 0.0
    tgt = str(target)
    return sum(1 for t in qtoks if t in tgt) / len(qtoks)


def has_any(text, pats):
    t = str(text)
    return float(any(p in t for p in pats))


def count_any(text, pats):
    t = str(text)
    return float(sum(t.count(p) for p in pats))


def build_features(df):
    rows = []
    for p, q in zip(df.paragraph.values, df.question.values):
        sents = sentences(p)
        qh = head_stems(q)
        qh_set = list(dict.fromkeys(qh))
        # remove very generic tokens
        qh_set = [t for t in qh_set if t not in ("있", "하", "이", "그", "저", "것")]

        cov_p = contain_ratio(qh_set, p)
        covs = [contain_ratio(qh_set, s) for s in sents]
        best_i = int(np.argmax(covs)) if covs else 0
        best_s = sents[best_i] if sents else ""
        cov_best = max(covs) if covs else 0.0
        cov_mean = float(np.mean(covs)) if covs else 0.0
        cov_2nd = float(sorted(covs)[-2]) if len(covs) > 1 else 0.0

        f = {}
        f["cov_p"] = cov_p
        f["cov_best"] = cov_best
        f["cov_mean"] = cov_mean
        f["cov_2nd"] = cov_2nd
        f["cov_gap"] = cov_best - cov_2nd
        f["n_qtok"] = len(qh_set)
        f["n_sent"] = len(sents)
        f["len_p"] = len(str(p)) / 100.0
        f["len_q"] = len(str(q)) / 10.0
        f["len_ratio"] = len(str(q)) / max(1, len(str(p)))

        for n in (2, 3, 4):
            qg, pg = ngrams(q, n), ngrams(p, n)
            f[f"jac{n}"] = len(qg & pg) / max(1, len(qg | pg))
            f[f"cont{n}"] = len(qg & pg) / max(1, len(qg))
            bg = ngrams(best_s, n)
            f[f"contb{n}"] = len(qg & bg) / max(1, len(qg))
        # longest common substring between q and p (normalized)
        f["lcs"] = _lcs(re.sub(r"\s+", "", str(q)), re.sub(r"\s+", "", str(p))) / max(1, len(str(q)))
        f["lcs_best"] = _lcs(re.sub(r"\s+", "", str(q)), re.sub(r"\s+", "", best_s)) / max(1, len(str(q)))

        # negation
        qn = has_any(q, NEG_PATTERNS)
        bn = has_any(best_s, NEG_PATTERNS)
        pn = has_any(p, NEG_PATTERNS)
        f["neg_q"] = qn
        f["neg_best"] = bn
        f["neg_p"] = pn
        f["neg_xor_best"] = float(qn != bn)
        f["neg_xor_p"] = float(qn != pn)
        f["neg_q_cnt"] = count_any(q, NEG_PATTERNS)

        # hedge / superlative in question
        f["hedge_q"] = has_any(q, HEDGE)
        f["hedge_p"] = has_any(p, HEDGE)
        f["hedge_xor"] = float(has_any(q, HEDGE) != has_any(p, HEDGE))

        # question form vs statement form
        f["is_question"] = has_any(q, QWORDS)
        f["ends_da"] = float(str(q).strip().rstrip(".").endswith("다"))

        # numbers
        qnums, pnums = set(_num.findall(str(q))), set(_num.findall(str(p)))
        f["n_qnum"] = len(qnums)
        f["num_match"] = len(qnums & pnums) / max(1, len(qnums))
        f["num_miss"] = float(len(qnums) > 0 and len(qnums - pnums) > 0)

        # position of best sentence
        f["best_pos"] = best_i / max(1, len(sents) - 1) if len(sents) > 1 else 0.0
        f["best_is_first"] = float(best_i == 0)

        # rare-token coverage: tokens that are long (more specific)
        longtok = [t for t in qh_set if len(t) >= 3]
        f["cov_long"] = contain_ratio(longtok, p)
        f["n_long"] = len(longtok)
        f["miss_long"] = float(len(longtok) - contain_ratio(longtok, p) * len(longtok))
        f["miss_all"] = len(qh_set) - cov_p * len(qh_set)
        rows.append(f)
    return pd.DataFrame(rows).astype(float)


def _lcs(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best

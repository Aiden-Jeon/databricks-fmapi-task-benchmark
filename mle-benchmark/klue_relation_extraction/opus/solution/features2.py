"""Extended feature builder for KLUE-RE."""
import re
import numpy as np
import pandas as pd
from features import (_find_pair, _etype, HAS_DIGIT, LATIN, HANJA, HANGUL,
                      TITLE_WORDS, FAMILY_WORDS)


def _tok(s):
    return s.split()


def build2(df):
    cols = {k: [] for k in [
        "marked", "between", "near_s", "near_o", "subj", "obj", "pattern",
        "sent", "subj_ex", "obj_ex", "mid_ex", "pat_mid", "edge", "heads",
        "obj_tail", "subj_tail", "marked_w"]}
    num = []
    for sent, s, o in zip(df.sentence.astype(str), df.subject_entity.astype(str),
                          df.object_entity.astype(str)):
        sa, ob = _find_pair(sent, s, o)
        se, oe = sa + len(s), ob + len(o)
        order = 0 if sa <= ob else 1
        if order == 0:
            first, fe, second, se2 = sa, se, ob, oe
        else:
            first, fe, second, se2 = ob, oe, sa, se
        st, ot = _etype(s), _etype(o)
        mid = sent[fe:second]
        lft = sent[max(0, first - 45):first]
        rgt = sent[se2:se2 + 45]
        tag = f"{st}{ot}{order}"
        if order == 0:
            m = f"{lft} @S#{st}# {mid} ^O@{ot}^ {rgt}"
        else:
            m = f"{lft} ^O@{ot}^ {mid} @S#{st}# {rgt}"
        cols["marked"].append(m)
        cols["marked_w"].append(f"T{tag} " + m)
        cols["between"].append(f"[{tag}] {mid}")
        cols["near_s"].append(f"[{st}] " + sent[max(0, sa - 18):sa] + " || " + sent[se:se + 18])
        cols["near_o"].append(f"[{ot}] " + sent[max(0, ob - 18):ob] + " || " + sent[oe:oe + 18])
        cols["subj"].append(s)
        cols["obj"].append(o)
        cols["pattern"].append(tag)
        cols["sent"].append(sent)
        cols["subj_ex"].append("S=" + s)
        cols["obj_ex"].append("O=" + o)
        ms = mid.strip()
        cols["mid_ex"].append("M=" + (ms if len(ms) <= 25 else ms[:12] + "~" + ms[-12:]))
        cols["pat_mid"].append(f"PM={tag}|" + (ms if len(ms) <= 20 else ms[:20]))
        # tokens adjacent to entities
        lt_s = _tok(sent[:sa])[-1:] or ["<B>"]
        rt_s = _tok(sent[se:])[:1] or ["<E>"]
        lt_o = _tok(sent[:ob])[-1:] or ["<B>"]
        rt_o = _tok(sent[oe:])[:1] or ["<E>"]
        cols["edge"].append(f"LS={lt_s[0]} RS={rt_s[0]} LO={lt_o[0]} RO={rt_o[0]}")
        mt = _tok(mid)
        cols["heads"].append(
            "MF=" + (mt[0] if mt else "<>") + " ML=" + (mt[-1] if mt else "<>") +
            " MN=" + str(min(len(mt), 6)) + " " + tag)
        cols["obj_tail"].append(f"ot1={o[-1:]} ot2={o[-2:]} oh1={o[:1]} oh2={o[:2]} ol={min(len(o),10)}")
        cols["subj_tail"].append(f"st1={s[-1:]} st2={s[-2:]} sh2={s[:2]} sl={min(len(s),10)}")
        dist = second - fe
        ctx = mid + " " + rgt[:25]
        wide = lft[-25:] + mid + rgt[:25]
        num.append([
            len(s), len(o), len(sent), dist, dist / (len(sent) + 1.0), order,
            float(st == "PER"), float(st == "ORG"), float(st == "LOC"),
            float(st == "DAT"), float(st == "NOH"), float(st == "MSC"),
            float(ot == "PER"), float(ot == "ORG"), float(ot == "LOC"),
            float(ot == "DAT"), float(ot == "NOH"), float(ot == "MSC"),
            float(bool(HAS_DIGIT.search(o))), float(bool(HAS_DIGIT.search(s))),
            float(bool(LATIN.search(o))), float(bool(HANJA.search(o))),
            float(bool(HANJA.search(mid))), float(bool(LATIN.search(s))),
            float(s in o or o in s),
            float(any(w in ctx for w in TITLE_WORDS)),
            float(any(w in ctx for w in FAMILY_WORDS)),
            float(any(w in mid for w in TITLE_WORDS)),
            float(any(w in mid for w in FAMILY_WORDS)),
            float(len(ms) == 0), float(ms in (",", "(", ")", "·", "-", "~")),
            float("(" in mid), float(")" in mid), float("~" in mid),
            float("출생" in wide), float("사망" in wide), float("졸업" in wide),
            float("설립" in wide or "창립" in wide or "창설" in wide),
            float("본사" in wide or "소재" in wide or "위치" in wide),
            float("취임" in wide or "임명" in wide), float("소속" in wide),
            float("출신" in wide), float("태어" in wide), float("해체" in wide or "폐지" in wide),
            float("결혼" in wide), float("아들" in wide or "딸" in wide),
            float("회장" in wide or "대표" in wide or "사장" in wide),
            float("대통령" in wide), float("감독" in wide), float("선수" in wide),
            float("교수" in wide), float("의원" in wide),
            len(mid), len(mt), len(_tok(sent)),
            sent.count(","), sent.count("("),
            first / (len(sent) + 1.0), se2 / (len(sent) + 1.0),
        ])
    return pd.DataFrame(cols), np.asarray(num, dtype=np.float32)


VEC_SPECS = [
    ("marked", dict(analyzer="char_wb", ngram_range=(2, 4), min_df=3, sublinear_tf=True, max_features=350000)),
    ("marked_w", dict(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
    ("between", dict(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=350000)),
    ("near_s", dict(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)),
    ("near_o", dict(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)),
    ("subj", dict(analyzer="char_wb", ngram_range=(1, 3), min_df=2, sublinear_tf=True)),
    ("obj", dict(analyzer="char_wb", ngram_range=(1, 3), min_df=2, sublinear_tf=True)),
    ("sent", dict(analyzer="word", ngram_range=(1, 1), min_df=3, sublinear_tf=True)),
    ("pattern", dict(analyzer="word", min_df=1, token_pattern=r"\S+")),
    ("subj_ex", dict(analyzer="word", min_df=1, token_pattern=r"\S+")),
    ("obj_ex", dict(analyzer="word", min_df=1, token_pattern=r"\S+")),
    ("mid_ex", dict(analyzer="word", min_df=1, token_pattern=r"\S+")),
    ("pat_mid", dict(analyzer="word", min_df=1, token_pattern=r"\S+")),
    ("edge", dict(analyzer="word", min_df=2, token_pattern=r"\S+")),
    ("heads", dict(analyzer="word", min_df=2, token_pattern=r"\S+")),
    ("obj_tail", dict(analyzer="word", min_df=2, token_pattern=r"\S+")),
    ("subj_tail", dict(analyzer="word", min_df=2, token_pattern=r"\S+")),
]

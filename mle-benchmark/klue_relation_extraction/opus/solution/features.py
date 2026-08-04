"""Feature engineering for KLUE-RE (classical ML, CPU only)."""
import re
import numpy as np
import pandas as pd

DATE_RE = re.compile(r"^\d{1,4}\s*(년|월|일|년대|세기)?$")
NUM_RE = re.compile(r"^[\d,\.]+\s*(명|개|여명|만명|만|억|원|%|년|월|일)?$")
HAS_DIGIT = re.compile(r"\d")
HANGUL = re.compile(r"[\uac00-\ud7a3]")
LATIN = re.compile(r"[A-Za-z]")
HANJA = re.compile(r"[\u4e00-\u9fff]")

ORG_SUF = ["사", "그룹", "당", "대학교", "대학", "학교", "청", "부", "원", "회", "단",
           "팀", "국", "위원회", "협회", "재단", "공사", "은행", "교회", "연구소",
           "센터", "협의회", "조합", "군", "시", "도", "구", "읍", "면", "동", "리",
           "주", "현", "성", "공화국", "왕국", "제국", "연맹", "리그", "클럽", "방송",
           "신문", "일보", "TV", "컴퍼니", "홀딩스", "전자", "화학", "중공업", "생명",
           "증권", "카드", "보험", "건설", "산업", "물산", "공업", "제철", "통신"]
LOC_SUF = ["시", "도", "군", "구", "읍", "면", "동", "리", "주", "현", "성", "국",
           "공화국", "왕국", "제국", "지방", "지역", "반도", "산", "강", "호", "섬",
           "대륙", "만", "해", "역", "공항", "로", "길", "가"]
TITLE_WORDS = ["대통령", "회장", "사장", "총리", "장관", "의원", "감독", "선수", "교수",
               "대표", "이사", "위원장", "부장", "차장", "과장", "실장", "국장", "본부장",
               "단장", "대표이사", "총장", "원장", "청장", "지사", "시장", "군수", "구청장",
               "부회장", "부사장", "상무", "전무", "고문", "위원", "의장", "코치", "작가",
               "가수", "배우", "아나운서", "기자", "변호사", "판사", "검사", "의사", "박사",
               "교사", "목사", "신부", "스님", "왕", "황제", "공작", "백작", "후작"]
FAMILY_WORDS = ["아들", "딸", "아버지", "어머니", "부친", "모친", "형", "동생", "누나",
                "언니", "오빠", "남편", "아내", "부인", "남편", "배우자", "결혼", "혼인",
                "장남", "차남", "장녀", "차녀", "조카", "삼촌", "이모", "고모", "사촌",
                "손자", "손녀", "며느리", "사위", "형제", "자매", "남매", "부부", "슬하"]


def _find_pair(sent, s, o):
    """Choose the (subj, obj) occurrence pair minimizing distance."""
    si = [m.start() for m in re.finditer(re.escape(s), sent)] or [0]
    oi = [m.start() for m in re.finditer(re.escape(o), sent)] or [0]
    best = None
    for a in si:
        for b in oi:
            d = abs(a - b)
            if best is None or d < best[0]:
                best = (d, a, b)
    return best[1], best[2]


def _etype(e):
    """Heuristic entity type."""
    if DATE_RE.match(e) or re.match(r"^\d{4}년", e) or re.search(r"\d+년\s*\d+월", e):
        return "DAT"
    if NUM_RE.match(e):
        return "NOH"
    if HAS_DIGIT.search(e) and len(e) <= 12:
        return "NUM"
    if any(e.endswith(x) for x in ORG_SUF) and len(e) >= 3:
        return "ORG"
    if any(e.endswith(x) for x in LOC_SUF) and len(e) >= 3:
        return "LOC"
    if 2 <= len(e) <= 4 and HANGUL.search(e) and not LATIN.search(e):
        return "PER"
    return "MSC"


def build(df):
    out = {}
    marked, between, left, right, near_s, near_o = [], [], [], [], [], []
    subj, obj, subj_ctx, obj_ctx, pattern = [], [], [], [], []
    num = []
    for sent, s, o in zip(df.sentence.astype(str), df.subject_entity.astype(str),
                          df.object_entity.astype(str)):
        sa, ob = _find_pair(sent, s, o)
        se, oe = sa + len(s), ob + len(o)
        if sa <= ob:
            first, fe, second, se2 = sa, se, ob, oe
            order = 0
        else:
            first, fe, second, se2 = ob, oe, sa, se
            order = 1
        st = _etype(s)
        ot = _etype(o)
        mid = sent[fe:second]
        lft = sent[max(0, first - 40):first]
        rgt = sent[se2:se2 + 40]
        if order == 0:
            m = f"{lft} @S#{st}# {mid} ^O@{ot}^ {rgt}"
        else:
            m = f"{lft} ^O@{ot}^ {mid} @S#{st}# {rgt}"
        marked.append(m)
        between.append(f"[{st}|{ot}|{order}] {mid}")
        left.append(lft)
        right.append(rgt)
        near_s.append(sent[max(0, sa - 15):sa] + " || " + sent[se:se + 15])
        near_o.append(sent[max(0, ob - 15):ob] + " || " + sent[oe:oe + 15])
        subj.append(s)
        obj.append(o)
        subj_ctx.append(f"{st} {s}")
        obj_ctx.append(f"{ot} {o}")
        pattern.append(f"{st}_{ot}_{order}")
        dist = second - fe
        ctx = mid + " " + rgt[:20]
        num.append([
            len(s), len(o), len(sent), dist, dist / (len(sent) + 1.0), order,
            float(st == "PER"), float(st == "ORG"), float(st == "LOC"),
            float(st == "DAT"), float(st == "NOH"), float(st == "MSC"),
            float(ot == "PER"), float(ot == "ORG"), float(ot == "LOC"),
            float(ot == "DAT"), float(ot == "NOH"), float(ot == "MSC"),
            float(bool(HAS_DIGIT.search(o))), float(bool(HAS_DIGIT.search(s))),
            float(bool(LATIN.search(o))), float(bool(HANJA.search(o))),
            float(bool(HANJA.search(mid))),
            float(s in o or o in s),
            float(any(w in ctx for w in TITLE_WORDS)),
            float(any(w in ctx for w in FAMILY_WORDS)),
            float(any(w in mid for w in TITLE_WORDS)),
            float(any(w in mid for w in FAMILY_WORDS)),
            float("(" in mid and ")" in rgt[:5] or mid.strip() in ("(", "(,", ",")),
            float(len(mid.strip()) == 0),
            float(mid.strip() in (",", "(", ")", "·", "-", "~")),
            float("출생" in ctx), float("사망" in ctx), float("졸업" in ctx),
            float("설립" in ctx or "창립" in ctx or "창설" in ctx),
            float("본사" in ctx or "위치" in ctx or "소재" in ctx),
            float("취임" in ctx or "임명" in ctx), float("소속" in ctx),
            float("출신" in ctx), float("태어" in ctx),
            len(mid), len(mid.split()),
        ])
    out["marked"] = marked
    out["between"] = between
    out["left"] = left
    out["right"] = right
    out["near_s"] = near_s
    out["near_o"] = near_o
    out["subj"] = subj
    out["obj"] = obj
    out["subj_ctx"] = subj_ctx
    out["obj_ctx"] = obj_ctx
    out["pattern"] = pattern
    X = pd.DataFrame(out)
    return X, np.asarray(num, dtype=np.float32)

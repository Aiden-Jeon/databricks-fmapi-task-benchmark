"""Feature engineering for KLUE-RE relation extraction (CPU / sklearn only)."""
import re
import numpy as np
import pandas as pd

S_MARK = "\u24c8"  # circled S
O_MARK = "\u24c4"  # circled O

RE_YEAR = re.compile(r"^\d{3,4}\s*년")
RE_DATE = re.compile(r"\d+\s*월|\d+\s*일|\d{3,4}\s*년")
RE_NUM = re.compile(r"^[\d,\.]+\s*(명|개|여명|만명|억|만|천|백|퍼센트|%|원|달러|년|월|일|위|회|차|호|배|km|㎞|m|kg)?$")
RE_HANJA = re.compile(r"[\u4e00-\u9fff]")
RE_LATIN = re.compile(r"[A-Za-z]")
RE_HANGUL = re.compile(r"[\uac00-\ud7a3]")

ORG_SUF = [
    "회사", "주식회사", "그룹", "은행", "공사", "공단", "협회", "재단", "학회", "위원회", "협회",
    "당", "정당", "연구소", "연구원", "센터", "청", "부", "처", "원", "국", "실", "과", "팀",
    "대학교", "대학", "고등학교", "중학교", "초등학교", "학교", "병원", "교회", "사찰", "신문",
    "방송", "방송국", "TV", "은행", "증권", "보험", "카드", "전자", "화학", "중공업", "건설",
    "항공", "해운", "물산", "산업", "제철", "통신", "텔레콤", "생명", "홀딩스", "코퍼레이션",
    "Inc", "Corp", "Ltd", "Co", "FC", "구단", "리그", "연맹", "노조", "조합", "군", "사단",
]
LOC_SUF = [
    "시", "군", "구", "읍", "면", "리", "동", "도", "주", "현", "성", "국", "왕국", "공화국",
    "산", "강", "호", "섬", "반도", "만", "해", "양", "대륙", "지방", "지역", "마을", "역",
    "공항", "항", "로", "길", "가", "대로",
]
TITLE_SUF = [
    "장", "사장", "회장", "대표", "이사", "부장", "과장", "팀장", "교수", "총장", "학장",
    "감독", "코치", "선수", "의원", "장관", "차관", "대통령", "총리", "시장", "군수", "지사",
    "위원", "위원장", "국장", "실장", "본부장", "센터장", "원장", "소장", "대사", "판사",
    "검사", "변호사", "의사", "기자", "작가", "가수", "배우", "화가", "시인", "박사", "왕",
    "황제", "제왕", "후보", "주장", "투수", "포수", "내야수", "외야수", "공격수", "수비수",
    "미드필더", "골키퍼", "목사", "신부", "스님", "주교", "추기경",
]
PER_HINT = ["씨", "군", "양", "님"]

COUNTRY = ["한국", "대한민국", "미국", "일본", "중국", "영국", "프랑스", "독일", "러시아",
           "북한", "조선", "이탈리아", "스페인", "캐나다", "호주", "인도", "브라질", "멕시코"]

RELIGION = ["기독교", "천주교", "불교", "이슬람", "개신교", "유교", "가톨릭", "힌두교",
            "성공회", "장로교", "감리교", "천도교", "원불교", "이슬람교", "유대교"]


def _suffix_flags(s, suffixes):
    return 1.0 if any(s.endswith(x) for x in suffixes) else 0.0


def entity_type_guess(s):
    """Coarse entity-type guess -> one of DAT NOH PER ORG LOC POH."""
    s = s.strip()
    if RE_DATE.search(s) and len(s) <= 20:
        return "DAT"
    if RE_NUM.match(s):
        return "NOH"
    if _suffix_flags(s, ORG_SUF):
        return "ORG"
    if _suffix_flags(s, LOC_SUF) and len(s) <= 8:
        return "LOC"
    if s in COUNTRY:
        return "LOC"
    if _suffix_flags(s, TITLE_SUF):
        return "POH"
    # Korean person names: 2-4 hangul chars, no spaces
    if RE_HANGUL.search(s) and len(s) <= 4 and " " not in s and not RE_LATIN.search(s):
        return "PER"
    return "POH"


def find_span(sent, ent):
    """Best occurrence of ent in sent: prefer one bounded by non-hangul."""
    idxs = []
    start = 0
    while True:
        i = sent.find(ent, start)
        if i < 0:
            break
        idxs.append(i)
        start = i + 1
    if not idxs:
        return -1, -1
    return idxs[0], idxs[0] + len(ent)


def _tok_window(sent, a, b, mark, k=2):
    """Whitespace-token window around span [a,b) with the entity replaced by mark."""
    pre = sent[:a].split(" ")
    post = sent[b:].split(" ")
    lt = pre[-1] if pre else ""
    rt = post[0] if post else ""
    lprev = " ".join([t for t in pre[-(k + 1):-1] if t])
    rnext = " ".join([t for t in post[1:1 + k] if t])
    return f"{lprev} {lt}{mark}{rt} {rnext}".strip()


def build_text_fields(df):
    n = len(df)
    marked, between, left, right, subj, obj = [], [], [], [], [], []
    tmarked, xbtw, sctx, octx = [], [], [], []
    num = np.zeros((n, 34), dtype=np.float32)
    sents = df["sentence"].values
    subs = df["subject_entity"].values
    objs = df["object_entity"].values

    for i in range(n):
        sent = str(sents[i])
        se = str(subs[i]).strip()
        oe = str(objs[i]).strip()
        ss, sesp = find_span(sent, se)
        os_, oe_sp = find_span(sent, oe)
        if ss < 0:
            ss, sesp = 0, 0
        if os_ < 0:
            os_, oe_sp = 0, 0

        st = entity_type_guess(se)
        ot = entity_type_guess(oe)

        # build marked sentence (handle overlap by ordering)
        spans = sorted([(ss, sesp, S_MARK), (os_, oe_sp, O_MARK)])
        out = []
        prev = 0
        ok = True
        if spans[0][1] > spans[1][0]:
            ok = False
        if ok:
            for a, b, m in spans:
                out.append(sent[prev:a])
                out.append(" " + m + " ")
                prev = b
            out.append(sent[prev:])
            ms = "".join(out)
        else:
            ms = sent.replace(se, " " + S_MARK + " ").replace(oe, " " + O_MARK + " ")
        marked.append(ms)
        tmarked.append(
            ms.replace(S_MARK, S_MARK + st).replace(O_MARK, O_MARK + ot)
        )

        # between text (with direction marker)
        if ss <= os_:
            btw = S_MARK + st + " " + sent[sesp:os_] + " " + O_MARK + ot
            lo, hi = ss, oe_sp
        else:
            btw = O_MARK + ot + " " + sent[oe_sp:ss] + " " + S_MARK + st
            lo, hi = os_, sesp
        btw = btw[:200]
        between.append(btw)
        # explicit entity-type x pattern cross features
        tp = st + ot
        xbtw.append(" ".join(tp + "|" + t for t in btw.split() if t)[:600])
        sctx.append(_tok_window(sent, ss, sesp, " " + S_MARK + st + " "))
        octx.append(_tok_window(sent, os_, oe_sp, " " + O_MARK + ot + " "))
        left.append(sent[max(0, lo - 25):lo])
        right.append(sent[hi:hi + 25])
        subj.append(se)
        obj.append(oe)

        L = max(len(sent), 1)
        dist = abs(os_ - ss)
        f = [
            len(sent) / 100.0,
            len(se) / 10.0,
            len(oe) / 10.0,
            dist / 50.0,
            min(dist, 100) / 100.0,
            1.0 if ss < os_ else 0.0,
            ss / L,
            os_ / L,
            1.0 if RE_YEAR.match(oe) else 0.0,
            1.0 if RE_DATE.search(oe) else 0.0,
            1.0 if RE_NUM.match(oe) else 0.0,
            1.0 if RE_DATE.search(se) else 0.0,
            1.0 if RE_HANJA.search(oe) else 0.0,
            1.0 if RE_HANJA.search(se) else 0.0,
            1.0 if RE_LATIN.search(oe) else 0.0,
            1.0 if RE_LATIN.search(se) else 0.0,
            _suffix_flags(oe, ORG_SUF),
            _suffix_flags(se, ORG_SUF),
            _suffix_flags(oe, LOC_SUF),
            _suffix_flags(se, LOC_SUF),
            _suffix_flags(oe, TITLE_SUF),
            _suffix_flags(se, TITLE_SUF),
            1.0 if oe in COUNTRY else 0.0,
            1.0 if oe in RELIGION else 0.0,
            1.0 if " " in oe else 0.0,
            1.0 if " " in se else 0.0,
            1.0 if se in oe or oe in se else 0.0,
            1.0 if dist <= 2 else 0.0,
            1.0 if dist <= 6 else 0.0,
            1.0 if dist > 40 else 0.0,
            sent.count(se) / 3.0,
            sent.count(oe) / 3.0,
            1.0 if "(" in sent[max(0, min(ss, os_) - 2):max(sesp, oe_sp) + 2] else 0.0,
            1.0 if sent[hi:hi + 2].startswith("이다") else 0.0,
        ]
        num[i] = f

    return pd.DataFrame({
        "pair": [str(a).strip() + "\u2016" + str(b).strip() for a, b in zip(subs, objs)],
        "marked": marked,
        "tmarked": tmarked,
        "between": between,
        "xbtw": xbtw,
        "sctx": sctx,
        "octx": octx,
        "left": left,
        "right": right,
        "subj": subj,
        "obj": obj,
        "stype": [entity_type_guess(str(x).strip()) for x in subs],
        "otype": [entity_type_guess(str(x).strip()) for x in objs],
        "sentence": sents,
    }), num

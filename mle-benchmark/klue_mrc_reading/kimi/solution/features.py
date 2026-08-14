"""Question analysis and context feature extraction."""
import re
from collections import Counter

from common import normalize_text, simple_tokenize, content_words, strip_ws

# Question-type patterns (rule-based classifier). Order matters: first match
# wins for some overlap cases.
QTYPE_PATTERNS = [
    ("NUM", [
        r"몇\s*(명|개|번|년|월|일|시|분|초|가지|곳|차례|회|종류|권|마리|대|척|채|인|점|km|m|원|달러|배|%|퍼센트)",
        r"얼마(나)?",
        r"(총|몇)\s",
        r"(수는|수가|규모|금액|연봉|나이|연령|길이|무게|높이|속도|거리|면적|인구|비율|점수|가격|숫자|기간|온도)",
        r"(\d|숫자|몇)",
    ]),
    ("DATE", [
        r"(언제|어느\s*(해|때|시기|시점|날|연도|년도))",
        r"(몇\s*년|어느\s*년|몇\s*월|몇\s*일|기간은|시기는|날짜는)",
        r"(해는|년은|월은|일은)",
    ]),
    ("PERSON", [
        r"(누구|누가|인물|사람|저자|작가|발명가|대통령|왕|감독|선수|배우|가수|과학자|의사|교수|연구자|설립자|창업자|대표|회장|사장|주인공)",
        r"(사람은|인물은|인물이|사람이)",
    ]),
    ("PLACE", [
        r"(어디|어느\s*(곳|나라|국가|도시|지역|장소|마을|섬|강|산|대륙|주|도|시|군|구|동|호수|바다|해협|공원|건물|대학|학교))",
        r"(장소는|위치는|곳은|나라는|국가는|도시는|지역은|어디에|어디에서|어디로)",
    ]),
    ("ORG", [
        r"(회사|기업|단체|조직|기관|학교|대학|정당|팀|그룹|협회|위원회|정부|부처|언론사|신문사|방송사|연구소|병원|은행|브랜드|제조사|업체)(는|이|을|에서)?\s*($|\?|은|는|이|가)",
        r"(어느\s*(회사|기업|단체|조직|기관|학교|대학|팀|협회|신문|잡지|방송|브랜드|나라))",
    ]),
    ("TITLE", [
        r"(제목|작품|논문|책|소설|영화|드라마|노래|앨범|곡|시|만화|게임|프로그램|기사|보고서|서적|명칭|이름은|이름을|제목은)",
    ]),
    ("DEF", [
        r"(무엇|뭐|뭘|무슨)(을|를|이|가|은|는|인가|인가요|입니까|일까)?",
    ]),
]

_Q_RE = [(t, [re.compile(p) for p in pats]) for t, pats in QTYPE_PATTERNS]


def qtype(question: str) -> str:
    q = normalize_text(question)
    for t, res in _Q_RE:
        for r in res:
            if r.search(q):
                return t
    return "OTHER"


STOPWORDS = set([
    "것", "수", "등", "때", "중", "후", "전", "이", "그", "저", "및", "또는",
    "그리고", "하지만", "그러나", "있다", "없다", "하다", "되다", "이다",
    "같다", "위해", "대해", "통해", "따라", "대한", "무엇", "어디", "누구",
    "언제", "어떤", "어느", "몇", "왜", "어떻게", "얼마나", "가장",
])


def question_keywords(question: str):
    toks = content_words(question)
    return [t for t in toks if t not in STOPWORDS and len(t) >= 1]


def context_word_counts(context: str):
    return Counter(content_words(context))

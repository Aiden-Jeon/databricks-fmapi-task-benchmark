"""Korean text normalization helpers for STS.

- Korean number words -> digits ("네 명" -> "4 명", "열세" -> "13")
- Lightweight particle (josa) stripping for word tokens
"""
import re

NUMW = {
    "영": 0, "일": 1, "이": 2, "삼": 3, "사": 4, "오": 5, "육": 6, "칠": 7,
    "팔": 8, "구": 9, "십": 10, "한": 1, "두": 2, "세": 3, "네": 4, "다섯": 5,
    "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9, "열": 10, "스무": 20,
    "서른": 30, "마흔": 40, "쉰": 50, "예순": 60, "일흔": 70, "여든": 80,
    "아흔": 90,
}

PARTICLES = sorted(
    ["에서는", "에게는", "까지는", "부터는", "으로써", "으로서", "이라도", "이라는",
     "으로는", "에도", "에서", "에게", "께서", "으로", "로써", "로서", "이라",
     "이며", "이고", "은", "는", "이", "가", "을", "를", "에", "의", "와", "과",
     "도", "로", "만", "나", "이나", "까지", "부터", "처럼", "보다", "랑",
     "이랑", "하고"],
    key=len, reverse=True,
)

_CLEAN_RE = re.compile(r"[^\w\s가-힣]")


def strip_particle(tok):
    for p in PARTICLES:
        if tok.endswith(p) and len(tok) > len(p):
            return tok[: -len(p)]
    return tok


def norm_num(tok):
    if re.fullmatch(r"\d+", tok):
        return tok
    if tok in NUMW:
        return str(NUMW[tok])
    # compound tens: "열세"->13, "스물다섯"->25, ...
    for ten in ["열", "스물", "서른", "마흔", "쉰", "예순", "일흔", "여든", "아흔"]:
        if tok.startswith(ten):
            rest = tok[len(ten):]
            base = NUMW.get(ten)
            r = NUMW.get(rest, 0) if rest else 0
            if base is not None:
                return str(base + r)
    return tok


def tokens(text, stem=False):
    """tokenize with number normalization; optionally strip particles."""
    toks = _CLEAN_RE.sub(" ", str(text).lower()).split()
    if stem:
        toks = [strip_particle(t) for t in toks]
    return [norm_num(t) for t in toks]


def normalize_number_text(text):
    """number-normalized plain text (for char-level tfidf)."""
    toks = _CLEAN_RE.sub(" ", str(text).lower()).split()
    return " ".join(norm_num(t) for t in toks)

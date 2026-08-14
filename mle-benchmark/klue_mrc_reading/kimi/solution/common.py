"""Shared utilities for KLUE-MRC style extractive QA (char-F1 metric)."""
import re
import unicodedata

_WS = re.compile(r"\s+")

# Sentence splitter for Korean news/wiki text: split after sentence-ending
# punctuation followed by whitespace, also split on newlines.
_SENT_SPLIT = re.compile(r"(?<=[.!?。？！])\s+|(?<=[다요음함]\.)\s*|\n+")

_CHAR_NORM_TABLE = {ord("·"): " ", ord("ㆍ"): " ", ord("・"): " "}


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFC", str(s))
    s = s.translate(_CHAR_NORM_TABLE)
    return s


def strip_ws(s: str) -> str:
    """Remove all whitespace (used for containment checks / len)."""
    return _WS.sub("", s)


def simple_tokenize(s: str):
    """Very light Korean-aware tokenizer: split on whitespace, keep
    punctuation attached except trailing sentence punctuation."""
    s = normalize_text(s)
    toks = []
    for w in _WS.split(s.strip()):
        if not w:
            continue
        toks.append(w)
    return toks


def content_words(s: str):
    """Return list of content-ish tokens: remove particles/punct, keep stems
    by stripping common Korean particles from the end (rough)."""
    out = []
    for t in simple_tokenize(s):
        t = t.strip(".,!?\"'“”‘’()<>《》〈〉·…~—:;[]{}-")
        if not t:
            continue
        # strip common particles (은/는/이/가/을/를/에/의/와/과/도/로/으로/에서/부터/까지/만/보다/처럼/에게/한테/께서/랑/이랑/하고)
        t2 = re.sub(
            r"(께서|에서|부터|까지|으로|에게|한테|처럼|이랑|하고|보다|은|는|이|가|을|를|에|의|와|과|도|로|만|랑|나|이나)$",
            "",
            t,
        )
        if len(t2) >= 1:
            out.append(t2)
    return out


def split_sentences(context: str):
    """Split context into sentences; return list of (start_char, sentence)."""
    context = normalize_text(context)
    spans = []
    for m in re.finditer(r"[^.!?\n]+(?:[.!?。]+[\"'”’)]*|$)", context):
        s = m.group(0)
        if s.strip():
            start = m.start()
            spans.append((start, s))
    # Fallback: whole context
    if not spans:
        spans = [(0, context)]
    return spans


def char_f1(pred: str, gold: str) -> float:
    """Character-level F1 between prediction and gold (multiset of chars,
    whitespace removed). Empty/empty -> 1.0, empty/non-empty -> 0."""
    p = strip_ws(normalize_text(pred))
    g = strip_ws(normalize_text(gold))
    if len(p) == 0 and len(g) == 0:
        return 1.0
    if len(p) == 0 or len(g) == 0:
        return 0.0
    from collections import Counter

    pc, gc = Counter(p), Counter(g)
    common = sum((pc & gc).values())
    if common == 0:
        return 0.0
    prec = common / len(p)
    rec = common / len(g)
    return 2 * prec * rec / (prec + rec)

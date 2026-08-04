"""Shared utilities: normalization, char-F1 metric, tokenization, candidate generation."""
import re
import unicodedata
from collections import Counter

# ---------------- metric (KorQuAD 1.0 style character F1) ----------------
_PUNC = set(
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    "\u2018\u2019\u201c\u201d\u2013\u2014\u00b7\u300c\u300d\u300e\u300f"
    "\u3008\u3009\u300a\u300b\uff08\uff09\u2026\u00ab\u00bb"
)


def normalize_answer(s):
    s = s.lower()
    s = re.sub(r"['\"\u2018\u2019\u201c\u201d]", " ", s)
    s = "".join(ch for ch in s if ch not in _PUNC)
    return " ".join(s.split())


def char_bag(s):
    return [c for tok in normalize_answer(s).split() for c in tok]


def char_f1(pred, gold):
    p, g = char_bag(pred), char_bag(gold)
    if len(p) == 0 or len(g) == 0:
        return float(len(p) == len(g))
    common = Counter(p) & Counter(g)
    ns = sum(common.values())
    if ns == 0:
        return 0.0
    prec = ns / len(p)
    rec = ns / len(g)
    return 2 * prec * rec / (prec + rec)


# ---------------- tokenization ----------------
_TOK_RE = re.compile(r"\S+")

# Korean particles / endings that are commonly attached to an answer span in text
JOSA = [
    "이라고도", "으로부터", "에서부터", "이라는", "라는", "이라고", "라고", "에서는",
    "에게서", "으로는", "로부터", "부터", "까지", "에게", "한테", "에서", "으로",
    "이라", "라며", "이며", "이고", "처럼", "보다", "만큼", "조차", "마저", "밖에",
    "이나", "나마", "라도", "이란", "이든", "와의", "과의", "에는", "에도", "으론",
    "은", "는", "이", "가", "을", "를", "의", "에", "와", "과", "로", "도", "만",
    "랑", "며", "요", "라", "야",
]
JOSA_SET = sorted(set(JOSA), key=len, reverse=True)

TRAIL_PUNC = "".join(sorted(_PUNC)) + " \t\n"
LEAD_PUNC = "".join(sorted(_PUNC)) + " \t\n"


def tokens_with_pos(text):
    """Return list of (token, start, end)."""
    return [(m.group(0), m.start(), m.end()) for m in _TOK_RE.finditer(text)]


def strip_span(text, s, e):
    """Strip leading/trailing punctuation from the span [s,e)."""
    while s < e and text[s] in LEAD_PUNC:
        s += 1
    while e > s and text[e - 1] in TRAIL_PUNC:
        e -= 1
    return s, e


def span_variants(text, s, e):
    """Generate end-boundary variants of a span, stripping trailing josa/punct."""
    out = []
    s0, e0 = strip_span(text, s, e)
    if e0 <= s0:
        return out
    out.append((s0, e0))
    # remove trailing josa
    surface = text[s0:e0]
    for j in JOSA_SET:
        if len(surface) > len(j) + 1 and surface.endswith(j):
            s1, e1 = strip_span(text, s0, e0 - len(j))
            if e1 > s1:
                out.append((s1, e1))
            break
    # bare 1-char trailing strip (catch-all for unlisted endings)
    if e0 - s0 > 2:
        s2, e2 = strip_span(text, s0, e0 - 1)
        if e2 > s2:
            out.append((s2, e2))
    # closing paren/quote-balanced version: if span contains an opening quote,
    # try trimming to before it
    return list(dict.fromkeys(out))


# ---------------- sentence splitting ----------------
_SENT_END = re.compile(r"(?<=[.!?\n])\s+|(?<=다\.)|(?<=요\.)")


def split_sentences(text):
    """Return list of (start, end) sentence spans (approximate)."""
    spans = []
    start = 0
    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        if ch in ".!?\n":
            # avoid splitting decimals / abbreviations like "1.5"
            if ch == "." and i + 1 < n and text[i + 1].isdigit() and i > 0 and text[i - 1].isdigit():
                i += 1
                continue
            j = i + 1
            while j < n and text[j] in " \t\n\"'\u2019\u201d)":
                j += 1
            if j - start > 5:
                spans.append((start, j))
                start = j
            i = j
            continue
        i += 1
    if start < n:
        spans.append((start, n))
    if not spans:
        spans = [(0, n)]
    # merge very short sentences into the previous one
    merged = []
    for s, e in spans:
        if merged and e - s < 20:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


# ---------------- lexical features ----------------
def ngrams_of(s, n):
    s = re.sub(r"\s+", "", s)
    return [s[i:i + n] for i in range(max(0, len(s) - n + 1))]


STOP_Q = set(
    "무엇 무슨 어디 어느 언제 누구 누가 어떤 어떻게 얼마 몇 왜 것 곳 사람 이름 때 년 후"
    " 대해 대한 대하여 관해 관한 있는 있던 하는 한 된 되는 위해 위한 통해 그 이 저 및".split()
)


def content_words(text, minlen=2):
    """Rough content-word extraction: strip trailing josa from each token."""
    out = []
    for tok in normalize_answer(text).split():
        base = tok
        for j in JOSA_SET:
            if len(base) > len(j) + 1 and base.endswith(j):
                base = base[: -len(j)]
                break
        if base and base not in STOP_Q:
            out.append(base)
        if tok != base and tok not in STOP_Q:
            out.append(tok)
    return out

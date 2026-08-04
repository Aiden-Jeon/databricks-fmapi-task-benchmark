"""Common utilities for t9_korquad span extraction."""
import re
import pandas as pd
import numpy as np

# Korean characters / punctuation ranges for boundary detection
KSPLIT = set(' \t\n.,;:()[]{}<>《》〈〉「」『』""\'\'·・-−—/|')
# Particles that often follow answers
JPARTICLES = set(['은', '는', '이', '가', '을', '를', '의', '에', '에서', '로', '으로',
                  '와', '과', '도', '만', '조차', '마저', '이나', '나', '에게', '께',
                  '한테', '보다', '처럼', '보다', '이라는', '라는', '이라며', '라며',
                  '이며', '며', '이고', '고', '이지만', '지만'])

def normalize(s):
    return re.sub(r'\s+', '', str(s))

def char_f1(pred, gold):
    """Character-level F1 used by the metric (whitespace removed, multiset F1)."""
    p = normalize(pred); g = normalize(gold)
    if len(p) == 0 and len(g) == 0:
        return 1.0
    if len(p) == 0 or len(g) == 0:
        return 0.0
    # multiset via Counter
    from collections import Counter
    pc = Counter(p); gc = Counter(g)
    common = sum((pc & gc).values())
    if common == 0:
        return 0.0
    prec = common / len(p); rec = common / len(g)
    return 2 * prec * rec / (prec + rec)

def extract_candidates(ctx):
    """Extract candidate answer spans from a Korean context.

    Strategy: split on whitespace/punctuation boundaries, but Korean text is
    often unspaced — we additionally split on post-position particles to get
    noun-like chunks. Returns list of (span_text, start, end).
    """
    c = str(ctx)
    spans = []
    n = len(c)
    i = 0
    while i < n:
        ch = c[i]
        if ch.isspace() or ch in KSPLIT:
            i += 1
            continue
        # find chunk end: advance until boundary char OR particle-followed position
        j = i
        while j < n and not c[j].isspace() and c[j] not in KSPLIT:
            j += 1
        # try to strip trailing particles
        tok = c[i:j]
        # split off trailing particle if it's in JPARTICLES and the remaining left part is non-empty
        # try longest particle match
        # but keep both versions: original tok and stripped
        # We'll add the original chunk and (optionally) stripped version
        # Also add the long sequence (continuous run until boundary) as candidate
        spans.append((tok, i, j))
        # Try suffix-stripped versions for short particles
        for plen in (2, 1, 3):
            if len(tok) > plen + 1 and tok[-plen:] in JPARTICLES:
                stripped = tok[:-plen]
                if len(stripped) >= 1:
                    spans.append((stripped, i, i + len(stripped)))
                break
        i = j
    return spans

if __name__ == '__main__':
    import sys
    s = "르네상스 시대에 고전 고대에 대한 재발견이 이루어지면서, 보티첼리의 《베누스의 탄생》가 있다."
    for t, a, b in extract_candidates(s):
        print(repr(t), a, b)

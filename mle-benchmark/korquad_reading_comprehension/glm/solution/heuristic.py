"""Heuristic baseline: score candidate spans by keyword overlap with question.

For each test row, extract candidate spans from context, then score them by
similarity to the question (excluding stopword-like particles). Also applies
several hand rules (numeric/date answers when the question asks about years).
"""
import re
import pandas as pd
import numpy as np
from collections import Counter
from common import extract_candidates, normalize, JPARTICLES

STOPQ = set(['는', '은', '이', '가', '을', '를', '의', '에', '에서', '로', '으로',
             '와', '과', '도', '만', '무엇', '누구', '어디', '언제', '몇', '어떤',
             '어느', '원인', '목적', '이유', '의미', '기준', '이름', '장소', '시기',
             '시대', '나라', '국가', '사람', '및', '그', '저', '이', '그', '그것',
             '수', '등', '및', '?', '!', '.', ',', "'", '"', '<', '>', '《', '》',
             '〈', '〉', '「', '」', '『', '』', '-', '·', '/', '(', ')', '[', ']'])

def q_tokens(q):
    """Tokenize question into meaningful Korean character chunks / words."""
    q = str(q)
    # remove trailing question mark / punctuation
    q = re.sub(r'[?？\.！!]+$', '', q).strip()
    # split on whitespace and punctuation
    toks = re.split(r'[\s<《》〈〉「」『』""\'\'·・\-/()\[\],;:]+', q)
    out = []
    for t in toks:
        if not t:
            continue
        # also split off trailing particle
        for plen in (2, 1, 3):
            if len(t) > plen + 1 and t[-plen:] in JPARTICLES:
                stripped = t[:-plen]
                if stripped and stripped not in STOPQ:
                    out.append(stripped)
                t = None
                break
        if t is not None and t not in STOPQ:
            out.append(t)
    return out

def score_span(span_text, q_toks, q_counter, ctx):
    """Score a candidate span given question tokens.

    Heuristics:
    - bonus for sharing character n-grams with question tokens
    - bonus for being a longer noun phrase (but capped)
    - small penalty for very short answers (1 char) unless question wants it
    - bonus if span appears near question keyword occurrences in context
    """
    s = span_text
    sl = len(s)
    if sl == 0:
        return -1e9
    # character n-gram overlap with question (char level)
    s_set = set(s)
    q_chars = set(''.join(q_toks))
    overlap = len(s_set & q_chars)
    # We want overlap but NOT all chars in question (that would be repeating question)
    # Reward if span has chars appearing in question tokens
    score = 0.0
    # Exact token match bonus
    if s in q_toks:
        score += 3.0
    # Partial token-substring bonus (any q tok contained in s or s contained in q tok)
    for qt in q_toks:
        if len(qt) >= 2 and (qt in s or s in qt):
            score += 1.5
            break
    # Character overlap (Jaccard)
    score += 0.5 * overlap / max(1, len(q_chars))
    # Length prior: answers are short-ish (median 5)
    if 2 <= sl <= 12:
        score += 0.5
    elif sl == 1:
        score -= 0.2
    elif sl > 20:
        score -= 0.5
    # Date/year handling
    if re.fullmatch(r'\d{4}년', s):
        score += 1.0
    if re.fullmatch(r'\d+년 \d+월', s) or re.fullmatch(r'\d+월 \d+일', s):
        score += 0.5
    # Numbers
    if re.fullmatch(r'\d+', s) and len(s) >= 1:
        score += 0.3
    # If question contains number, span containing that number is good
    qnums = set(re.findall(r'\d+', ' '.join(q_toks)))
    if qnums:
        snums = set(re.findall(r'\d+', s))
        if snums & qnums:
            score += 1.5
    # Penalize if span is exactly a particle or stopword
    if s in STOPQ or s in JPARTICLES:
        score -= 2.0
    # Proximity bonus: if the span text appears in context right next to a question keyword
    return score

def predict_row(ctx, question):
    cands = extract_candidates(ctx)
    qtoks = q_tokens(question)
    qcounter = Counter(qtoks)
    if not cands:
        # fallback: take whole context (truncated)
        return str(ctx)[:50]
    best = None; best_score = -1e18
    for text, s, e in cands:
        sc = score_span(text, qtoks, qcounter, ctx)
        if sc > best_score:
            best_score = sc; best = (text, s, e)
    return best[0]

def main():
    import sys, os
    base = os.path.dirname(os.path.abspath(__file__))
    proj = os.path.dirname(base)
    test = pd.read_csv(os.path.join(proj, 'test.csv'))
    preds = []
    for i, row in test.iterrows():
        p = predict_row(row['context'], row['question'])
        preds.append(p)
        if i % 1000 == 0:
            print(f"... {i}/{len(test)}", flush=True)
    out = pd.DataFrame({'id': test['id'], 'answer': preds})
    out.to_csv(os.path.join(proj, 'outputs', 'submission.csv'), index=False)
    print("Wrote", len(out), "rows")

if __name__ == '__main__':
    main()

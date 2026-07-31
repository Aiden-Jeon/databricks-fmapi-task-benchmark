"""Feature extraction for KoBEST COPA (no external data / no pretrained weights)."""
import re
import numpy as np
import pandas as pd
from scipy import sparse

HANGUL = re.compile(r'[가-힣A-Za-z0-9]+')

# ---------- basic text utils ----------

def words(s):
    return HANGUL.findall(str(s))


def stems(s, n=2):
    """Pseudo-stem: first n syllables of each eojeol (crude Korean morphology proxy)."""
    return [w[:n] for w in words(s)]


def tails(s, n=3):
    """Last n syllables of each eojeol (captures verb endings / particles)."""
    return [w[-n:] for w in words(s)]


def chars(s):
    return re.sub(r'[^가-힣A-Za-z0-9]', '', str(s))


def ngrams(s, lo=2, hi=4):
    c = chars(s)
    out = []
    for n in range(lo, hi + 1):
        out += [c[i:i + n] for i in range(len(c) - n + 1)]
    return out


# ---------- numeric (hand-crafted) features ----------

def _jac(a, b):
    a, b = set(a), set(b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _cov(a, b):
    """coverage of b by a"""
    a, b = set(a), set(b)
    if not b:
        return 0.0
    return len(a & b) / len(b)


NEG_PAT = re.compile(r'(안 |않|못 |없|아니|말았|마라|지 않)')


def numeric_feats(premise, question, alt):
    p, a = str(premise), str(alt)
    wp, wa = words(p), words(a)
    sp, sa = stems(p), stems(a)
    cp, ca = set(chars(p)), set(chars(a))
    ng_p, ng_a = ngrams(p, 2, 3), ngrams(a, 2, 3)
    f = [
        len(chars(a)) / 10.0,
        len(wa) / 5.0,
        (len(chars(a)) - len(chars(p))) / 10.0,
        _jac(wp, wa),
        _cov(wp, wa),
        _cov(wa, wp),
        _jac(sp, sa),
        _cov(sp, sa),
        _jac(cp, ca),
        _cov(cp, ca),
        _jac(ng_p, ng_a),
        _cov(set(ng_p), set(ng_a)),
        float(bool(NEG_PAT.search(a))),
        float(bool(NEG_PAT.search(a))) - float(bool(NEG_PAT.search(p))),
        float(a.endswith('다.') or a.endswith('다')),
        len(set(wa)) / max(1, len(wa)),
    ]
    return f


NUM_DIM = len(numeric_feats('가', '원인', '나'))


def numeric_block(df, alt_col):
    X = np.array([numeric_feats(p, q, a) for p, q, a in
                  zip(df.premise, df.question, df[alt_col])], dtype=np.float64)
    return X


# ---------- token bag documents ----------

def alt_doc(premise, question, alt):
    """Bag of tokens describing the alternative itself (+ question type conjunctions)."""
    q = 'C' if str(question).strip() == '원인' else 'E'
    out = []
    for t in ngrams(alt, 2, 4):
        out.append('a:' + t)
        out.append('a%s:%s' % (q, t))
    for t in stems(alt, 2):
        out.append('s:' + t)
    for t in tails(alt, 3):
        out.append('t:' + t)
        out.append('t%s:%s' % (q, t))
    ws = words(alt)
    for w in ws:
        out.append('w:' + w)
    for i in range(len(ws) - 1):
        out.append('b:' + ws[i][:2] + '_' + ws[i + 1][:2])
    # positional: last eojeol (the predicate) matters most
    if ws:
        out.append('last:' + ws[-1])
        out.append('last3:' + ws[-1][-3:])
        out.append('last%s:%s' % (q, ws[-1][-3:]))
    return out


def cross_doc(premise, question, alt):
    """Interaction tokens: premise content stem x alternative predicate/stem."""
    q = 'C' if str(question).strip() == '원인' else 'E'
    ps = sorted(set(stems(premise, 2)))
    aw = words(alt)
    a_last = [aw[-1][-3:], aw[-1][:2]] if aw else []
    a_st = sorted(set(stems(alt, 2)))
    out = []
    for x in ps:
        for y in a_last:
            out.append('x:%s>%s' % (x, y))
            out.append('x%s:%s>%s' % (q, x, y))
    for x in ps:
        for y in a_st:
            out.append('y:%s>%s' % (x, y))
    return out


# ---------- variant docs for ablation ----------

def alt_ng(premise, question, alt):
    """char n-grams of the alternative only, with question-type conjunction."""
    q = 'C' if str(question).strip() == '원인' else 'E'
    out = []
    for t in ngrams(alt, 1, 5):
        out.append('a:' + t)
        out.append('a%s:%s' % (q, t))
    return out


def alt_lex(premise, question, alt):
    """word / stem / ending level tokens of the alternative only."""
    q = 'C' if str(question).strip() == '원인' else 'E'
    ws = words(alt)
    out = []
    for w in ws:
        out += ['w:' + w, 's:' + w[:2], 't:' + w[-2:], 't3:' + w[-3:],
                'w%s:%s' % (q, w), 's%s:%s' % (q, w[:2])]
    for i in range(len(ws) - 1):
        out.append('b:' + ws[i][:2] + '_' + ws[i + 1][:2])
    if ws:
        out += ['last:' + ws[-1], 'last3:' + ws[-1][-3:], 'lastq%s:%s' % (q, ws[-1][-3:]),
                'first:' + ws[0][:2]]
    return out


def cross_ng(premise, question, alt):
    """premise char-trigram x alternative predicate ending."""
    q = 'C' if str(question).strip() == '원인' else 'E'
    aw = words(alt)
    if not aw:
        return []
    heads = {aw[-1][:2], aw[-1][-3:]}
    out = []
    for x in sorted(set(ngrams(premise, 2, 3))):
        for h in heads:
            out.append('n:%s>%s' % (x, h))
            out.append('n%s:%s>%s' % (q, x, h))
    return out


def cross_word(premise, question, alt):
    """premise word x alternative word (content-level co-occurrence)."""
    q = 'C' if str(question).strip() == '원인' else 'E'
    out = []
    pw = sorted(set(words(premise)))
    aw = sorted(set(words(alt)))
    for x in pw:
        for y in aw:
            out.append('cw:%s>%s' % (x[:3], y[:3]))
            out.append('cwq%s:%s>%s' % (q, x[:2], y[:2]))
    return out

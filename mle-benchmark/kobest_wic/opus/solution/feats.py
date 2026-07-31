"""Feature engineering utilities for KoBEST WiC (classical ML, no external data)."""
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

BR = re.compile(r'\[([^\]]*)\]')

# common Korean particles / endings to strip from eojeol tails
PARTICLES = ['으로서', '으로써', '이라고', '라고', '에서는', '에서', '으로', '로서', '로써',
             '에게', '한테', '까지', '부터', '조차', '마저', '처럼', '보다', '같이',
             '이나', '나마', '이라', '들의', '들을', '들이', '들은',
             '으', '는', '은', '이', '가', '을', '를', '에', '의', '와', '과', '도', '만',
             '로', '라', '고', '며', '서', '야', '요', '든', '나', '든지', '이다', '다']


def split_marked(text):
    """Return (full_text_no_brackets, target_surface, left, right)."""
    m = BR.search(text)
    if not m:
        return text, '', text, ''
    tgt = m.group(1)
    left = text[:m.start()]
    right = text[m.end():]
    return left + tgt + right, tgt, left, right


def mask_text(text, mask=' TGTTGT '):
    return BR.sub(mask, text)


def stem(tok):
    t = re.sub(r'[^\w가-힣]', '', tok)
    if len(t) <= 1:
        return t
    for p in sorted(PARTICLES, key=len, reverse=True):
        if t.endswith(p) and len(t) - len(p) >= 1:
            return t[:-len(p)]
    return t


def tokens(text, do_stem=True):
    t = mask_text(text, ' ')
    toks = [w for w in re.split(r'[\s\.,!\?;:"\'()\u2026\u3010\u3011\u300c\u300d~/\-\u2014]+', t) if w]
    if do_stem:
        toks = [stem(w) for w in toks]
    return [w for w in toks if w]


def eojeol_with_target(text):
    """The whitespace token that contains the bracketed target, and its tail after target."""
    m = BR.search(text)
    if not m:
        return '', '', ''
    s = text.rfind(' ', 0, m.start()) + 1
    e = text.find(' ', m.end())
    if e == -1:
        e = len(text)
    ej = text[s:e]
    pre = text[s:m.start()]
    post = text[m.end():e]
    post = re.sub(r'[\.,!\?;:"\'()\u2026]+$', '', post)
    return ej, pre, post


def jac(a, b):
    a, b = set(a), set(b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def ovl(a, b):
    a, b = set(a), set(b)
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def window(text, nchar=12):
    m = BR.search(text)
    if not m:
        return text
    return text[max(0, m.start() - nchar):m.start()] + ' ' + text[m.end():m.end() + nchar]


class Rep:
    """Builds several vector representations of all contexts in the corpus."""

    def __init__(self, texts):
        self.texts = texts
        self.masked = [mask_text(t, ' ') for t in texts]
        self.tokstr = [' '.join(tokens(t)) for t in texts]
        self.reps = {}
        self._build()

    def _build(self):
        # char n-grams on masked text
        v1 = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=2,
                             sublinear_tf=True)
        X1 = v1.fit_transform(self.masked)
        self.reps['char'] = normalize(X1)
        # stemmed word unigrams
        v2 = TfidfVectorizer(analyzer='word', token_pattern=r'\S+', ngram_range=(1, 1),
                             min_df=1, sublinear_tf=True)
        X2 = v2.fit_transform(self.tokstr)
        self.reps['word'] = normalize(X2)
        # LSA on both
        for k, n in (('char', 200), ('word', 200)):
            X = self.reps[k]
            n = min(n, X.shape[1] - 1, X.shape[0] - 1)
            svd = TruncatedSVD(n_components=n, random_state=0)
            Z = svd.fit_transform(X)
            self.reps['lsa_' + k] = normalize(Z)

    def sim(self, key, i, j):
        R = self.reps[key]
        if hasattr(R, 'getrow'):
            return np.asarray(R[i].multiply(R[j]).sum(axis=1)).ravel()
        return np.einsum('ij,ij->i', R[i], R[j])

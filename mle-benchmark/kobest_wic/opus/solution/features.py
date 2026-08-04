"""Feature engineering for KoBEST WiC (sklearn-only, no pretrained models)."""
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

HANGUL = r'[가-힣A-Za-z0-9]+'


def split_ctx(s, w):
    """Return (prefix, suffix) around the bracketed target word."""
    s = str(s)
    i, j = s.find('['), s.find(']')
    if i >= 0 and j > i:
        return s[:i], s[j + 1:]
    k = s.find(w)
    if k >= 0:
        return s[:k], s[k + len(w):]
    return s, ''


def clean(s):
    return re.sub(r'[\[\]]', '', str(s))


def toks(s):
    return re.findall(HANGUL, clean(s))


def ngrams(s, n):
    s = re.sub(r'\s+', '', clean(s))
    return set(s[i:i + n] for i in range(len(s) - n + 1))


def jac(A, B):
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def ovl(A, B):
    if not A or not B:
        return 0.0
    return len(A & B) / min(len(A), len(B))


# Korean case/particle markers commonly attached right after a noun
JOSA = ['은', '는', '이', '가', '을', '를', '의', '에', '와', '과', '도', '로', '으',
        '만', '부', '까', '보', '라', '나', '든', '든지', '처', '같']


def josa_class(suffix):
    """Rough classification of what follows the target word."""
    if suffix == '':
        return 'END'
    c = suffix[0]
    if c in ' \t':
        return 'SPACE'          # likely compound / modifier usage
    if c in '.,!?;:\'"’”)':
        return 'PUNCT'
    if c in JOSA:
        return 'JOSA'
    return 'SUFFIX'             # derivational suffix / compounding without space


def struct_frame(df):
    """Per-row structural (surface) features."""
    pre1, post1, pre2, post2 = [], [], [], []
    for w, c1, c2 in zip(df.word, df.context_1, df.context_2):
        a, b = split_ctx(c1, w)
        c, d = split_ctx(c2, w)
        pre1.append(a); post1.append(b); pre2.append(c); post2.append(d)
    out = {}

    def tok_first(t):
        t = t.strip()
        return t.split(' ')[0] if t else ''

    def tok_last(t):
        t = t.strip()
        return t.split(' ')[-1] if t else ''

    # attached suffix string (chars glued to the target word, before next space)
    att1 = [p.split(' ')[0] for p in post1]
    att2 = [p.split(' ')[0] for p in post2]
    # token right before / after the target word
    prev1 = [tok_last(p) for p in pre1]
    prev2 = [tok_last(p) for p in pre2]
    nxt1 = [tok_first(p[len(a):]) for p, a in zip(post1, att1)]
    nxt2 = [tok_first(p[len(a):]) for p, a in zip(post2, att2)]

    out['same_att'] = [float(a == b and a != '') for a, b in zip(att1, att2)]
    out['same_att1c'] = [float(a[:1] == b[:1]) for a, b in zip(att1, att2)]
    out['same_prev'] = [float(a == b and a != '') for a, b in zip(prev1, prev2)]
    out['same_next'] = [float(a == b and a != '') for a, b in zip(nxt1, nxt2)]
    out['prev_share'] = [ovl(ngrams(a, 1), ngrams(b, 1)) if a and b else 0.0
                         for a, b in zip(prev1, prev2)]
    out['next_share'] = [ovl(ngrams(a, 1), ngrams(b, 1)) if a and b else 0.0
                         for a, b in zip(nxt1, nxt2)]
    jc1 = [josa_class(p) for p in post1]
    jc2 = [josa_class(p) for p in post2]
    out['same_josaclass'] = [float(a == b) for a, b in zip(jc1, jc2)]
    for k in ['END', 'SPACE', 'PUNCT', 'JOSA', 'SUFFIX']:
        out['jc_both_' + k] = [float(a == k and b == k) for a, b in zip(jc1, jc2)]
        out['jc_one_' + k] = [float((a == k) != (b == k)) for a, b in zip(jc1, jc2)]
    out['att_len1'] = [len(a) for a in att1]
    out['att_len2'] = [len(a) for a in att2]
    out['att_len_diff'] = [abs(len(a) - len(b)) for a, b in zip(att1, att2)]

    L1 = df.context_1.str.len().values.astype(float)
    L2 = df.context_2.str.len().values.astype(float)
    out['len1'] = L1
    out['len2'] = L2
    out['len_diff'] = np.abs(L1 - L2)
    out['len_ratio'] = np.minimum(L1, L2) / np.maximum(L1, L2)
    # relative position of the target inside the sentence
    p1 = np.array([len(a) for a in pre1], float) / np.maximum(L1, 1)
    p2 = np.array([len(a) for a in pre2], float) / np.maximum(L2, 1)
    out['pos1'] = p1
    out['pos2'] = p2
    out['pos_diff'] = np.abs(p1 - p2)
    out['ntok1'] = [float(len(toks(c))) for c in df.context_1]
    out['ntok2'] = [float(len(toks(c))) for c in df.context_2]

    # lexical overlap of the two contexts (target excluded)
    T1 = [set(toks(c)) - {w} for c, w in zip(df.context_1, df.word)]
    T2 = [set(toks(c)) - {w} for c, w in zip(df.context_2, df.word)]
    out['tok_jac'] = [jac(a, b) for a, b in zip(T1, T2)]
    out['tok_ovl'] = [ovl(a, b) for a, b in zip(T1, T2)]
    out['tok_n_shared'] = [float(len(a & b)) for a, b in zip(T1, T2)]
    for n in (2, 3, 4):
        G1 = [ngrams(c, n) for c in df.context_1]
        G2 = [ngrams(c, n) for c in df.context_2]
        out[f'cg{n}_jac'] = [jac(a, b) for a, b in zip(G1, G2)]
        out[f'cg{n}_ovl'] = [ovl(a, b) for a, b in zip(G1, G2)]
    out['word_len'] = df.word.str.len().values.astype(float)
    # does the other context contain the target word more than once?
    out['rep1'] = [float(str(c).count(w) > 1) for c, w in zip(df.context_1, df.word)]
    out['rep2'] = [float(str(c).count(w) > 1) for c, w in zip(df.context_2, df.word)]
    return pd.DataFrame(out, index=df.index)


class SimSpace:
    """TF-IDF / SVD context space fit transductively on all available contexts."""

    def __init__(self, contexts, n_comp=180, seed=0):
        self.docs = [clean(c) for c in contexts]
        self.char = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4),
                                    min_df=2, sublinear_tf=True)
        self.wordv = TfidfVectorizer(analyzer='word', token_pattern=HANGUL,
                                     ngram_range=(1, 1), min_df=1, sublinear_tf=True)
        Xc = self.char.fit_transform(self.docs)
        Xw = self.wordv.fit_transform(self.docs)
        self.Xc = normalize(Xc)
        self.Xw = normalize(Xw)
        n_comp = min(n_comp, min(Xc.shape) - 1)
        self.svd = TruncatedSVD(n_components=n_comp, random_state=seed)
        self.Zc = normalize(self.svd.fit_transform(self.Xc))
        self.svdw = TruncatedSVD(n_components=n_comp, random_state=seed)
        self.Zw = normalize(self.svdw.fit_transform(self.Xw))

    def transform(self, contexts):
        d = [clean(c) for c in contexts]
        Xc = normalize(self.char.transform(d))
        Xw = normalize(self.wordv.transform(d))
        return dict(Xc=Xc, Xw=Xw,
                    Zc=normalize(self.svd.transform(Xc)),
                    Zw=normalize(self.svdw.transform(Xw)))


def sim_frame(df, space, all_ctx_by_word=None):
    A = space.transform(df.context_1.tolist())
    B = space.transform(df.context_2.tolist())
    out = {}
    for k in ['Xc', 'Xw']:
        out['cos_' + k] = np.asarray(A[k].multiply(B[k]).sum(axis=1)).ravel()
    for k in ['Zc', 'Zw']:
        out['cos_' + k] = (A[k] * B[k]).sum(axis=1)
    return pd.DataFrame(out, index=df.index), A, B

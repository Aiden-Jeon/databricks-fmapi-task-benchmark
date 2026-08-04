"""Shared feature/model utilities for NSMC sentiment classification."""
import os
import re
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

# task root = parent directory of solution/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------- text utils
CHO = list("\u3131\u3132\u3134\u3137\u3138\u3139\u3141\u3142\u3143\u3145\u3146\u3147\u3148\u3149\u314a\u314b\u314c\u314d\u314e")
JUNG = list("\u314f\u3150\u3151\u3152\u3153\u3154\u3155\u3156\u3157\u3158\u3159\u315a\u315b\u315c\u315d\u315e\u315f\u3160\u3161\u3162\u3163")
JONG = [''] + list("\u3131\u3132\u3133\u3134\u3135\u3136\u3137\u3139\u313a\u313b\u313c\u313d\u313e\u313f\u3140\u3141\u3142\u3144\u3145\u3146\u3147\u3148\u314a\u314b\u314c\u314d\u314e")


def decompose_jamo(text):
    """Decompose Hangul syllables into jamo sequence (keeps other chars as-is)."""
    out = []
    for ch in text:
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:
            idx = code - 0xAC00
            out.append(CHO[idx // 588])
            out.append(JUNG[(idx % 588) // 28])
            j = JONG[idx % 28]
            if j:
                out.append(j)
        else:
            out.append(ch)
    return ''.join(out)


_REPEAT = re.compile(r'(.)\1{2,}')


def normalize(text):
    """Light normalization: collapse long char repeats (ㅋㅋㅋㅋ -> ㅋㅋㅋ), squeeze spaces."""
    text = str(text)
    text = _REPEAT.sub(r'\1\1\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_data(train_path=None, test_path=None):
    tr = pd.read_csv(train_path or os.path.join(ROOT, 'train.csv'))
    te = pd.read_csv(test_path or os.path.join(ROOT, 'test.csv'))
    tr['text'] = tr.document.astype(str).map(normalize)
    te['text'] = te.document.astype(str).map(normalize)
    return tr, te


# ---------------------------------------------------------------- features
def build_feature_blocks(train_txt, test_txt, blocks=('char', 'word', 'jamo'), verbose=True):
    """Fit vectorizers on train_txt, return dict name -> (Xtr, Xte)."""
    res = {}
    if 'char' in blocks:
        v = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 5), min_df=3,
                            sublinear_tf=True, max_features=800000)
        a = v.fit_transform(train_txt)
        b = v.transform(test_txt)
        res['char'] = (a, b)
    if 'word' in blocks:
        v = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=2,
                            sublinear_tf=True, token_pattern=r'(?u)\S+')
        a = v.fit_transform(train_txt)
        b = v.transform(test_txt)
        res['word'] = (a, b)
    if 'jamo' in blocks:
        jtr = [decompose_jamo(t) for t in train_txt]
        jte = [decompose_jamo(t) for t in test_txt]
        v = TfidfVectorizer(analyzer='char', ngram_range=(2, 6), min_df=3,
                            sublinear_tf=True, max_features=800000)
        a = v.fit_transform(jtr)
        b = v.transform(jte)
        res['jamo'] = (a, b)
    if verbose:
        for k, (a, _) in res.items():
            print(f'  block {k}: {a.shape[1]} feats')
    return res


def hstack_blocks(blocks, names):
    Xtr = sparse.hstack([blocks[n][0] for n in names]).tocsr()
    Xte = sparse.hstack([blocks[n][1] for n in names]).tocsr()
    return Xtr, Xte


# ---------------------------------------------------------------- NB-SVM
def nb_ratio(X, y):
    """log-count ratio r for NBSVM feature weighting."""
    p = np.asarray(X[y == 1].sum(axis=0)).ravel() + 1.0
    q = np.asarray(X[y == 0].sum(axis=0)).ravel() + 1.0
    p /= p.sum()
    q /= q.sum()
    return np.log(p / q)


def apply_nb(X, r):
    return X.multiply(r).tocsr()

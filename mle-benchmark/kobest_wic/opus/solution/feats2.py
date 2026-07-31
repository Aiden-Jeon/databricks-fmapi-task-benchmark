"""Second-generation features: within-word normalisation, collocations,
PPMI-SVD embeddings, learned metric on LSA space."""
import re
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from scipy import sparse
from feats import BR, tokens, stem, mask_text, eojeol_with_target

VERB_MARK = ['하', '되', '받', '주', '있', '없', '보', '가', '오', '만들', '이루', '치르']


def next_eojeol(text, k=1):
    m = BR.search(text)
    if not m:
        return ''
    rest = text[m.end():]
    parts = [p for p in rest.split(' ') if p]
    # first part is the tail of the same eojeol
    if len(parts) > k:
        return parts[k]
    return ''


def prev_eojeol(text, k=1):
    m = BR.search(text)
    if not m:
        return ''
    head = text[:m.start()]
    parts = [p for p in head.split(' ') if p]
    if len(parts) >= k:
        return parts[-k]
    return ''


def clean(t):
    return re.sub(r'[^\w가-힣]', '', t)


def within_word_norm(S_pair, groups):
    """S_pair: dict pairkey->value ; not used. placeholder."""
    raise NotImplementedError


def build_ppmi_emb(sentences, dim=120, min_count=2, window=None):
    """word-word PPMI (sentence co-occurrence) + SVD."""
    toks = [tokens(s) for s in sentences]
    cnt = Counter(w for t in toks for w in t)
    vocab = {w: i for i, w in enumerate(w for w, c in cnt.items() if c >= min_count)}
    if len(vocab) < 10:
        return {}, None
    rows, cols, vals = [], [], []
    for si, t in enumerate(toks):
        ids = [vocab[w] for w in t if w in vocab]
        for i in set(ids):
            rows.append(si); cols.append(i); vals.append(1.0)
    D = sparse.csr_matrix((vals, (rows, cols)), shape=(len(toks), len(vocab)))
    C = (D.T @ D).tocoo()  # word-word sentence cooccurrence
    tot = C.data.sum()
    marg = np.asarray(D.sum(axis=0)).ravel()
    margsum = marg.sum()
    pmi_r, pmi_c, pmi_v = [], [], []
    for i, j, v in zip(C.row, C.col, C.data):
        p = v / tot
        pi = marg[i] / margsum
        pj = marg[j] / margsum
        x = np.log(max(p, 1e-12) / max(pi * pj, 1e-12))
        if x > 0:
            pmi_r.append(i); pmi_c.append(j); pmi_v.append(x)
    M = sparse.csr_matrix((pmi_v, (pmi_r, pmi_c)), shape=(len(vocab), len(vocab)))
    d = min(dim, min(M.shape) - 1)
    svd = TruncatedSVD(n_components=d, random_state=0)
    E = svd.fit_transform(M)
    E = normalize(E)
    return vocab, E


def emb_context_vectors(texts, vocab, E, idf=None):
    V = np.zeros((len(texts), E.shape[1]))
    for i, t in enumerate(texts):
        ws = [w for w in tokens(t) if w in vocab]
        if not ws:
            continue
        if idf is not None:
            wt = np.array([idf.get(w, 1.0) for w in ws])
        else:
            wt = np.ones(len(ws))
        V[i] = (E[[vocab[w] for w in ws]] * wt[:, None]).sum(axis=0)
    return normalize(V)

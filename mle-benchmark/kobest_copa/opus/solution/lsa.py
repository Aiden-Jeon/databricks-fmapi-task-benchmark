"""Low-rank bilinear premise x alternative interaction features via LSA."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from features import ngrams, words


def sent_doc(s):
    return ' '.join(ngrams(s, 2, 4) + ['w:' + w for w in words(s)] + ['s:' + w[:2] for w in words(s)])


def build_lsa(tr, te, dim=48, seed=0):
    corpus = []
    for d in (tr, te):
        corpus += list(d.premise) + list(d.alternative_1) + list(d.alternative_2)
    docs = [sent_doc(s) for s in corpus]
    vec = TfidfVectorizer(analyzer=lambda x: x.split(), min_df=2, sublinear_tf=True)
    M = vec.fit_transform(docs)
    svd = TruncatedSVD(n_components=dim, random_state=seed)
    svd.fit(M)

    def emb(texts):
        return normalize(svd.transform(vec.transform([sent_doc(s) for s in texts])))
    return emb


def bilinear_block(tr, te, dim=48, qsplit=True, seed=0):
    emb = build_lsa(tr, te, dim=dim, seed=seed)
    out = []
    for d in (tr, te):
        P = emb(list(d.premise))
        A1, A2 = emb(list(d.alternative_1)), emb(list(d.alternative_2))
        q = (d.question.astype(str).str.strip() == '원인').values.astype(float)[:, None] * 2 - 1
        o1 = (P[:, :, None] * A1[:, None, :]).reshape(len(d), -1)
        o2 = (P[:, :, None] * A2[:, None, :]).reshape(len(d), -1)
        D = o1 - o2
        if qsplit:
            D = np.hstack([D, D * q])
        # cosine sims + own-embedding delta
        c1 = (P * A1).sum(1)[:, None]; c2 = (P * A2).sum(1)[:, None]
        D = np.hstack([D, A1 - A2, c1 - c2, (c1 - c2) * q])
        out.append(D)
    return out

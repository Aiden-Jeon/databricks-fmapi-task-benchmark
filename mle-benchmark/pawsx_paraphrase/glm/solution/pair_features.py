"""Pairwise TF-IDF feature extraction.

Each sentence is encoded as a TF-IDF vector.  We then build pair-level
features: cosine similarity, element-wise absolute difference (via
sparse operations on a shared vocabulary), and the hadamard product.
These capture whether the two sentences use the *same* vocabulary, which
is what paraphrase detection needs, while the existing hand features
capture order / structure.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sp_normalize


def _tokenize(s):
    import re
    s = re.sub(r"\s+", " ", str(s).strip())
    s = re.sub(r"([.,!?;:()\"'\-~/])", r" \1 ", s)
    return s.split()


def build_pairwise_tfidf(train_df, test_df, max_features=40000):
    vec = TfidfVectorizer(
        analyzer="word",
        tokenizer=_tokenize,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        max_features=max_features,
        norm="l2",
    )
    vec.fit(
        pd.concat([train_df["sentence1"].astype(str), train_df["sentence2"].astype(str)]).tolist()
    )
    S1tr = vec.transform(train_df["sentence1"].astype(str))
    S2tr = vec.transform(train_df["sentence2"].astype(str))
    S1te = vec.transform(test_df["sentence1"].astype(str))
    S2te = vec.transform(test_df["sentence2"].astype(str))

    # cosine similarity (rows already L2-normalized)
    cos_tr = np.array(S1tr.multiply(S2tr).sum(axis=1)).ravel()
    cos_te = np.array(S1te.multiply(S2te).sum(axis=1)).ravel()

    # |s1 - s2| features in sparse space (then top-k by variance could be used)
    diff_tr = (S1tr - S2tr)
    diff_tr.data = np.abs(diff_tr.data)
    prod_tr = S1tr.multiply(S2tr)
    diff_te = (S1te - S2te)
    diff_te.data = np.abs(diff_te.data)
    prod_te = S1te.multiply(S2te)

    from scipy.sparse import hstack
    Xtr = hstack([diff_tr, prod_tr]).tocsr()
    Xte = hstack([diff_te, prod_te]).tocsr()
    return Xtr, Xte, cos_tr, cos_te

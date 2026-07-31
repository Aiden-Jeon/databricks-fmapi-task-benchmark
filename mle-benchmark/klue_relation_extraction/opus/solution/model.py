"""Sparse feature matrix builder + linear models for KLUE-RE."""
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

from features import build_text_fields
from neighbors import NeighborFeatures


def make_vectorizers():
    return [
        ("marked_c", "tmarked", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4), min_df=3, max_features=400000,
            sublinear_tf=True, lowercase=False)),
        ("marked_w", "tmarked", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=2, max_features=300000,
            sublinear_tf=True, lowercase=False, token_pattern=r"\S+")),
        ("btw_c", "between", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_features=500000,
            sublinear_tf=True, lowercase=False)),
        ("btw_w", "between", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 3), min_df=2, max_features=300000,
            sublinear_tf=True, lowercase=False, token_pattern=r"\S+")),
        ("subj_c", "subj", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(1, 4), min_df=2, max_features=200000,
            sublinear_tf=True, lowercase=False)),
        ("obj_c", "obj", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(1, 4), min_df=2, max_features=200000,
            sublinear_tf=True, lowercase=False)),
        ("subj_f", "subj", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 1), min_df=1, lowercase=False,
            token_pattern=r".+")),
        ("obj_f", "obj", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 1), min_df=1, lowercase=False,
            token_pattern=r".+")),
        ("xbtw_w", "xbtw", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=2, max_features=400000,
            sublinear_tf=True, lowercase=False, token_pattern=r"\S+")),
        ("sctx_c", "sctx", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_features=250000,
            sublinear_tf=True, lowercase=False)),
        ("octx_c", "octx", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_features=250000,
            sublinear_tf=True, lowercase=False)),
        ("sctx_w", "sctx", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=2, max_features=200000,
            sublinear_tf=True, lowercase=False, token_pattern=r"\S+")),
        ("octx_w", "octx", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=2, max_features=200000,
            sublinear_tf=True, lowercase=False, token_pattern=r"\S+")),
        ("pair_f", "pair", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 1), min_df=1, lowercase=False,
            token_pattern=r".+")),
        ("left_c", "left", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4), min_df=3, max_features=150000,
            sublinear_tf=True, lowercase=False)),
        ("right_c", "right", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4), min_df=3, max_features=150000,
            sublinear_tf=True, lowercase=False)),
        ("sent_c", "sentence", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4), min_df=3, max_features=300000,
            sublinear_tf=True, lowercase=False)),
    ]


WEIGHTS = {
    "marked_c": 1.0, "marked_w": 1.0, "btw_c": 1.3, "btw_w": 1.3,
    "subj_c": 0.8, "obj_c": 1.0, "subj_f": 0.7, "obj_f": 0.9, "pair_f": 1.0,
    "left_c": 0.6, "right_c": 0.6, "sent_c": 0.5, "type_oh": 1.5, "num": 1.0,
    "xbtw_w": 1.0, "sctx_c": 0.9, "octx_c": 0.9, "sctx_w": 0.9, "octx_w": 0.9,
}


NEIGH_W = 1.0


class FeatureBuilder:
    """Sparse features. `use_neighbors=True` adds leak-free relational features
    derived from the labels of *other* training rows."""

    def __init__(self, use_neighbors=True, neigh_w=NEIGH_W):
        self.vecs = make_vectorizers()
        self.ohe = OneHotEncoder(handle_unknown="ignore")
        self.use_neighbors = use_neighbors
        self.neigh_w = neigh_w
        self.nf = None

    def fit_transform(self, df, y=None):
        tf, num = build_text_fields(df)
        blocks = []
        for name, col, v in self.vecs:
            blocks.append(v.fit_transform(tf[col].values) * WEIGHTS[name])
        tp = np.stack([tf["stype"].values, tf["otype"].values], axis=1)
        blocks.append(self.ohe.fit_transform(tp) * WEIGHTS["type_oh"])
        blocks.append(sp.csr_matrix(num) * WEIGHTS["num"])
        if self.use_neighbors:
            assert y is not None, "y required for neighbour features"
            self.nf = NeighborFeatures().fit(df, y)
            blocks.append(self.nf.transform(df) * self.neigh_w)
        return sp.hstack(blocks, format="csr").astype(np.float32)

    def transform(self, df):
        tf, num = build_text_fields(df)
        blocks = []
        for name, col, v in self.vecs:
            blocks.append(v.transform(tf[col].values) * WEIGHTS[name])
        tp = np.stack([tf["stype"].values, tf["otype"].values], axis=1)
        blocks.append(self.ohe.transform(tp) * WEIGHTS["type_oh"])
        blocks.append(sp.csr_matrix(num) * WEIGHTS["num"])
        if self.use_neighbors:
            blocks.append(self.nf.transform(df) * self.neigh_w)
        return sp.hstack(blocks, format="csr").astype(np.float32)

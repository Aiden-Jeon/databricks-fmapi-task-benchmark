"""Feature engineering for KoBEST BoolQ.

Shared by train.py (cross-validation + training) and predict.py so that
train-time and inference-time features are identical.
"""
import re
from difflib import SequenceMatcher

import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer

TAG_RE = re.compile(r"<[^>]+>")


def clean(s):
    return TAG_RE.sub(" ", str(s))


def build_vectorizers(train_df):
    """Fit TF-IDF vectorizers on train data only (no leakage from test)."""
    train_p = train_df["paragraph"].map(clean)
    train_q = train_df["question"].map(clean)
    train_all = pd.concat([train_p, train_q])

    # word-order features over concatenated (paragraph, question)
    v_word = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                             min_df=2, sublinear_tf=True)
    v_word.fit(train_all)

    # symmetric word-level features
    v_diff = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    v_diff.fit(train_all)

    # question-only features
    v_q = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                          min_df=2, sublinear_tf=True)
    v_q.fit(train_q)

    return v_word, v_diff, v_q


def hand_features(df):
    """Lexical-overlap / containment features between paragraph and question."""
    rows = []
    for p, q in zip(df["paragraph"], df["question"]):
        p = clean(p)
        q = clean(q).strip()
        pt = p.split()
        qt = q.split()
        pset, qset = set(pt), set(qt)
        q_content = set(t for t in qt if len(t) >= 2)
        inter = len(qset & pset)
        inter_c = len(q_content & pset)
        union = len(qset | pset)

        p_sub = p.replace(" ", "")
        q_sub = q.replace(" ", "")
        bigrams = [q_sub[i:i + 2] for i in range(len(q_sub) - 1)]
        bg_hit = (sum(1 for b in bigrams if b in p_sub) / max(len(bigrams), 1))
        m = SequenceMatcher(None, q_sub, p_sub, autojunk=False) \
            .find_longest_match(0, len(q_sub), 0, len(p_sub))

        rows.append([
            len(qt),
            len(pt),
            inter / max(len(qset), 1),
            inter / max(union, 1),
            inter_c / max(len(q_content), 1),
            bg_hit,
            m.size / max(len(q_sub), 1),
            float(p_sub.find(q_sub[: max(len(q_sub) // 2, 4)]) >= 0),
            len(q),
            len(p),
            float(q.endswith("?")),
            len(qt) - len(qset),
        ])
    cols = ["q_toklen", "p_toklen", "jac_q", "jac_union", "contain_content",
            "q_bigram_in_p", "lcs_ratio", "q_prefix_in_p", "q_len", "p_len",
            "ends_qmark", "q_repeat_tok"]
    return pd.DataFrame(rows, columns=cols, index=df.index)


def build_matrix(df, v_word, v_diff, v_q):
    """Full feature matrix: ordered TF-IDF (p+q and q+p) + word-level + q-only
    + handcrafted overlap features."""
    p = df["paragraph"].map(clean)
    q = df["question"].map(clean)
    X = (v_word.transform(p + " " + q) + v_word.transform(q + " " + p)).tocsr()
    X2 = v_diff.transform(p + " " + q)
    X3 = v_q.transform(q)
    hf = hand_features(df)
    X_full = hstack([X, X2, X3, csr_matrix(hf.values)]).tocsr()
    return X_full, hf

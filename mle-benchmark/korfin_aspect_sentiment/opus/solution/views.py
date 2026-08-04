"""Extra text views (directional context, clause around aspect, suffix stripping)
and additional base models built on them."""
import os
import re
import sys
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, add_feats, CLASSES, TASK, MASK, to_jamo  # noqa

CACHE = os.path.join(TASK, "solution", "cache")
SEED = 42
NFOLD = 5
CLAUSE_SEP = re.compile(r"(?<=[,\.])|(?<=지만)|(?<=하며)|(?<=으며)|(?<=하고)|(?<=면서)|(?<=반면)")

JOSA = ["으로서", "에서는", "에게는", "으로", "에서", "에게", "까지", "부터", "보다",
        "이란", "라는", "는", "은", "이", "가", "을", "를", "의", "에", "도", "와",
        "과", "만", "로", "께", "야"]


def left_right(s, a, w=40):
    if not (isinstance(a, str) and a and a in s):
        return s, s
    i = s.index(a)
    return s[max(0, i - w):i], s[i + len(a):i + len(a) + w]


def clause(s, a):
    if not (isinstance(a, str) and a and a in s):
        return s
    parts = [p for p in CLAUSE_SEP.split(s) if p]
    hit = [p for p in parts if a in p]
    return (" || ".join(hit) if hit else s).replace(a, MASK)


def strip_josa(text):
    out = []
    for tok in text.split():
        for j in JOSA:
            if tok.endswith(j) and len(tok) - len(j) >= 2:
                tok = tok[: -len(j)]
                break
        out.append(tok)
    return " ".join(out)


def add_views(df):
    lr_ = [left_right(s, a) for s, a in zip(df.sentence, df.aspect)]
    df["left"] = [x[0] for x in lr_]
    df["right"] = [x[1] for x in lr_]
    df["dir"] = ("L " + df["left"] + " |R " + df["right"].apply(
        lambda t: " ".join("R@" + w for w in t.split())))
    df["clause"] = [clause(s, a) for s, a in zip(df.sentence, df.aspect)]
    df["stripped"] = [strip_josa(t) for t in df["masked"]]
    df["cj"] = df["clause"] + " ~~ " + df["masked"]
    df["cj_jamo"] = [to_jamo(t) for t in df["cj"]]
    return df


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


SPECS = {
    "dir": [("dir", dict(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
            ("dir", dict(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True))],
    "clause": [("cj", dict(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
               ("clause", dict(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True))],
    "strip": [("stripped", dict(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
              ("stripped", dict(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True))],
    "right": [("right", dict(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
              ("right", dict(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True))],
    "multi": [("cj", dict(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
              ("dir", dict(analyzer="char_wb", ngram_range=(2, 4), min_df=3, sublinear_tf=True)),
              ("text2", dict(analyzer="word", ngram_range=(1, 3), min_df=2, sublinear_tf=True)),
              ("win15", dict(analyzer="char_wb", ngram_range=(1, 5), min_df=2, sublinear_tf=True))],
}

SPECS["multi2"] = [
    ("cj_jamo", dict(analyzer="char_wb", ngram_range=(2, 6), min_df=3, sublinear_tf=True)),
    ("stripped", dict(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
    ("dir", dict(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
    ("clause", dict(analyzer="char_wb", ngram_range=(1, 4), min_df=2, sublinear_tf=True)),
]
SPECS["multi3"] = SPECS["multi"] + [
    ("stripped", dict(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
    ("right", dict(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
    ("aspect", dict(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)),
]

MODELS = [("lr_dir", "dir", "lr"), ("svc_dir", "dir", "svc"),
          ("lr_clause", "clause", "lr"), ("svc_clause", "clause", "svc"),
          ("lr_strip", "strip", "lr"), ("svc_strip", "strip", "svc"),
          ("svc_right", "right", "svc"),
          ("lr_multi", "multi", "lr"), ("svc_multi", "multi", "svc"),
          ("lr_multi2", "multi2", "lr"), ("svc_multi2", "multi2", "svc"),
          ("lr_multi3", "multi3", "lr"), ("svc_multi3", "multi3", "svc"),
          ("svcA_multi", "multi", "svcA"), ("lrA_multi", "multi", "lrA"),
          ("ridge_multi", "multi", "ridge")]


def main():
    tr, te = load()
    tr, te = add_views(tr), add_views(te)
    le = LabelEncoder().fit(CLASSES)
    y = le.transform(tr.label.values)
    folds = list(StratifiedKFold(NFOLD, shuffle=True, random_state=SEED).split(tr, y))
    for name, fs, kind in MODELS:
        fo = f"{CACHE}/{name}_oof.npy"
        if os.path.exists(fo):
            print(name, "cached")
            continue
        t0 = time.time()
        oof = np.zeros((len(tr), 3))
        tep = np.zeros((len(te), 3))
        for i_tr, i_va in folds:
            Xa, Xb, Xc = [], [], []
            for col, kw in SPECS[fs]:
                v = TfidfVectorizer(**kw)
                Xa.append(v.fit_transform(tr[col].iloc[i_tr]))
                Xb.append(v.transform(tr[col].iloc[i_va]))
                Xc.append(v.transform(te[col]))
            Xa, Xb, Xc = sp.hstack(Xa).tocsr(), sp.hstack(Xb).tocsr(), sp.hstack(Xc).tocsr()
            if kind in ("lr", "lrA"):
                C = 2 if kind == "lr" else 8
                m = LogisticRegression(C=C, max_iter=3000).fit(Xa, y[i_tr])
                oof[i_va] = m.predict_proba(Xb)
                tep += m.predict_proba(Xc) / NFOLD
            elif kind == "ridge":
                from sklearn.linear_model import RidgeClassifier
                m = RidgeClassifier(alpha=1.0).fit(Xa, y[i_tr])
                oof[i_va] = softmax(2 * m.decision_function(Xb))
                tep += softmax(2 * m.decision_function(Xc)) / NFOLD
            else:
                C = 0.3 if kind == "svc" else 0.1
                m = LinearSVC(C=C, dual=True, max_iter=5000).fit(Xa, y[i_tr])
                oof[i_va] = softmax(2 * m.decision_function(Xb))
                tep += softmax(2 * m.decision_function(Xc)) / NFOLD
        np.save(fo, oof)
        np.save(f"{CACHE}/{name}_test.npy", tep)
        print(f"{name:12s} f1={f1_score(y, oof.argmax(1), average='macro'):.4f} "
              f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()

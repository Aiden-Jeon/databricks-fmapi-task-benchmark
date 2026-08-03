import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

NAMES = ["여성/가족", "남성", "성소수자", "인종/국적", "연령", "지역", "종교", "기타혐오", "악플/욕설", "clean"]
NL = 10


def load(base="."):
    tr = pd.read_csv(f"{base}/train.csv", dtype={"labels": str})
    te = pd.read_csv(f"{base}/test.csv")
    Y = np.array([[int(c) for c in s] for s in tr["labels"]], dtype=np.int8)
    return tr, te, Y


_ws = re.compile(r"\s+")
_rep = re.compile(r"(.)\1{2,}")


def norm(s):
    s = str(s)
    s = s.replace("\u200b", " ")
    s = _rep.sub(r"\1\1", s)          # collapse 3+ repeats to 2
    s = _ws.sub(" ", s).strip()
    return s


def make_vec():
    return FeatureUnion([
        ("cw", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                               sublinear_tf=True, max_features=400000)),
        ("w", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                              sublinear_tf=True)),
    ])


def macro_f1(Yt, Yp):
    """Macro F1 over labels present in ground truth (as spec describes)."""
    fs = []
    for j in range(Yt.shape[1]):
        t, p = Yt[:, j], Yp[:, j]
        if t.sum() == 0:
            continue
        tp = int((t & p).sum()); fp = int(((1 - t) & p).sum()); fn = int((t & (1 - p)).sum())
        fs.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(fs))


def tune_thresholds(Yt, P, base=0.5):
    """Per-label threshold maximizing that label's F1."""
    th = np.full(P.shape[1], base)
    for j in range(P.shape[1]):
        t = Yt[:, j]
        if t.sum() == 0:
            continue
        cand = np.unique(np.round(np.quantile(P[:, j], np.linspace(0.001, 0.999, 400)), 5))
        best, bt = -1, base
        for c in cand:
            p = (P[:, j] >= c).astype(np.int8)
            tp = int((t & p).sum()); fp = int(((1 - t) & p).sum()); fn = int((t & (1 - p)).sum())
            f = 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)
            if f > best:
                best, bt = f, c
        th[j] = bt
    return th


def decide(P, th):
    """Binarize with thresholds; enforce >=1 label and clean-exclusivity."""
    Yp = (P >= th[None, :]).astype(np.int8)
    # rows with nothing -> argmax of P/th ratio
    empty = Yp.sum(1) == 0
    if empty.any():
        r = P / np.maximum(th[None, :], 1e-9)
        Yp[empty, r[empty].argmax(1)] = 1
    # clean exclusivity: clean=1 only if no hate label
    hate = Yp[:, :9].sum(1) > 0
    Yp[hate, 9] = 0
    # if clean predicted alone stays; if clean was the only one and got zeroed handled above
    return Yp

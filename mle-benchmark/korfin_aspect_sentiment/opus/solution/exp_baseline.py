import sys, time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from scipy import sparse

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import build_frame, numeric_feats

SEED = 42
LABELS = ["NEGATIVE", "NEUTRAL", "POSITIVE"]


def make_blocks(Ftr, Fte):
    """Return list of (name, Xtr, Xte) sparse blocks."""
    specs = [
        ("marked_char", "marked", dict(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                       sublinear_tf=True, max_features=300000)),
        ("marked_word", "marked", dict(analyzer="word", ngram_range=(1, 2), min_df=2,
                                       sublinear_tf=True)),
        ("masked_char", "masked", dict(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                       sublinear_tf=True, max_features=300000)),
        ("ctx_char", "ctx", dict(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                 sublinear_tf=True, max_features=300000)),
        ("ctxs_char", "ctx_s", dict(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                    sublinear_tf=True, max_features=200000)),
        ("clause_char", "clause", dict(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                       sublinear_tf=True, max_features=300000)),
        ("clause_word", "clause", dict(analyzer="word", ngram_range=(1, 2), min_df=2,
                                       sublinear_tf=True)),
        ("aspect_char", "aspect", dict(analyzer="char_wb", ngram_range=(2, 4), min_df=2,
                                       sublinear_tf=True)),
    ]
    out = []
    for name, col, kw in specs:
        v = TfidfVectorizer(**kw)
        Xtr = v.fit_transform(Ftr[col])
        Xte = v.transform(Fte[col])
        out.append((name, Xtr, Xte))
    return out


def cv_eval(X, y, model_fn, n_splits=5, seed=SEED):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros((len(y), 3))
    for tr_i, va_i in skf.split(X, y):
        m = model_fn()
        m.fit(X[tr_i], y[tr_i])
        oof[va_i] = m.predict_proba(X[va_i])
    pred = oof.argmax(1)
    return f1_score(y, pred, average="macro"), oof


def main():
    tr = pd.read_csv("train.csv")
    te = pd.read_csv("test.csv")
    y = tr.label.map({l: i for i, l in enumerate(LABELS)}).values
    Ftr, Fte = build_frame(tr), build_frame(te)
    blocks = make_blocks(Ftr, Fte)

    t0 = time.time()
    for name, Xtr, Xte in blocks:
        s, _ = cv_eval(Xtr, y, lambda: LogisticRegression(C=4, max_iter=2000))
        print(f"{name:14s} dim={Xtr.shape[1]:7d} LR_C4 macroF1={s:.4f}  ({time.time()-t0:.0f}s)")

    # combined
    combos = {
        "all": [b[0] for b in blocks],
        "marked+ctx+clause": ["marked_char", "marked_word", "ctx_char", "clause_char", "clause_word"],
        "marked_only": ["marked_char", "marked_word"],
        "core": ["marked_char", "marked_word", "masked_char", "ctx_char", "clause_char"],
    }
    d = {b[0]: (b[1], b[2]) for b in blocks}
    for cname, names in combos.items():
        Xtr = sparse.hstack([d[n][0] for n in names]).tocsr()
        for C in [1, 2, 4, 8]:
            s, _ = cv_eval(Xtr, y, lambda C=C: LogisticRegression(C=C, max_iter=3000))
            print(f"COMBO {cname:20s} C={C} macroF1={s:.4f} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()

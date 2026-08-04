"""Generate OOF + test probability matrices for a bank of models."""
import os
import sys
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, numeric_feats, CLASSES, TASK  # noqa

CACHE = os.path.join(TASK, "solution", "cache")
os.makedirs(CACHE, exist_ok=True)
NFOLD = 5
SEED = 42


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


# ---------------- vectorizer specs ----------------
def vec_char(col, lo=2, hi=5, mindf=2):
    return (col, lambda: TfidfVectorizer(analyzer="char_wb", ngram_range=(lo, hi),
                                         min_df=mindf, sublinear_tf=True,
                                         max_features=500000))


def vec_word(col, lo=1, hi=2, mindf=2):
    return (col, lambda: TfidfVectorizer(analyzer="word", ngram_range=(lo, hi),
                                         min_df=mindf, sublinear_tf=True))


FEATSETS = {
    # name: list of (col, vectorizer factory)
    "text_cw": [vec_char("text", 2, 5), vec_word("text", 1, 2)],
    "text2_cw": [vec_char("text2", 2, 5), vec_word("text2", 1, 3)],
    "jamo": [vec_char("jamo", 2, 6, 3)],
    "win_cw": [vec_char("win", 1, 5), vec_word("win", 1, 2)],
    "masked_cw": [vec_char("masked", 2, 6), vec_word("masked", 1, 2)],
    "cnt_text": [("text", lambda: CountVectorizer(analyzer="char_wb",
                                                  ngram_range=(2, 4), min_df=2)),
                 ("text", lambda: CountVectorizer(analyzer="word",
                                                  ngram_range=(1, 2), min_df=2))],
}


def make_clf(kind):
    if kind == "lr":
        return LogisticRegression(C=5, max_iter=3000)
    if kind == "lr2":
        return LogisticRegression(C=1, max_iter=3000)
    if kind == "lr20":
        return LogisticRegression(C=20, max_iter=3000)
    if kind == "svc":
        return LinearSVC(C=0.5, dual=True, max_iter=5000)
    if kind == "svc2":
        return LinearSVC(C=0.15, dual=True, max_iter=5000)
    if kind == "sgd":
        return SGDClassifier(loss="modified_huber", alpha=1e-5, max_iter=3000,
                             random_state=SEED)
    if kind == "ridge":
        return RidgeClassifier(alpha=1.0)
    if kind == "cnb":
        return ComplementNB(alpha=0.3)
    if kind == "mnb":
        return MultinomialNB(alpha=0.2)
    raise ValueError(kind)


def proba(clf, X):
    if hasattr(clf, "predict_proba"):
        p = clf.predict_proba(X)
        return np.clip(p, 1e-7, 1)
    d = clf.decision_function(X)
    return softmax(d * 2.0)


def run_linear(name, featset, kind, tr, te, y, folds):
    specs = FEATSETS[featset]
    oof = np.zeros((len(tr), 3))
    tep = np.zeros((len(te), 3))
    for f, (i_tr, i_va) in enumerate(folds):
        Xtr, Xva, Xte = [], [], []
        for col, fac in specs:
            v = fac()
            Xtr.append(v.fit_transform(tr[col].iloc[i_tr]))
            Xva.append(v.transform(tr[col].iloc[i_va]))
            Xte.append(v.transform(te[col]))
        Xtr, Xva, Xte = sp.hstack(Xtr).tocsr(), sp.hstack(Xva).tocsr(), sp.hstack(Xte).tocsr()
        clf = make_clf(kind).fit(Xtr, y[i_tr])
        oof[i_va] = proba(clf, Xva)
        tep += proba(clf, Xte) / len(folds)
    return oof, tep


def run_lgbm(name, featset, tr, te, y, folds, nsvd=250):
    import lightgbm as lgb
    specs = FEATSETS[featset]
    ntr_num = numeric_feats(tr)
    nte_num = numeric_feats(te)
    oof = np.zeros((len(tr), 3))
    tep = np.zeros((len(te), 3))
    for f, (i_tr, i_va) in enumerate(folds):
        Xtr, Xva, Xte = [], [], []
        for col, fac in specs:
            v = fac()
            Xtr.append(v.fit_transform(tr[col].iloc[i_tr]))
            Xva.append(v.transform(tr[col].iloc[i_va]))
            Xte.append(v.transform(te[col]))
        Xtr, Xva, Xte = sp.hstack(Xtr).tocsr(), sp.hstack(Xva).tocsr(), sp.hstack(Xte).tocsr()
        svd = TruncatedSVD(nsvd, random_state=SEED).fit(Xtr)
        A = np.c_[svd.transform(Xtr), ntr_num[i_tr]]
        B = np.c_[svd.transform(Xva), ntr_num[i_va]]
        C = np.c_[svd.transform(Xte), nte_num]
        m = lgb.LGBMClassifier(n_estimators=900, learning_rate=0.05, num_leaves=31,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
                               min_child_samples=20, reg_lambda=1.0, verbose=-1,
                               random_state=SEED, objective="multiclass", num_class=3)
        m.fit(A, y[i_tr], eval_set=[(B, y[i_va])],
              callbacks=[lgb.early_stopping(60, verbose=False)])
        oof[i_va] = m.predict_proba(B)
        tep += m.predict_proba(C) / len(folds)
    return oof, tep


MODELS = [
    ("lr_text", "text_cw", "lr"),
    ("lr2_text", "text_cw", "lr2"),
    ("lr20_text", "text_cw", "lr20"),
    ("svc_text", "text_cw", "svc"),
    ("svc2_text", "text_cw", "svc2"),
    ("sgd_text", "text_cw", "sgd"),
    ("ridge_text", "text_cw", "ridge"),
    ("lr_text2", "text2_cw", "lr"),
    ("svc_text2", "text2_cw", "svc"),
    ("lr_jamo", "jamo", "lr"),
    ("svc_jamo", "jamo", "svc"),
    ("lr_win", "win_cw", "lr"),
    ("svc_win", "win_cw", "svc"),
    ("lr_masked", "masked_cw", "lr"),
    ("cnb_text", "cnt_text", "cnb"),
    ("mnb_text", "cnt_text", "mnb"),
]


def main():
    tr, te = load()
    le = LabelEncoder().fit(CLASSES)
    y = le.transform(tr.label.values)
    skf = StratifiedKFold(NFOLD, shuffle=True, random_state=SEED)
    folds = list(skf.split(tr, y))
    np.save(f"{CACHE}/y.npy", y)

    only = sys.argv[1:] if len(sys.argv) > 1 else None
    for name, fs, kind in MODELS:
        if only and name not in only:
            continue
        fo, ft = f"{CACHE}/{name}_oof.npy", f"{CACHE}/{name}_test.npy"
        if os.path.exists(fo):
            oof = np.load(fo)
            print(f"{name:12s} cached  f1={f1_score(y, oof.argmax(1), average='macro'):.4f}")
            continue
        t0 = time.time()
        oof, tep = run_linear(name, fs, kind, tr, te, y, folds)
        np.save(fo, oof)
        np.save(ft, tep)
        print(f"{name:12s} f1={f1_score(y, oof.argmax(1), average='macro'):.4f} "
              f"({time.time()-t0:.0f}s)")

    if not only or "lgbm" in (only or []):
        for nm, fs in [("lgbm_text", "text_cw")]:
            fo = f"{CACHE}/{nm}_oof.npy"
            if not os.path.exists(fo):
                t0 = time.time()
                oof, tep = run_lgbm(nm, fs, tr, te, y, folds)
                np.save(fo, oof)
                np.save(f"{CACHE}/{nm}_test.npy", tep)
                print(f"{nm:12s} f1={f1_score(y, oof.argmax(1), average='macro'):.4f} "
                      f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()

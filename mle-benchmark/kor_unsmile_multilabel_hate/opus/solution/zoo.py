"""Model zoo: computes 5-fold OOF + full-train test predictions for several
base learners and caches them to solution/cache/*.npz."""
import os
import sys
import time
import numpy as np
from scipy import sparse
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import common as C

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE, exist_ok=True)
SEED = 42
NFOLD = 5


# ----------------------------------------------------------------- features
def feats_main(tr_txt, te_txt):
    return C.build_features(tr_txt, te_txt)


def feats_jamo(tr_txt, te_txt):
    a = [C.decompose_hangul(C.normalize(t)) for t in tr_txt]
    b = [C.decompose_hangul(C.normalize(t)) for t in te_txt]
    v = TfidfVectorizer(analyzer="char", ngram_range=(1, 6), min_df=2,
                        sublinear_tf=True, max_features=300000)
    return v.fit_transform(a), v.transform(b)


def feats_jamo_word(tr_txt, te_txt):
    """Jamo char n-grams + whitespace word n-grams (strongest single space)."""
    an = [C.normalize(t) for t in tr_txt]
    bn = [C.normalize(t) for t in te_txt]
    aj = [C.decompose_hangul(t) for t in an]
    bj = [C.decompose_hangul(t) for t in bn]
    v1 = TfidfVectorizer(analyzer="char", ngram_range=(1, 6), min_df=2,
                         sublinear_tf=True, max_features=300000)
    v2 = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                         sublinear_tf=True, token_pattern=r"(?u)\S+")
    return (sparse.hstack([v1.fit_transform(aj), v2.fit_transform(an)]).tocsr(),
            sparse.hstack([v1.transform(bj), v2.transform(bn)]).tocsr())


def feats_jamo_short(tr_txt, te_txt):
    a = [C.decompose_hangul(C.normalize(t)) for t in tr_txt]
    b = [C.decompose_hangul(C.normalize(t)) for t in te_txt]
    v = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=2,
                        sublinear_tf=True)
    return v.fit_transform(a), v.transform(b)


def feats_charwb(tr_txt, te_txt):
    a = [C.normalize(t) for t in tr_txt]
    b = [C.normalize(t) for t in te_txt]
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 5), min_df=2,
                        sublinear_tf=True)
    return v.fit_transform(a), v.transform(b)


def feats_counts(tr_txt, te_txt):
    a = [C.normalize(t) for t in tr_txt]
    b = [C.normalize(t) for t in te_txt]
    v = CountVectorizer(analyzer="char_wb", ngram_range=(1, 4), min_df=2,
                        max_features=200000)
    return v.fit_transform(a), v.transform(b)


def feats_svd(tr_txt, te_txt, n=350):
    Xa, Xb = C.build_features(tr_txt, te_txt)
    svd = TruncatedSVD(n_components=n, random_state=SEED)
    A = svd.fit_transform(Xa)
    B = svd.transform(Xb)
    nrm = Normalizer()
    return nrm.fit_transform(A), nrm.transform(B)


FEATS = {}


def get_feats(name, tr_txt, te_txt):
    if name not in FEATS:
        t = time.time()
        FEATS[name] = {"main": feats_main, "jamo": feats_jamo,
                       "jw": feats_jamo_word, "js": feats_jamo_short,
                       "cwb": feats_charwb,
                       "counts": feats_counts, "svd": feats_svd}[name](tr_txt, te_txt)
        print(f"  [feat {name}] {FEATS[name][0].shape} in {time.time()-t:.1f}s", flush=True)
    return FEATS[name]


# ------------------------------------------------------------------- models
def scores_ovr(make_model, Xtr, Xte, Y, proba=True):
    """Per-label one-vs-rest OOF + test scores."""
    kf = KFold(n_splits=NFOLD, shuffle=True, random_state=SEED)
    oof = np.zeros((Xtr.shape[0], C.N_LABELS))
    tst = np.zeros((Xte.shape[0], C.N_LABELS))
    folds = list(kf.split(np.arange(Xtr.shape[0])))
    for j in range(C.N_LABELS):
        for trn, val in folds:
            m = make_model()
            m.fit(Xtr[trn], Y[trn, j])
            oof[val, j] = _score(m, Xtr[val], proba)
        m = make_model()
        m.fit(Xtr, Y[:, j])
        tst[:, j] = _score(m, Xte, proba)
    return oof, tst


def _score(m, X, proba):
    if proba and hasattr(m, "predict_proba"):
        return m.predict_proba(X)[:, 1]
    d = m.decision_function(X)
    return 1.0 / (1.0 + np.exp(-d))


def scores_multi(make_model, Xtr, Xte, Y):
    """Multi-output model (single fit for all labels)."""
    kf = KFold(n_splits=NFOLD, shuffle=True, random_state=SEED)
    oof = np.zeros((Xtr.shape[0], C.N_LABELS))
    tst = np.zeros((Xte.shape[0], C.N_LABELS))
    for trn, val in kf.split(np.arange(Xtr.shape[0])):
        m = make_model()
        m.fit(Xtr[trn], Y[trn])
        oof[val] = np.asarray(m.predict_proba(Xtr[val]))
    m = make_model()
    m.fit(Xtr, Y)
    tst[:] = np.asarray(m.predict_proba(Xte))
    return oof, tst


# name -> (feature set, builder, kind)
MODELS = {
    "lr_main_c4":   ("main", lambda: LogisticRegression(C=4.0, max_iter=3000, solver="liblinear"), "ovr"),
    "lr_main_c12":  ("main", lambda: LogisticRegression(C=12.0, max_iter=3000, solver="liblinear"), "ovr"),
    "lr_main_bal":  ("main", lambda: LogisticRegression(C=4.0, max_iter=3000, solver="liblinear", class_weight="balanced"), "ovr"),
    "svc_main":     ("main", lambda: LinearSVC(C=0.3, max_iter=5000), "ovr"),
    "svc_main_bal": ("main", lambda: LinearSVC(C=0.2, max_iter=5000, class_weight="balanced"), "ovr"),
    "sgd_main":     ("main", lambda: SGDClassifier(loss="modified_huber", alpha=1e-5,
                                                   max_iter=3000, tol=1e-4, random_state=SEED), "ovr"),
    "ridge_main":   ("main", lambda: RidgeClassifier(alpha=1.0), "ovr"),
    "lr_jamo":      ("jamo", lambda: LogisticRegression(C=6.0, max_iter=3000, solver="liblinear"), "ovr"),
    "svc_jamo":     ("jamo", lambda: LinearSVC(C=0.3, max_iter=5000), "ovr"),
    "cnb_counts":   ("counts", lambda: ComplementNB(alpha=0.5), "ovr"),
    "lr_jamo_c2":   ("jamo", lambda: LogisticRegression(C=2.0, max_iter=3000, solver="liblinear"), "ovr"),
    "lr_jamo_c16":  ("jamo", lambda: LogisticRegression(C=16.0, max_iter=3000, solver="liblinear"), "ovr"),
    "lr_jamo_bal":  ("jamo", lambda: LogisticRegression(C=6.0, max_iter=3000, solver="liblinear", class_weight="balanced"), "ovr"),
    "svc_jamo_bal": ("jamo", lambda: LinearSVC(C=0.2, max_iter=5000, class_weight="balanced"), "ovr"),
    "lr_jw":        ("jw",   lambda: LogisticRegression(C=6.0, max_iter=3000, solver="liblinear"), "ovr"),
    "lr_jw_bal":    ("jw",   lambda: LogisticRegression(C=6.0, max_iter=3000, solver="liblinear", class_weight="balanced"), "ovr"),
    "svc_jw":       ("jw",   lambda: LinearSVC(C=0.3, max_iter=5000), "ovr"),
    "lr_js":        ("js",   lambda: LogisticRegression(C=6.0, max_iter=3000, solver="liblinear"), "ovr"),
    "svc_js_bal":   ("js",   lambda: LinearSVC(C=0.2, max_iter=5000, class_weight="balanced"), "ovr"),
    "lr_cwb":       ("cwb",  lambda: LogisticRegression(C=6.0, max_iter=3000, solver="liblinear"), "ovr"),
    "svc_cwb_bal":  ("cwb",  lambda: LinearSVC(C=0.2, max_iter=5000, class_weight="balanced"), "ovr"),
    "sgd_jamo":     ("jamo", lambda: SGDClassifier(loss="modified_huber", alpha=3e-6, max_iter=4000, tol=1e-4, random_state=SEED), "ovr"),
    "mlp_svd":      ("svd", lambda: MLPClassifier(hidden_layer_sizes=(384,), alpha=1e-4,
                                                 max_iter=400, random_state=SEED,
                                                 early_stopping=True, n_iter_no_change=15,
                                                 learning_rate_init=2e-3), "multi"),
    "mlp_main":     ("main", lambda: MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-4,
                                                  max_iter=60, random_state=SEED,
                                                  learning_rate_init=1e-3), "multi"),
}


def run(names, tr, te, Y):
    out = {}
    for name in names:
        path = os.path.join(CACHE, name + ".npz")
        if os.path.exists(path):
            d = np.load(path)
            out[name] = (d["oof"], d["tst"])
            print(f"[cached] {name}  oof macroF1(tuned)={d['score']:.4f}", flush=True)
            continue
        fname, builder, kind = MODELS[name]
        Xtr, Xte = get_feats(fname, tr.sentence, te.sentence)
        t = time.time()
        if kind == "ovr":
            oof, tst = scores_ovr(builder, Xtr, Xte, Y)
        else:
            oof, tst = scores_multi(builder, Xtr, Xte, Y)
        thr, sc = C.tune_thresholds(Y, oof, rounds=2)
        np.savez_compressed(path, oof=oof, tst=tst, score=sc)
        out[name] = (oof, tst)
        print(f"[done] {name}  oof macroF1(tuned)={sc:.4f}  ({time.time()-t:.0f}s)", flush=True)
    return out


if __name__ == "__main__":
    tr, te = C.load(os.environ.get("DATA_DIR", "."))
    Y = C.labels_to_matrix(tr.labels)
    names = sys.argv[1:] or list(MODELS)
    run(names, tr, te, Y)

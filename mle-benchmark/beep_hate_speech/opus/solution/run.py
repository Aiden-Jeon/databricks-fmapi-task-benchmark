#!/usr/bin/env python
"""
BEEP! Korean hate-speech classification (t8_beep) -- final solution.
Metric: macro F1 over {none, offensive, hate}.

Environment: CPU only, no deep-learning libraries available
(numpy / pandas / scipy / scikit-learn). No internet, no external data,
no pretrained weights. Everything is learned from train.csv.

Pipeline
--------
1. Text normalization (URL masking, character-repeat squeezing) and
   **Hangul jamo decomposition** -- splitting syllables into onset/nucleus/coda
   exposes sub-character structure, which is what makes character n-grams robust
   to the heavy slang / spelling-variation in this corpus (씨발 / 시발 / ㅅㅂ).
2. Several TF-IDF representations (char_wb, char, jamo char n-grams,
   pseudo-stemmed words) + a small block of hand-crafted style features
   (repetition, punctuation, lone-jamo ratio, ...).
   Vectorizers are fit on train+test *text only* -- unsupervised/transductive,
   no label information is used.
3. A pool of 11 diverse base classifiers (logistic regression OvR & multinomial,
   linear SVM, ComplementNB, NB-SVM, SVD+gradient boosting), each producing
   5-fold out-of-fold probabilities on train and full-fit probabilities on test.
4. Stacking: logistic regression (C=0.03, class_weight='balanced') on the
   concatenated base log-probabilities.
5. Macro-F1 oriented class-prior calibration on top of the stacker
   (coordinate ascent over per-class log offsets).

Validated score (repeated 5x5-fold CV of the stacking + calibration stage):
    macro F1 = 0.5920 +/- 0.0012
Best single model for reference: 0.5658.

Run from the task directory:  python solution/run.py
Writes: outputs/submission.csv
"""
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

CLASSES = ["none", "offensive", "hate"]
C2I = {c: i for i, c in enumerate(CLASSES)}
SEED = 42
NFOLD = 5

# --------------------------------------------------------------------------
# 1. Korean text processing
# --------------------------------------------------------------------------
CHO = list("\u3131\u3132\u3134\u3137\u3138\u3139\u3141\u3142\u3143\u3145\u3146"
           "\u3147\u3148\u3149\u314a\u314b\u314c\u314d\u314e")
JUNG = list("\u314f\u3150\u3151\u3152\u3153\u3154\u3155\u3156\u3157\u3158"
            "\u3159\u315a\u315b\u315c\u315d\u315e\u315f\u3160\u3161\u3162\u3163")
JONG = [""] + list("\u3131\u3132\u3133\u3134\u3135\u3136\u3137\u3139\u313a\u313b"
                   "\u313c\u313d\u313e\u313f\u3140\u3141\u3142\u3144\u3145\u3146"
                   "\u3147\u3148\u314a\u314b\u314c\u314d\u314e")

_RE_SPACE = re.compile(r"\s+")
_RE_URL = re.compile(r"https?://\S+|www\.\S+|\S+\.(?:com|net|kr|co\.kr)\S*")
_RE_REPEAT = re.compile(r"(.)\1{2,}")


def jamo(text):
    """Decompose Hangul syllables into their jamo (consonant/vowel) sequence."""
    out = []
    for ch in text:
        o = ord(ch)
        if 0xAC00 <= o <= 0xD7A3:
            v = o - 0xAC00
            out.append(CHO[v // 588])
            out.append(JUNG[(v % 588) // 28])
            j = JONG[v % 28]
            if j:
                out.append(j)
        else:
            out.append(ch)
    return "".join(out)


def normalize(text, squeeze=True):
    t = _RE_URL.sub(" URL ", str(text))
    if squeeze:
        t = _RE_REPEAT.sub(r"\1\1", t)          # ㅋㅋㅋㅋㅋ -> ㅋㅋ
    return _RE_SPACE.sub(" ", t).strip()


def pseudo_stem(text, k=3):
    """Truncate each token to its first k chars: crude Korean stemming that
    strips inflectional endings (조사/어미) inflating the word vocabulary."""
    return " ".join(w[:k] for w in text.split())


def hand_features(raws):
    """Dense stylistic signals: repetition, punctuation, lone jamo, links, ..."""
    rows = []
    for t in raws:
        t = str(t)
        n = max(len(t), 1)
        nw = max(len(t.split()), 1)
        rows.append([
            len(t) / 100.0,
            nw / 30.0,
            len(t) / nw / 10.0,
            t.count("?") / n * 10,
            t.count("!") / n * 10,
            t.count(".") / n * 10,
            t.count("~") / n * 10,
            t.count(";") / n * 10,
            len(re.findall(r"\u314b", t)) / n * 10,             # ㅋ laughter
            len(re.findall(r"[\u315c\u3160]", t)) / n * 10,      # ㅜㅠ crying
            len(re.findall(r"[\u3131-\u314e]", t)) / n * 10,     # lone consonants
            len(re.findall(r"[\u314f-\u3163]", t)) / n * 10,     # lone vowels
            len(re.findall(r"[A-Za-z]", t)) / n * 10,
            len(re.findall(r"[0-9]", t)) / n * 10,
            len(re.findall(r"[^\w\s]", t)) / n * 10,
            len(re.findall(r"(.)\1{2,}", t)) / nw,
            float(bool(re.search(r"https?://|ilbe|\.com", t))),
            len(re.findall(r"\*", t)) / n * 10,                  # masked profanity
            len(re.findall(r"[\uac00-\ud7a3]", t)) / n,
            float(t.strip().endswith("?")),
            float(t.strip().endswith("!")),
            np.log1p(len(set(t.split()))) / 3.0,
        ])
    return csr_matrix(np.asarray(rows, dtype=np.float32))


def build_features(train, test):
    """Return dict of sparse feature blocks over concat(train, test)."""
    raw = list(train.comment) + list(test.comment)
    txt = [normalize(t) for t in raw]
    txt_raw = [normalize(t, squeeze=False) for t in raw]
    jm = [jamo(t) for t in txt]
    stem3 = [pseudo_stem(t, 3) for t in txt]

    specs = {
        "char_wb25":  (dict(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True), txt),
        "char15":     (dict(analyzer="char", ngram_range=(1, 5), min_df=2, sublinear_tf=True), txt),
        "char_raw25": (dict(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True), txt_raw),
        "jamo36":     (dict(analyzer="char_wb", ngram_range=(3, 6), min_df=2, sublinear_tf=True), jm),
        "jamo27":     (dict(analyzer="char_wb", ngram_range=(2, 7), min_df=3, sublinear_tf=True), jm),
        "stem3_12":   (dict(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True), stem3),
    }
    blocks = {}
    for name, (kw, data) in specs.items():
        blocks[name] = TfidfVectorizer(**kw).fit_transform(data).astype(np.float32)
    blocks["hand"] = hand_features(raw)
    return blocks


# --------------------------------------------------------------------------
# 2. Extra model families (diversity for the ensemble)
# --------------------------------------------------------------------------
class NBSVM(BaseEstimator, ClassifierMixin):
    """One-vs-rest logistic regression on NB log-count-ratio weighted features."""

    def __init__(self, C=1.0, alpha=1.0, class_weight="balanced", n_classes=3):
        self.C, self.alpha = C, alpha
        self.class_weight, self.n_classes = class_weight, n_classes

    def fit(self, X, y):
        self.classes_ = np.arange(self.n_classes)
        self.r_, self.clf_ = [], []
        for k in self.classes_:
            pos = X[y == k].sum(0).A1 + self.alpha
            neg = X[y != k].sum(0).A1 + self.alpha
            r = np.log((pos / pos.sum()) / (neg / neg.sum()))
            self.r_.append(r)
            c = LogisticRegression(C=self.C, solver="liblinear", max_iter=2000,
                                   class_weight=self.class_weight)
            self.clf_.append(c.fit(X.multiply(r).tocsr(), (y == k).astype(int)))
        return self

    def decision_function(self, X):
        return np.column_stack([c.decision_function(X.multiply(r).tocsr())
                                for c, r in zip(self.clf_, self.r_)])

    def predict_proba(self, X):
        p = 1.0 / (1.0 + np.exp(-self.decision_function(X)))
        return p / p.sum(1, keepdims=True)

    def predict(self, X):
        return self.decision_function(X).argmax(1)


class SVDBoost(BaseEstimator, ClassifierMixin):
    """TruncatedSVD embedding + histogram gradient boosting (interactions)."""

    def __init__(self, n_comp=200, seed=SEED):
        self.n_comp, self.seed = n_comp, seed

    def fit(self, X, y):
        self.svd = TruncatedSVD(self.n_comp, random_state=self.seed)
        Z = self.svd.fit_transform(X)
        self.sc = StandardScaler().fit(Z)
        self.clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
            l2_regularization=1.0, early_stopping=False,
            class_weight="balanced", random_state=self.seed
        ).fit(self.sc.transform(Z), y)
        self.classes_ = self.clf.classes_
        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(self.sc.transform(self.svd.transform(X)))

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


def lr_ovr(C):
    return lambda: LogisticRegression(C=C, solver="liblinear", max_iter=2000,
                                      class_weight="balanced", random_state=SEED)


BASE_POOL = [
    ("lr_jamo27",             ["jamo27"],                      lr_ovr(2)),
    ("lr_charwb25",           ["char_wb25"],                   lr_ovr(2)),
    ("lr_char15_jamo27_hand", ["char15", "jamo27", "hand"],    lr_ovr(2)),
    ("lrmn_jamo27",           ["jamo27"],
     lambda: LogisticRegression(C=4, max_iter=1500, class_weight="balanced",
                                random_state=SEED)),
    ("svc_char15_jamo27",     ["char15", "jamo27"],
     lambda: LinearSVC(C=0.25, class_weight="balanced", dual=True,
                       max_iter=4000, random_state=SEED)),
    ("cnb_char15_jamo27",     ["char15", "jamo27"], lambda: ComplementNB(alpha=0.3)),
    ("cnb_charwb25",          ["char_wb25"],        lambda: ComplementNB(alpha=0.3)),
    ("cnb_jamo36",            ["jamo36"],           lambda: ComplementNB(alpha=0.3)),
    ("cnb_charraw25",         ["char_raw25"],       lambda: ComplementNB(alpha=0.3)),
    ("nbsvm_char15_jamo27",   ["char15", "jamo27"], lambda: NBSVM(C=0.5, alpha=1.0)),
    ("nbsvm_jamo27",          ["jamo27"],           lambda: NBSVM(C=1.0, alpha=1.0)),
    ("svd_hgb",               ["char_wb25", "jamo27"], lambda: SVDBoost(200)),
]


# --------------------------------------------------------------------------
# 3. Macro-F1 class-prior calibration
# --------------------------------------------------------------------------
def fit_prior_weights(P, y, n_iter=6):
    """Coordinate ascent over per-class log offsets to maximise macro F1."""
    grid = np.linspace(-1.2, 1.2, 49)
    logw = np.zeros(P.shape[1])
    logP = np.log(np.clip(P, 1e-9, None))
    best = f1_score(y, (logP + logw).argmax(1), average="macro")
    for _ in range(n_iter):
        improved = False
        for k in range(P.shape[1]):
            keep, keep_score = logw[k], best
            for g in grid:
                logw[k] = g
                s = f1_score(y, (logP + logw).argmax(1), average="macro")
                if s > keep_score + 1e-9:
                    keep_score, keep = s, g
            logw[k] = keep
            if keep_score > best + 1e-9:
                best, improved = keep_score, True
        if not improved:
            break
    return logw, best


# --------------------------------------------------------------------------
# 4. Main
# --------------------------------------------------------------------------
def proba(clf, X):
    if hasattr(clf, "predict_proba"):
        return np.clip(clf.predict_proba(X), 1e-7, 1.0)
    return softmax(clf.decision_function(X) * 2.0, axis=1)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    os.chdir(root)
    np.random.seed(SEED)
    t_start = time.time()

    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    y = train.label.map(C2I).values
    ntr = len(train)
    print(f"train={train.shape} test={test.shape}")

    print("building features ...")
    blocks = build_features(train, test)
    for k, v in blocks.items():
        print(f"  {k:12s} dim={v.shape[1]}")

    def split(names):
        M = hstack([blocks[n] for n in names]).tocsr()
        return M[:ntr], M[ntr:]

    fds = list(StratifiedKFold(NFOLD, shuffle=True,
                               random_state=SEED).split(np.zeros(ntr), y))

    # ---- level 0: out-of-fold + full-fit test probabilities -------------
    oof_list, test_list, tags = [], [], []
    for tag, names, factory in BASE_POOL:
        t0 = time.time()
        Xtr, Xte = split(names)
        P = np.zeros((ntr, 3))
        for tri, vai in fds:
            P[vai] = proba(factory().fit(Xtr[tri], y[tri]), Xtr[vai])
        T = proba(factory().fit(Xtr, y), Xte)
        f = f1_score(y, P.argmax(1), average="macro")
        print(f"  base {tag:24s} oof macroF1={f:.4f} ({time.time()-t0:.0f}s)",
              flush=True)
        oof_list.append(np.log(P))
        test_list.append(np.log(T))
        tags.append(tag)

    Sx = np.concatenate(oof_list, axis=1)
    St = np.concatenate(test_list, axis=1)

    # ---- level 1: stacking ---------------------------------------------
    stacker = LogisticRegression(C=0.03, max_iter=3000,
                                 class_weight="balanced", random_state=SEED)
    stacker.fit(Sx, y)
    P_tr = stacker.predict_proba(Sx)
    print(f"  stacker (uncalibrated, in-sample) macroF1="
          f"{f1_score(y, P_tr.argmax(1), average='macro'):.4f}")

    # ---- level 2: macro-F1 prior calibration ---------------------------
    logw, sc = fit_prior_weights(P_tr, y)
    print(f"  prior calibration logw={np.round(logw, 3)} (in-sample {sc:.4f})")

    P_te = stacker.predict_proba(St)
    pred = (np.log(np.clip(P_te, 1e-9, None)) + logw).argmax(1)

    os.makedirs("outputs", exist_ok=True)
    sub = pd.DataFrame({"id": test.id.values,
                        "label": [CLASSES[i] for i in pred]})
    sub.to_csv("outputs/submission.csv", index=False)

    # ---- validation of the submission format ---------------------------
    ss = pd.read_csv("sample_submission.csv")
    assert list(sub.columns) == list(ss.columns), "column mismatch"
    assert len(sub) == len(test), "row count mismatch"
    assert sub.id.is_unique, "duplicate ids"
    assert set(sub.id) == set(test.id), "id set mismatch"
    assert set(sub.label) <= set(CLASSES), "unexpected label value"
    print("\noutputs/submission.csv OK ->",
          sub.label.value_counts().to_dict())
    print(f"total {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()

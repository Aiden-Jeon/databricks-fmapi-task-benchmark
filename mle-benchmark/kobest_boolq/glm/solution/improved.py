"""Improved model with handcrafted features + TF-IDF for KoBEST BoolQ."""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score
import unicodedata
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "train.csv")
TEST = os.path.join(ROOT, "test.csv")
SUB = os.path.join(ROOT, "outputs", "submission.csv")
RANDOM_STATE = 42

# ---------- helpers ----------
JAMO_LEAD = set("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
JAMO_VOWEL = set("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
JAMO_TAIL = set(" ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")

def split_jamo(s):
    out = []
    for ch in s:
        if "가" <= ch <= "힣":
            code = ord(ch) - ord("가")
            lead = code // 588
            vowel = (code % 588) // 28
            tail = code % 28
            out.append(list(JAMO_LEAD)[lead])
            out.append(list(JAMO_VOWEL)[vowel])
            if tail != 0:
                out.append(list(JAMO_TAIL)[tail])
        else:
            out.append(ch)
    return "".join(out)

def normalize(s):
    s = unicodedata.normalize("NFC", str(s))
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize(s):
    return re.findall(r"\w+", s)

def overlap_count(p, q):
    wp = set(tokenize(p))
    wq = set(tokenize(q))
    if not wq:
        return 0, 0, 0, 0
    inter = wp & wq
    return len(inter), len(inter) / len(wq), len(inter) / max(len(wp), 1), len(wp & wq) / len(wq)

def char_overlap(p, q):
    cp = set(p)
    cq = set(q)
    if not cq:
        return 0, 0
    inter = cp & cq
    return len(inter), len(inter) / len(cq)

def neg_cues(q):
    cues = ["아니", "없", "못", "안", "아닌", "아닐", "거짓", "틀린", "없다", "없다.", "없나요", "없습니까", "없을까", "없음", "없는", "아니다", "아니라"]
    ql = q.lower()
    return [1 if c in ql else 0 for c in cues]

def n_words(s):
    return len(tokenize(s))

def n_chars(s):
    return len(s)

def make_features(df):
    feats = []
    for _, row in df.iterrows():
        p = normalize(row["paragraph"])
        q = normalize(row["question"])
        io, io_q, io_p, iou = overlap_count(p, q)
        co, co_q = char_overlap(p, q)
        f = [
            n_words(p), n_words(q),
            n_chars(p), n_chars(q),
            io, io_q, io_p, iou,
            co, co_q,
            n_words(p) - n_words(q),
            n_chars(p) - n_chars(q),
            1 if "?" in q else 0,
            1 if q.endswith("?") else 0,
            1 if q.endswith(".") else 0,
            1 if q.endswith("?") and len(q) > 0 else 0,
        ]
        f.extend(neg_cues(q))
        feats.append(f)
    return np.array(feats, dtype=float)


def build_tfidf_features(train_text, test_text):
    word_vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 3), min_df=2,
                               max_df=0.95, max_features=80000, analyzer="word",
                               token_pattern=r"(?u)\b\w+\b")
    char_vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(2, 4), min_df=2,
                               max_df=0.95, max_features=80000, analyzer="char_wb")
    jamo_vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(2, 4), min_df=2,
                               max_df=0.95, max_features=40000, analyzer="char_wb",
                               preprocessor=split_jamo)
    Xw = word_vec.fit_transform(train_text)
    Xw_te = word_vec.transform(test_text)
    Xc = char_vec.fit_transform(train_text)
    Xc_te = char_vec.transform(test_text)
    Xj = jamo_vec.fit_transform(train_text)
    Xj_te = jamo_vec.transform(test_text)
    return hstack([Xw, Xc, Xj]).tocsr(), hstack([Xw_te, Xc_te, Xj_te]).tocsr()


def main():
    train = pd.read_csv(TRAIN)
    test = pd.read_csv(TEST)
    train["text"] = train["paragraph"].astype(str) + " " + train["question"].astype(str)
    test["text"] = test["paragraph"].astype(str) + " " + test["question"].astype(str)
    train["text"] = train["text"].str.replace("\n", " ")
    test["text"] = test["text"].str.replace("\n", " ")

    Xtf, Xtf_te = build_tfidf_features(train["text"], test["text"])
    Xhf_train = make_features(train)
    Xhf_test = make_features(test)

    scaler = StandardScaler()
    Xhf_train_s = scaler.fit_transform(Xhf_train)
    Xhf_test_s = scaler.transform(Xhf_test)

    X = hstack([Xtf, csr_matrix(Xhf_train_s)]).tocsr()
    X_te = hstack([Xtf_te, csr_matrix(Xhf_test_s)]).tocsr()
    y = train["label"].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    models = {
        "lr_c2": LogisticRegression(C=2.0, max_iter=3000, solver="liblinear", random_state=RANDOM_STATE),
        "lr_c4": LogisticRegression(C=4.0, max_iter=3000, solver="liblinear", random_state=RANDOM_STATE),
        "lr_c8": LogisticRegression(C=8.0, max_iter=3000, solver="liblinear", random_state=RANDOM_STATE),
        "lr_saga_l2": LogisticRegression(C=4.0, max_iter=5000, solver="saga", random_state=RANDOM_STATE),
        "linearsvc": LinearSVC(C=1.0, max_iter=5000, random_state=RANDOM_STATE, dual="auto"),
    }
    results = {}
    oof_preds = {}
    for name, clf in models.items():
        s = cross_val_score(clf, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
        results[name] = s.mean()
        oof = cross_val_predict(clf, X, y, cv=skf, method="predict" if not isinstance(clf, LinearSVC) else "decision_function", n_jobs=-1)
        oof_preds[name] = oof
        print(f"{s.mean():.4f} +/- {s.std():.4f}  {name}", file=sys.stderr)

    # ensemble simple average of decision values / probabilities
    # Convert OOF to binary for ensemble
    bin_preds = {}
    for name, oof in oof_preds.items():
        if oof.ndim > 1 and oof.shape[1] == 2:
            oof = oof[:, 1]
        bin_preds[name] = (oof > 0).astype(int)
    # majority vote
    vote = np.zeros(len(y))
    for name, bp in bin_preds.items():
        vote += bp
    vote = (vote >= len(bin_preds) / 2).astype(int)
    acc = accuracy_score(y, vote)
    print(f"Ensemble vote acc: {acc:.4f}", file=sys.stderr)

    # average of decision functions
    dec_sum = np.zeros(len(y))
    for name, oof in oof_preds.items():
        if oof.ndim > 1 and oof.shape[1] == 2:
            oof = oof[:, 1]
        # normalize
        dec_sum += (oof - oof.mean()) / (oof.std() + 1e-9)
    dec_vote = (dec_sum > 0).astype(int)
    acc2 = accuracy_score(y, dec_vote)
    print(f"Ensemble dec-avg acc: {acc2:.4f}", file=sys.stderr)

    # Fit final models on full train and average decision values on test
    final_decs = []
    final_bins = []
    for name, clf in models.items():
        clf.fit(X, y)
        if hasattr(clf, "predict_proba"):
            d = clf.predict_proba(X_te)[:, 1]
            b = (d > 0.5).astype(int)
        elif hasattr(clf, "decision_function"):
            d = clf.decision_function(X_te)
            b = (d > 0).astype(int)
        else:
            d = clf.predict(X_te).astype(float)
            b = d.astype(int)
        final_decs.append(d)
        final_bins.append(b)
    # Majority vote on binary predictions
    vote_sum = np.sum(np.vstack(final_bins), axis=0)
    preds = (vote_sum >= len(final_bins) / 2).astype(int)

    out = pd.DataFrame({"id": test["id"], "label": preds.astype(int)})
    os.makedirs(os.path.dirname(SUB), exist_ok=True)
    out.to_csv(SUB, index=False)
    print("Saved", SUB, "with", len(out), "rows", file=sys.stderr)
    print("label dist", out.label.value_counts().to_dict(), file=sys.stderr)


if __name__ == "__main__":
    main()

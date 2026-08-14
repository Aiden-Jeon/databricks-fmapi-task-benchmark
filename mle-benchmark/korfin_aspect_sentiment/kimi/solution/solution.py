# -*- coding: utf-8 -*-
"""
KorFin-ASC: Aspect-based sentiment analysis on Korean financial news.
Approach: TF-IDF (char n-grams + word n-grams) features on aspect-marked text
          + Logistic Regression / Linear SVM ensemble.
Metric: Macro F1 (3 classes: NEGATIVE, NEUTRAL, POSITIVE)
"""
import argparse
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

LABELS = ["NEGATIVE", "NEUTRAL", "POSITIVE"]


def mark_aspect(sentence: str, aspect: str) -> str:
    """Wrap the aspect occurrence(s) in the sentence with special tokens."""
    s = str(sentence)
    a = str(aspect)
    if a and a in s:
        s = s.replace(a, f" __ASP__{a}__ASP__ ")
    return f"{s} __ASPECT__ {a}"


def build_features(train_texts, test_texts):
    # char n-grams: robust for Korean (captures morphemes/endings)
    char_vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5),
        min_df=2, max_df=0.98, sublinear_tf=True, max_features=200000,
    )
    # word n-grams
    word_vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), token_pattern=r"(?u)\S+",
        min_df=2, max_df=0.98, sublinear_tf=True, max_features=100000,
    )
    Xc_tr = char_vec.fit_transform(train_texts)
    Xc_te = char_vec.transform(test_texts)
    Xw_tr = word_vec.fit_transform(train_texts)
    Xw_te = word_vec.transform(test_texts)
    Xtr = hstack([Xc_tr, Xw_tr]).tocsr()
    Xte = hstack([Xc_te, Xw_te]).tocsr()
    return Xtr, Xte


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cv_eval(X, y, seed=42):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_lr = np.zeros((len(y), 3))
    oof_svc = np.zeros((len(y), 3))
    for tr_idx, va_idx in skf.split(X, y):
        lr = LogisticRegression(C=2.0, max_iter=3000, class_weight="balanced")
        lr.fit(X[tr_idx], y[tr_idx])
        oof_lr[va_idx] = lr.predict_proba(X[va_idx])
        svc = LinearSVC(C=0.5, class_weight="balanced")
        svc.fit(X[tr_idx], y[tr_idx])
        oof_svc[va_idx] = softmax(svc.decision_function(X[va_idx]))
    f1_lr = f1_score(y, oof_lr.argmax(1), average="macro")
    f1_svc = f1_score(y, oof_svc.argmax(1), average="macro")
    ens = 0.5 * oof_lr + 0.5 * oof_svc
    f1_ens = f1_score(y, ens.argmax(1), average="macro")
    print(f"CV macro-F1  LR={f1_lr:.4f}  SVC={f1_svc:.4f}  ENS={f1_ens:.4f}")
    return f1_ens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train.csv")
    ap.add_argument("--test", default="test.csv")
    ap.add_argument("--out", default="outputs/submission.csv")
    ap.add_argument("--skip-cv", action="store_true")
    args = ap.parse_args()

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)

    tr_texts = [mark_aspect(s, a) for s, a in zip(train.sentence, train.aspect)]
    te_texts = [mark_aspect(s, a) for s, a in zip(test.sentence, test.aspect)]

    Xtr, Xte = build_features(tr_texts, te_texts)
    y = train.label.map({l: i for i, l in enumerate(LABELS)}).values

    if not args.skip_cv:
        cv_eval(Xtr, y)

    lr = LogisticRegression(C=2.0, max_iter=3000, class_weight="balanced")
    lr.fit(Xtr, y)
    p_lr = lr.predict_proba(Xte)

    svc = LinearSVC(C=0.5, class_weight="balanced")
    svc.fit(Xtr, y)
    p_svc = softmax(svc.decision_function(Xte))

    pred = (0.5 * p_lr + 0.5 * p_svc).argmax(1)
    sub = pd.DataFrame({"id": test.id, "label": [LABELS[i] for i in pred]})
    sub.to_csv(args.out, index=False)
    print("saved", args.out, sub.label.value_counts().to_dict())


if __name__ == "__main__":
    main()

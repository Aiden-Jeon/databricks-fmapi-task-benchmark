#!/usr/bin/env python3
"""
KoBEST WiC (Word-in-Context) baseline solution.

Task: Predict whether the same target `word` is used with the SAME meaning (1)
or DIFFERENT meaning (0) in two contexts.

Approach (no internet / no pretrained weights):
- Build char-level TF-IDF representations of the context window around the
  bracketed target word.
- Compute multiple cosine-similarity features between context_1 and context_2
  representations (different window sizes and ngram ranges).
- Combine similarity features with simple length features and train a
  LogisticRegression classifier using GroupKFold (grouped by `word`) for
  model selection, then refit on all training data and predict on test.
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.dirname(HERE)
TRAIN = os.path.join(TASK_DIR, "train.csv")
TEST = os.path.join(TASK_DIR, "test.csv")
OUT = os.path.join(TASK_DIR, "outputs", "submission.csv")


def get_window(ctx, word, w=30):
    """Return the context window of size 2*w around the bracketed target word.

    If the bracketed target is not found, return the full context unchanged.
    """
    pat = "[" + word + "]"
    i = ctx.find(pat)
    if i < 0:
        return ctx
    before = ctx[:i]
    after = ctx[i + len(pat):]
    # Take up to w chars on each side of the target position.
    return before[-w:] + after[:w]


def build_sim_features(df, windows=(10, 20, 30, 50, 100), ngrams=((1, 5), (2, 4), (2, 3))):
    """Build a matrix of cosine-similarity features between context_1 and
    context_2 windows for several (window, ngram) configurations."""
    feats = []
    for w in windows:
        w1 = df.apply(lambda r: get_window(r["context_1"], r["word"], w), axis=1).tolist()
        w2 = df.apply(lambda r: get_window(r["context_2"], r["word"], w), axis=1).tolist()
        for ng in ngrams:
            vec = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=ng,
                min_df=2,
                sublinear_tf=True,
                norm="l2",
            )
            W1 = vec.fit_transform(w1)
            W2 = vec.transform(w2)
            # cosine similarity per row (vectors are L2-normalized by TfidfVectorizer)
            dots = np.asarray(W1.multiply(W2).sum(axis=1)).ravel()
            feats.append(dots)
    feats = np.column_stack(feats)
    return feats


def build_extra_features(df):
    """Simple non-text features: lengths and length difference."""
    len1 = df["context_1"].str.len().values.astype(float)
    len2 = df["context_2"].str.len().values.astype(float)
    lendiff = np.abs(len1 - len2)
    return np.column_stack([len1, len2, lendiff, len1 + len2])


def main():
    print("Loading data...")
    train = pd.read_csv(TRAIN)
    test = pd.read_csv(TEST)
    print(f"train={train.shape} test={test.shape}")

    y = train["label"].values.astype(int)
    groups = train["word"].values

    print("Building similarity features (train)...")
    sim_tr = build_sim_features(train)
    extra_tr = build_extra_features(train)
    Xtr = np.column_stack([sim_tr, extra_tr])

    print("Building similarity features (test)...")
    # Fit vectorizers on train, transform train+test together so vocab matches.
    sim_te = build_sim_features(test)
    extra_te = build_extra_features(test)
    Xte = np.column_stack([sim_te, extra_te])

    print(f"feature matrix: train={Xtr.shape} test={Xte.shape}")

    # Cross-validation for sanity checking.
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(train))
    for fold, (tr, va) in enumerate(gkf.split(Xtr, y, groups)):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced"),
        )
        clf.fit(Xtr[tr], y[tr])
        oof[va] = clf.predict_proba(Xtr[va])[:, 1]
    cv_acc = accuracy_score(y, (oof > 0.5).astype(int))
    cv_auc = roc_auc_score(y, oof)
    print(f"CV accuracy={cv_acc:.4f}  AUC={cv_auc:.4f}")

    # Refit on all training data and predict on test.
    print("Fitting final model on all training data...")
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced"),
    )
    clf.fit(Xtr, y)
    pred = clf.predict_proba(Xte)[:, 1]
    label = (pred > 0.5).astype(int)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out = pd.DataFrame({"id": test["id"], "label": label})
    out.to_csv(OUT, index=False)
    print(f"Wrote submission to {OUT} with {len(out)} rows")
    print(f"label distribution: {pd.Series(label).value_counts().to_dict()}")


if __name__ == "__mainmain__":
    main()

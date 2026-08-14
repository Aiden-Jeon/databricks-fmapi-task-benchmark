"""Final solution: Multi-label Korean hate-speech classification (Korean UnSmile).

Pipeline
--------
1. Features (TF-IDF, sublinear):
   - char_wb n-grams (1,5), up to 300k
   - word n-grams (1,2), up to 100k
   - jamo-decomposed syllable n-grams (1,4): each Hangul syllable is split into
     its constituent jamo (choseong/jungseong/jongseong) and whitespace-joined;
     this makes the model robust to spelling variation / obfuscation common in
     Korean hate speech. Big gain (+0.02 F1).
2. Model: one LogisticRegression per label (liblinear, class_weight='balanced',
   C=0.5).
3. Thresholds: per-label decision thresholds tuned on 5-fold stratified
   out-of-fold probabilities via coordinate ascent to maximize macro F1
   (with constraints applied).
4. Constraints: 'clean' (label 10) is mutually exclusive with the 9 hate
   labels in the training data, so:
   - if any hate label is predicted -> clean = 0
   - if nothing is predicted -> clean = 1 when P(clean) >= 0.5, else the
     argmax label is kept.
   This constraint was worth ~+0.04 macro F1 on OOF.

5-fold OOF macro F1: ~0.704.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

N_LABELS = 10
RANDOM_STATE = 42

# ------------------------------- jamo utils --------------------------------
CHO = [chr(c) for c in range(0x1100, 0x1113)]
JUNG = [chr(c) for c in range(0x1161, 0x1176)]
JONG = [chr(c) for c in range(0x11A7, 0x11C3)]


def decompose_jamo(s):
    """Split Hangul syllables into constituent jamo, whitespace-joined."""
    out = []
    for ch in s:
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:
            offset = code - 0xAC00
            out.append(CHO[offset // 588])
            out.append(JUNG[(offset % 588) // 28])
            jong = offset % 28
            if jong > 0:
                out.append(JONG[jong])
        else:
            out.append(ch)
    return " ".join(out)


# ------------------------------ label helpers ------------------------------
def load_labels(series):
    return np.array([[int(c) for c in str(s)] for s in series], dtype=np.int64)


def macro_f1(y_true, y_pred):
    f1s = []
    for k in range(N_LABELS):
        tp = int(((y_true[:, k] == 1) & (y_pred[:, k] == 1)).sum())
        fp = int(((y_true[:, k] == 0) & (y_pred[:, k] == 1)).sum())
        fn = int(((y_true[:, k] == 1) & (y_pred[:, k] == 0)).sum())
        if tp == 0:
            f1 = 0.0 if (fp + fn) > 0 else 1.0
        else:
            p, r = tp / (tp + fp), tp / (tp + fn)
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


def apply_constraints(pred, proba, clean_threshold=0.5):
    """Enforce clean/hate mutual exclusivity."""
    hate = pred[:, :9].copy()
    clean = pred[:, 9].copy()
    any_hate = hate.sum(axis=1) > 0
    clean[any_hate] = 0
    none_pred = (hate.sum(axis=1) == 0) & (clean == 0)
    clean[none_pred] = (proba[none_pred, 9] >= clean_threshold).astype(int)
    still_zero = (hate.sum(axis=1) == 0) & (clean == 0)
    if still_zero.any():
        am = proba[still_zero].argmax(axis=1)
        tmp = np.zeros((still_zero.sum(), N_LABELS), dtype=int)
        tmp[np.arange(still_zero.sum()), am] = 1
        hate[still_zero] = tmp[:, :9]
        clean[still_zero] = tmp[:, 9]
    return np.concatenate([hate, clean[:, None]], axis=1)


# ------------------------------- features ----------------------------------
def build_vectorizers(train_texts):
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 5),
                               min_df=2, max_df=0.98, sublinear_tf=True,
                               max_features=300_000)
    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                               max_df=0.98, sublinear_tf=True,
                               max_features=100_000)
    jamo_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 4), min_df=2,
                               max_df=0.98, sublinear_tf=True,
                               token_pattern=r"\S+")
    char_vec.fit(train_texts)
    word_vec.fit(train_texts)
    jamo_vec.fit([decompose_jamo(t) for t in train_texts])
    return char_vec, word_vec, jamo_vec


def transform(char_vec, word_vec, jamo_vec, texts):
    Xc = char_vec.transform(texts)
    Xw = word_vec.transform(texts)
    Xj = jamo_vec.transform([decompose_jamo(t) for t in texts])
    return sparse.hstack([Xc, Xw, Xj]).tocsr()


# -------------------------------- models -----------------------------------
def train_models(X, y, C=0.5):
    models = []
    for k in range(N_LABELS):
        clf = LogisticRegression(C=C, class_weight="balanced",
                                 solver="liblinear", random_state=RANDOM_STATE,
                                 max_iter=2000)
        clf.fit(X, y[:, k])
        models.append(clf)
    return models


def predict_proba(models, X):
    return np.column_stack([m.predict_proba(X)[:, 1] for m in models])


def tune_thresholds(y, oof, lo=0.10, hi=0.90, step=0.01):
    """Coordinate ascent on per-label thresholds, maximizing constrained F1."""
    cand = np.arange(lo, hi, step)
    cur = np.full(N_LABELS, 0.5)
    for _ in range(3):
        for k in range(N_LABELS):
            best_local, best_t = -1.0, cur[k]
            for t in cand:
                cur[k] = t
                pred = apply_constraints((oof >= cur).astype(int), oof)
                f1 = macro_f1(y, pred)
                if f1 > best_local:
                    best_local, best_t = f1, t
            cur[k] = best_t
    pred = apply_constraints((oof >= cur).astype(int), oof)
    return cur, macro_f1(y, pred)


# --------------------------------- main ------------------------------------
def main():
    tr = pd.read_csv("train.csv", dtype={"labels": str})
    te = pd.read_csv("test.csv")
    y = load_labels(tr["labels"])

    print("Fitting vectorizers on full train...")
    char_vec, word_vec, jamo_vec = build_vectorizers(tr["sentence"])
    X_tr = transform(char_vec, word_vec, jamo_vec, tr["sentence"])
    X_te = transform(char_vec, word_vec, jamo_vec, te["sentence"])
    print("Feature dims:", X_tr.shape)

    print("Computing 5-fold OOF probabilities...")
    from collections import Counter
    key = np.array(["".join(map(str, r)) for r in y])
    cnt = Counter(key)
    strat = np.array([k if cnt[k] >= 5 else "rare" for k in key], dtype=object)
    oof = np.zeros((len(y), N_LABELS))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for tr_i, va_i in skf.split(X_tr, strat):
        m = train_models(X_tr[tr_i], y[tr_i])
        oof[va_i] = predict_proba(m, X_tr[va_i])

    base = macro_f1(y, apply_constraints((oof >= 0.5).astype(int), oof))
    print(f"OOF macro F1 @0.5 (constrained): {base:.4f}")

    print("Tuning per-label thresholds...")
    best_th, best_f1 = tune_thresholds(y, oof)
    print(f"OOF macro F1 tuned: {best_f1:.4f}")
    print("Thresholds:", np.round(best_th, 2).tolist())

    print("Training final models on full data...")
    models = train_models(X_tr, y)
    proba_te = predict_proba(models, X_te)
    pred_te = apply_constraints((proba_te >= best_th).astype(int), proba_te)

    labels_str = ["".join(map(str, row)) for row in pred_te]
    sub = pd.DataFrame({"id": te["id"], "labels": labels_str})
    os.makedirs("outputs", exist_ok=True)
    sub.to_csv("outputs/submission.csv", index=False)
    print("Wrote outputs/submission.csv, rows:", len(sub))
    print("Predicted label counts:", pred_te.sum(0).tolist())

    os.makedirs("solution/artifacts", exist_ok=True)
    joblib.dump({"models": models, "char_vec": char_vec, "word_vec": word_vec,
                 "jamo_vec": jamo_vec, "thresholds": best_th},
                "solution/artifacts/model.joblib")
    with open("solution/artifacts/cv_score.json", "w") as f:
        json.dump({"oof_macro_f1": best_f1, "oof_macro_f1_at_0.5": base,
                   "thresholds": best_th.tolist()}, f, indent=2)


if __name__ == "__main__":
    main()

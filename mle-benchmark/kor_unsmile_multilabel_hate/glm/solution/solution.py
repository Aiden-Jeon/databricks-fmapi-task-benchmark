#!/usr/bin/env python3
"""Korean UnSmile multi-label classification (final solution).

Approach: TF-IDF char n-grams (analyzer='char') + char_wb n-grams,
per-class logistic regression (one-vs-rest) with class_weight='balanced',
multi-seed cross-validation bagging, per-class threshold tuning on OOF
probabilities to maximize Macro F1.

No internet, no pretrained weights, no external data.
"""

import os
import re
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.dirname(HERE)
TRAIN_CSV = os.path.join(TASK_DIR, "train.csv")
TEST_CSV = os.path.join(TASK_DIR, "test.csv")
SAMPLE_CSV = os.path.join(TASK_DIR, "sample_submission.csv")
OUT_CSV = os.path.join(TASK_DIR, "outputs", "submission.csv")

N_CLASSES = 10
CLASSES = [
    "여성/가족", "남성", "성소수자", "인종/국적", "연령",
    "지역", "종교", "기타 혐오", "악플/욕설", "clean",
]
SEEDS = [42, 7, 123, 99, 2024]
N_SPLITS = 5
C = 6.0
CHAR_NG = (1, 6)
CHAR_WB_NG = (2, 4)
MAX_CHAR = 80000
MAX_CHAR_WB = 60000
MIN_DF = 2


def normalize(text: str) -> str:
    text = str(text).lower()
    text = text.replace("ㆍ", " ").replace("·", " ")
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    return text


def build_features(train_text, test_text):
    char_vec = TfidfVectorizer(
        analyzer="char", ngram_range=CHAR_NG, min_df=MIN_DF,
        sublinear_tf=True, max_features=MAX_CHAR,
    )
    char_wb_vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=CHAR_WB_NG, min_df=MIN_DF,
        sublinear_tf=True, max_features=MAX_CHAR_WB,
    )
    Xt_c = char_vec.fit_transform(train_text)
    Xs_c = char_vec.transform(test_text)
    Xt_w = char_wb_vec.fit_transform(train_text)
    Xs_w = char_wb_vec.transform(test_text)
    Xt = hstack([Xt_c, Xt_w]).tocsr()
    Xs = hstack([Xs_c, Xs_w]).tocsr()
    return Xt, Xs


def per_class_f1(yt, yp):
    if yt.sum() == 0 and yp.sum() == 0:
        return None
    tp = int((yt & yp).sum())
    fp = int((~yt.astype(bool) & yp.astype(bool)).sum())
    fn = int((yt.astype(bool) & ~yp.astype(bool)).sum())
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return 2 * prec * rec / (prec + rec)


def macro_f1(y_true, y_pred):
    f1s = []
    for i in range(y_true.shape[1]):
        f = per_class_f1(y_true[:, i], y_pred[:, i])
        if f is not None:
            f1s.append(f)
    return float(np.mean(f1s)) if f1s else 0.0, f1s


def tune_thresholds(oof_proba, y):
    thresholds = np.full(N_CLASSES, 0.5, dtype=float)
    for c in range(N_CLASSES):
        if y[:, c].sum() == 0:
            continue
        best_t, best_f = 0.5, -1.0
        for t in np.linspace(0.05, 0.95, 91):
            pred = (oof_proba[:, c] >= t).astype(int)
            f = per_class_f1(y[:, c], pred) or 0.0
            if f > best_f:
                best_f = f
                best_t = t
        thresholds[c] = float(best_t)
    return thresholds


def apply_clean_rule(pred):
    pred = pred.copy()
    # clean is mutually exclusive with hate labels in the training data
    pred[:, 9] = pred[:, 9] * (pred[:, :9].sum(axis=1) == 0).astype(int)
    # if no label is set, default to clean (common pattern in data)
    no_label = pred.sum(axis=1) == 0
    pred[no_label, 9] = 1
    return pred


def main():
    print("Loading data...", flush=True)
    train = pd.read_csv(TRAIN_CSV, dtype={"labels": str})
    test = pd.read_csv(TEST_CSV)
    sample = pd.read_csv(SAMPLE_CSV, dtype={"labels": str})

    train_text = train["sentence"].map(normalize).tolist()
    test_text = test["sentence"].map(normalize).tolist()
    y = np.array([list(s) for s in train["labels"]]).astype(int)
    assert y.shape == (len(train), N_CLASSES)
    print(f"train={len(train)} test={len(test)}", flush=True)

    print("Building TF-IDF features (char + char_wb)...", flush=True)
    Xt, Xs = build_features(train_text, test_text)
    print(f"Xt={Xt.shape} Xs={Xs.shape}", flush=True)

    primary = y.argmax(axis=1)
    oof_proba = np.zeros((len(train), N_CLASSES), dtype=float)
    test_proba = np.zeros((len(test), N_CLASSES), dtype=float)

    for si, seed in enumerate(SEEDS):
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for fold, (tr_idx, va_idx) in enumerate(skf.split(Xt, primary)):
            Xt_tr = Xt[tr_idx]
            y_tr = y[tr_idx]
            Xt_va = Xt[va_idx]
            for c in range(N_CLASSES):
                if y_tr[:, c].sum() < 2:
                    continue
                clf = LogisticRegression(
                    C=C, max_iter=2000, solver="liblinear",
                    class_weight="balanced",
                )
                clf.fit(Xt_tr, y_tr[:, c])
                oof_proba[va_idx, c] += clf.predict_proba(Xt_va)[:, 1] / len(SEEDS)
                test_proba[:, c] += clf.predict_proba(Xs)[:, 1] / (
                    len(SEEDS) * N_SPLITS
                )
        print(f"  seed {seed} done", flush=True)

    print("Tuning thresholds on OOF...", flush=True)
    thresholds = tune_thresholds(oof_proba, y)
    print("thresholds:", thresholds.round(2), flush=True)

    oof_pred = (oof_proba >= thresholds).astype(int)
    oof_pred = apply_clean_rule(oof_pred)
    f1, f1s = macro_f1(y, oof_pred)
    print(f"OOF Macro F1: {f1:.4f}", flush=True)
    print("per-class F1:", [round(x, 3) for x in f1s], flush=True)

    test_pred = (test_proba >= thresholds).astype(int)
    test_pred = apply_clean_rule(test_pred)

    labels = ["".join(str(int(x)) for x in row) for row in test_pred]
    for s in labels:
        assert len(s) == N_CLASSES, f"bad label len: {s}"

    out = pd.DataFrame({"id": test["id"], "labels": labels})
    out = out.set_index("id").loc[sample["id"]].reset_index()

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}: {len(out)} rows", flush=True)

    chk = pd.read_csv(OUT_CSV, dtype={"labels": str})
    assert (chk["id"].values == sample["id"].values).all(), "id mismatch"
    assert (chk["labels"].str.len() == N_CLASSES).all(), "bad label length"
    assert (~chk["labels"].str.contains(r"[^01]")).all(), "non 0/1 char"
    print("Validation OK", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Korean UnSmile multi-label hate-speech classification - final solution.

Key idea (discovered from data):
  In the training data the labels form an exclusive three-way structure:
    - `clean` (idx 9) == 1  <=>  all hate labels are 0
    - `악플/욕설` curse (idx 8) == 1  <=>  all other labels (incl. clean) are 0
    - targeted hate (idx 0..7) can co-occur with each other, but never with
      clean nor with curse.
  Every row belongs to exactly one of: {clean, curse, targeted-hate}.

Pipeline:
  1. TF-IDF features: word (1-2 gram) + char_wb (1-2 gram), min_df=1.
  2. A dedicated binary "curse-vs-clean" SVC trained only on rows whose head
     label is clean or curse; used to disambiguate rows with no targeted hate.
  3. Eight independent one-vs-rest calibrated LinearSVC classifiers for the
     targeted hate labels (idx 0..7), each with a per-label C chosen by CV.
  4. Per-label decision thresholds tuned by 5-fold CV to maximize F1.
  5. Final decision logic:
       a. For each targeted label j in 0..7, predict 1 if proba >= threshold_j.
       b. If any targeted label is 1  -> curse=0, clean=0.
       c. Else use curse-vs-clean classifier: curse=1 if p_curse >= t_cc,
          otherwise clean=1.
  6. Multi-seed (3 seeds) averaging for robustness.
  7. Guarantees no all-zero rows (defaults to clean).

Only train.csv is used. No internet / pretrained weights.
Reproducible: fixed seeds, deterministic CV splits.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")

LABEL_NAMES = [
    "여성/가족", "남성", "성소수자", "인종/국적", "연령",
    "지역", "종교", "기타 혐오", "악플/욕설", "clean",
]
N_LABELS = 10
N_TARGETED = 8       # labels 0..7
CURSE_IDX = 8
CLEAN_IDX = 9
RNG = 42
SEEDS = [42, 7, 123]

# Per-label C for the targeted-hate classifiers (idx 0..7), chosen by CV.
TARGETED_C = np.array([8.0, 16.0, 2.0, 16.0, 0.5, 16.0, 1.0, 8.0])
# Curse-vs-clean binary classifier C.
CC_C = 2.0

ROOT = Path(__file__).resolve().parent.parent
TRAIN_CSV = ROOT / "train.csv"
TEST_CSV = ROOT / "test.csv"
OUT_CSV = ROOT / "outputs" / "submission.csv"


def parse_labels(label_series):
    arr = np.zeros((len(label_series), N_LABELS), dtype=np.int8)
    for i, s in enumerate(label_series.astype(str)):
        s = s.strip()
        if len(s) != N_LABELS:
            raise ValueError(f"bad label length at row {i}: {s!r}")
        arr[i] = [int(c) for c in s]
    return arr


def build_features(train_text, test_text):
    """Word (1-2 gram) + char_wb (1-2 gram) TF-IDF, min_df=1, concatenated."""
    word_vec = TfidfVectorizer(
        sublinear_tf=True, strip_accents=None, lowercase=False,
        ngram_range=(1, 2), min_df=1, max_df=0.95, max_features=200000,
        token_pattern=r"(?u)\b\w+\b|[^\w\s]",
    )
    char_vec = TfidfVectorizer(
        sublinear_tf=True, strip_accents=None, lowercase=False,
        analyzer="char_wb", ngram_range=(1, 2), min_df=1, max_df=0.95,
        max_features=200000,
    )
    Xw_tr = word_vec.fit_transform(train_text)
    Xw_te = word_vec.transform(test_text)
    Xc_tr = char_vec.fit_transform(train_text)
    Xc_te = char_vec.transform(test_text)
    X_tr = hstack([Xw_tr, Xc_tr]).tocsr()
    X_te = hstack([Xw_te, Xc_te]).tocsr()
    return X_tr, X_te


def make_head_label(y):
    """3-way head: 0=clean, 1=curse, 2=targeted-hate (any of idx 0..7)."""
    head = np.zeros(len(y), dtype=np.int8)
    head[y[:, CLEAN_IDX] == 1] = 0
    head[y[:, CURSE_IDX] == 1] = 1
    head[y[:, :N_TARGETED].sum(axis=1) > 0] = 2
    return head


def fit_calibrated_svc(X, y_bin, C, seed):
    """Calibrated one-vs-rest LinearSVC; returns calibrated classifier."""
    base = LinearSVC(
        C=C, class_weight="balanced", max_iter=5000,
        random_state=seed, dual="auto",
    )
    clf = CalibratedClassifierCV(base, cv=3, method="sigmoid")
    clf.fit(X, y_bin)
    return clf


def predict_targeted_oof(X, y, splits, seed):
    """OOF probabilities for the 8 targeted labels with per-label C."""
    n = len(y)
    oof = np.zeros((n, N_TARGETED), dtype=np.float32)
    for tr_idx, va_idx in splits:
        for j in range(N_TARGETED):
            clf = fit_calibrated_svc(X[tr_idx], y[tr_idx, j], TARGETED_C[j], seed)
            oof[va_idx, j] = clf.predict_proba(X[va_idx])[:, 1]
    return oof


def predict_cc_oof(X, head_y, splits, seed):
    """OOF P(curse) from a binary SVC trained only on clean+curse rows."""
    n = len(head_y)
    oof = np.zeros(n, dtype=np.float32)
    cc_mask = (head_y == 0) | (head_y == 1)
    for tr_idx, va_idx in splits:
        cc_tr = tr_idx[cc_mask[tr_idx]]
        clf = fit_calibrated_svc(X[cc_tr], head_y[cc_tr], CC_C, seed)
        oof[va_idx] = clf.predict_proba(X[va_idx])[:, 1]
    return oof


def tune_targeted_thresholds(y, oof_t):
    """Per-label threshold maximizing F1 over a fine grid."""
    th = np.full(N_TARGETED, 0.5, dtype=np.float64)
    cand = np.linspace(0.05, 0.75, 141)
    for j in range(N_TARGETED):
        if y[:, j].sum() == 0:
            continue
        best_t, best_f = 0.5, -1.0
        for t in cand:
            f = f1_score(y[:, j], (oof_t[:, j] >= t).astype(np.int8), zero_division=0)
            if f > best_f:
                best_f, best_t = f, t
        th[j] = best_t
    return th


def tune_cc_threshold(y, oof_t, oof_cc, th_t):
    """Find curse-vs-clean threshold maximizing Macro F1."""
    best_s, best_t = -1.0, 0.5
    for tcc in np.linspace(0.2, 0.8, 61):
        pred = build_pred(oof_t, oof_cc, th_t, tcc)
        s = macro_f1_present(y, pred)
        if s > best_s:
            best_s, best_t = s, tcc
    return float(best_t)


def build_pred(oof_t, oof_cc, th_t, tcc):
    """Assemble final 10-bit prediction from targeted OOF, curse-vs-clean OOF."""
    n = oof_t.shape[0]
    pred = np.zeros((n, N_LABELS), dtype=np.int8)
    for j in range(N_TARGETED):
        pred[:, j] = (oof_t[:, j] >= th_t[j]).astype(np.int8)
    any_target = pred[:, :N_TARGETED].sum(axis=1) > 0
    pred[any_target, CURSE_IDX] = 0
    pred[any_target, CLEAN_IDX] = 0
    no_t = ~any_target
    curse_win = oof_cc >= tcc
    pred[curse_win & no_t, CURSE_IDX] = 1
    pred[(~curse_win) & no_t, CLEAN_IDX] = 1
    # safety: no all-zero rows -> default to clean
    all_zero = pred.sum(axis=1) == 0
    pred[all_zero, CLEAN_IDX] = 1
    return pred


def macro_f1_present(yt, yp):
    """Macro F1 over labels present in the ground truth (per spec)."""
    fs = []
    for j in range(N_LABELS):
        if yt[:, j].sum() == 0:
            continue
        fs.append(f1_score(yt[:, j], yp[:, j], zero_division=0))
    return float(np.mean(fs)) if fs else 0.0


def main():
    print("loading data ...", flush=True)
    tr = pd.read_csv(TRAIN_CSV, dtype={"labels": str})
    te = pd.read_csv(TEST_CSV)
    y = parse_labels(tr["labels"])
    train_text = tr["sentence"].astype(str).tolist()
    test_text = te["sentence"].astype(str).tolist()
    print(f"train={len(tr)} test={len(te)} y={y.shape}", flush=True)

    print("building TF-IDF features ...", flush=True)
    X_tr, X_te = build_features(train_text, test_text)
    print(f"X_tr={X_tr.shape} X_te={X_te.shape}", flush=True)

    head_y = make_head_label(y)
    print(f"head dist (clean/curse/targeted): {np.bincount(head_y)}", flush=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
    label_count = y.sum(axis=1)
    splits = list(skf.split(np.zeros(len(y)), label_count))

    print(f"collecting OOF over seeds {SEEDS} ...", flush=True)
    oof_t_list, oof_cc_list = [], []
    for si, seed in enumerate(SEEDS):
        oof_t = predict_targeted_oof(X_tr, y, splits, seed)
        oof_cc = predict_cc_oof(X_tr, head_y, splits, seed)
        oof_t_list.append(oof_t)
        oof_cc_list.append(oof_cc)
        print(f"  seed {seed} done", flush=True)

    oof_t = np.mean(oof_t_list, axis=0)
    oof_cc = np.mean(oof_cc_list, axis=0)

    print("tuning thresholds via CV ...", flush=True)
    th_t = tune_targeted_thresholds(y, oof_t)
    tcc = tune_cc_threshold(y, oof_t, oof_cc, th_t)
    print("targeted thresholds:", [round(float(t), 3) for t in th_t], flush=True)
    print(f"curse-vs-clean threshold: {tcc:.3f}", flush=True)

    cv_pred = build_pred(oof_t, oof_cc, th_t, tcc)
    cv_macro = macro_f1_present(y, cv_pred)
    print(f"CV OOF macro-F1 (present labels): {cv_macro:.4f}", flush=True)

    # per-label F1
    print("per-label CV F1:", flush=True)
    for j in range(N_LABELS):
        if y[:, j].sum() == 0:
            continue
        f = f1_score(y[:, j], cv_pred[:, j], zero_division=0)
        print(f"  {j} {LABEL_NAMES[j]:8s}: F1={f:.3f}", flush=True)

    print("fitting final models on full train ...", flush=True)
    test_t = np.zeros((len(te), N_TARGETED), dtype=np.float32)
    test_cc = np.zeros(len(te), dtype=np.float32)
    for seed in SEEDS:
        for j in range(N_TARGETED):
            clf = fit_calibrated_svc(X_tr, y[:, j], TARGETED_C[j], seed)
            test_t[:, j] += clf.predict_proba(X_te)[:, 1] / len(SEEDS)
        cc_mask = (head_y == 0) | (head_y == 1)
        clf_cc = fit_calibrated_svc(X_tr[cc_mask], head_y[cc_mask], CC_C, seed)
        test_cc += clf_cc.predict_proba(X_te)[:, 1] / len(SEEDS)

    pred = build_pred(test_t, test_cc, th_t, tcc)

    labels_str = ["".join(str(int(v)) for v in row) for row in pred]
    out = pd.DataFrame({"id": te["id"].values, "labels": labels_str})
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV} ({len(out)} rows)", flush=True)

    # self-checks
    assert len(out) == len(te), "row count mismatch"
    assert (out["labels"].str.len() == 10).all(), "bad label length"
    assert out["labels"].str.match(r"^[01]{10}$").all(), "non-binary label"
    assert set(out["id"]) == set(te["id"]), "id mismatch"
    assert out["id"].value_counts().max() == 1, "duplicate ids"
    print("self-checks passed", flush=True)


if __name__ == "__main__":
    main()

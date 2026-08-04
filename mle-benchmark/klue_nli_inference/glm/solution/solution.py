#!/usr/bin/env python
"""
K-MLE-Bench: KLUE-NLI natural language inference solution.

Approach (no external data / no internet / no pretrained weights):
  - TF-IDF char_wb n-grams (2-4) on premise and hypothesis separately.
  - Engineered numeric features: token overlap, char overlap, bigram overlap,
    length ratios, negation cues, subset flags, position flags.
  - Logistic Regression with combined (sparse text + dense numeric) features.
  - 5-fold out-of-fold predictions averaged for test.

Reproducible (fixed seed). Designed to run within time budget.
"""

import os
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

SEED = 42
HERE = Path(__file__).resolve().parent
TASK_DIR = HERE.parent
TRAIN_CSV = TASK_DIR / "train.csv"
TEST_CSV = TASK_DIR / "test.csv"
SAMPLE_CSV = TASK_DIR / "sample_submission.csv"
OUT_CSV = TASK_DIR / "outputs" / "submission.csv"

CLASSES = ["entailment", "neutral", "contradiction"]

NEG_CUES = {
    "안", "못", "않", "없", "아니", "없다", "no", "not", "never", "without",
    "지", "하지", "but", "그러나", "하지만", "아니고", "아니며", "아니라",
    "안됨", "없고", "없으며", "없지", "아닌", "아닙", "없는", "없이",
    "안된", "안됐", "못하", "하지", "않는", "않다", "않고", "않으며",
    "아닌가", "아니함", "없음", "없어", "없어요",
}


def tok(s: str):
    return re.findall(r"[가-힣]+|[a-zA-Z]+|[0-9]+|[^\s가-힣a-zA-Z0-9]", str(s).lower())


def overlap_feats(df: pd.DataFrame) -> np.ndarray:
    feats = []
    for _, row in df.iterrows():
        p = tok(row["premise"])
        h = tok(row["hypothesis"])
        ps = set(p)
        hs = set(h)
        inter = len(ps & hs)
        union = len(ps | hs) or 1
        plen = max(len(p), 1)
        hlen = max(len(h), 1)
        pchar = len(str(row["premise"]))
        hchar = len(str(row["hypothesis"]))
        pneg = sum(1 for t in p if t in NEG_CUES)
        hneg = sum(1 for t in h if t in NEG_CUES)
        p_big = set(zip(p[:-1], p[1:]))
        h_big = set(zip(h[:-1], h[1:]))
        big_inter = len(p_big & h_big)
        big_union = len(p_big | h_big) or 1
        # char-level overlap
        pset_c = set(str(row["premise"]))
        hset_c = set(str(row["hypothesis"]))
        char_inter = len(pset_c & hset_c)
        char_union = len(pset_c | hset_c) or 1
        feats.append([
            inter / plen, inter / hlen, inter / union, inter, len(ps & hs), len(ps), len(hs),
            len(p), len(h), len(h) - len(p), len(p) / max(len(h), 1),
            1.0 if hs.issubset(ps) else 0.0,
            1.0 if ps.issubset(hs) else 0.0,
            pchar, hchar, hchar - pchar, hchar / max(pchar, 1),
            pneg, hneg, hneg - pneg, 1.0 if (pneg + hneg) > 0 else 0.0,
            big_inter, big_inter / max(len(h_big), 1), big_inter / big_union,
            len(p_big), len(h_big),
            1.0 if (len(h) > 0 and h[0] in ps) else 0.0,
            1.0 if (len(h) > 0 and h[-1] in ps) else 0.0,
            char_inter / max(len(pset_c), 1), char_inter / max(len(hset_c), 1),
            char_inter / char_union,
            1.0 if (pneg > 0 and hneg == 0) else 0.0,  # neg only in premise
            1.0 if (hneg > 0 and pneg == 0) else 0.0,  # neg only in hypothesis
            1.0 if (pneg > 0 and hneg > 0) else 0.0,  # neg in both
        ])
    return np.array(feats, dtype=np.float32)


def main():
    t0 = time.time()
    print("Loading data ...", flush=True)
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    print(f"train: {train_df.shape}, test: {test_df.shape}", flush=True)

    y = train_df["label"].values
    le = LabelEncoder()
    le.fit(CLASSES)
    y_enc = le.transform(y)

    print("Building numeric features ...", flush=True)
    f_tr = overlap_feats(train_df)
    f_te = overlap_feats(test_df)
    scaler = StandardScaler()
    f_tr_s = scaler.fit_transform(f_tr)
    f_te_s = scaler.transform(f_te)

    print("Building TF-IDF features ...", flush=True)
    v_p = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True,
        min_df=3, max_features=30000,
    )
    v_h = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True,
        min_df=3, max_features=30000,
    )
    P_tr = v_p.fit_transform(train_df["premise"].fillna(""))
    H_tr = v_h.fit_transform(train_df["hypothesis"].fillna(""))
    P_te = v_p.transform(test_df["premise"].fillna(""))
    H_te = v_h.transform(test_df["hypothesis"].fillna(""))

    Xtr_text = hstack([P_tr, H_tr]).tocsr()
    Xte_text = hstack([P_te, H_te]).tocsr()
    print(f"text features: train={Xtr_text.shape}, test={Xte_text.shape}", flush=True)

    Xtr = hstack([Xtr_text, csr_matrix(f_tr_s)]).tocsr()
    Xte = hstack([Xte_text, csr_matrix(f_te_s)]).tocsr()
    print(f"combined features: train={Xtr.shape}, test={Xte.shape}", flush=True)

    print("Training with 5-fold OOF ...", flush=True)
    n_classes = len(CLASSES)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    test_proba = np.zeros((len(test_df), n_classes))
    accs = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(Xtr, y_enc)):
        m = LogisticRegression(
            C=0.5, max_iter=2000, solver="liblinear",
            random_state=SEED, class_weight="balanced",
        )
        m.fit(Xtr[tr_idx], y_enc[tr_idx])
        va_pred = m.predict(Xtr[va_idx])
        a = accuracy_score(y_enc[va_idx], va_pred)
        accs.append(a)
        test_proba += m.predict_proba(Xte) / skf.n_splits
        print(f"  fold {fold} acc {a:.4f}", flush=True)
    print(f"OOF mean acc {np.mean(accs):.4f} +/- {np.std(accs):.4f}", flush=True)

    final_pred = le.inverse_transform(test_proba.argmax(1))

    sub = pd.DataFrame({"id": test_df["id"].values, "label": final_pred})
    os.makedirs(OUT_CSV.parent, exist_ok=True)
    sub.to_csv(OUT_CSV, index=False)
    print(f"Saved submission to {OUT_CSV}", flush=True)
    print(f"Total time: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

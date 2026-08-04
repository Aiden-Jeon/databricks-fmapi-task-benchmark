"""Experiment 2: pair-wise TF-IDF features (diff + hadamard product)."""
import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack, csr_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import build_pair_features, make_hand_features
from pair_features import build_pairwise_tfidf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def quick_cv(Xtr, y, name, n_splits=3, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    svc_oof = np.zeros(len(y))
    lr_oof = np.zeros(len(y))
    for fold, (tr_idx, va_idx) in enumerate(skf.split(Xtr, y)):
        svc = LinearSVC(C=0.5, max_iter=5000, dual="auto")
        svc.fit(Xtr[tr_idx], y[tr_idx])
        svc_oof[va_idx] = svc.decision_function(Xtr[va_idx])
        lr = LogisticRegression(C=4.0, max_iter=3000, solver="liblinear")
        lr.fit(Xtr[tr_idx], y[tr_idx])
        lr_oof[va_idx] = lr.predict_proba(Xtr[va_idx])[:, 1]
        print(
            f"  {name} fold {fold} svc={accuracy_score(y[va_idx], (svc_oof[va_idx]>0).astype(int)):.4f} "
            f"lr={accuracy_score(y[va_idx], (lr_oof[va_idx]>0.5).astype(int)):.4f}",
            flush=True,
        )
    print(
        f"{name} SVC OOF acc={accuracy_score(y, (svc_oof>0).astype(int)):.4f} "
        f"LR OOF acc={accuracy_score(y, (lr_oof>0.5).astype(int)):.4f}",
        flush=True,
    )
    return svc_oof, lr_oof


def main():
    t0 = time.time()
    train_df = pd.read_csv(os.path.join(ROOT, "train.csv"))
    test_df = pd.read_csv(os.path.join(ROOT, "test.csv"))
    y = train_df["label"].values

    print("pairwise tfidf...", flush=True)
    Xtr_p, Xte_p, cos_tr, cos_te = build_pairwise_tfidf(train_df, test_df)
    print(f"pair tfidf: {Xtr_p.shape} {Xte_p.shape}", flush=True)

    Ftr = make_hand_features(train_df)
    Fte = make_hand_features(test_df)
    # add cosine as a hand feature column
    Ftr2 = np.column_stack([Ftr, cos_tr])
    Fte2 = np.column_stack([Fte, cos_te])

    # combination A: pair tfidf only
    Xtr_a = Xtr_p
    print("== pair tfidf only ==", flush=True)
    quick_cv(Xtr_a, y, "pair_only")

    # combination B: pair tfidf + hand feats
    Xtr_b = hstack([Xtr_p, csr_matrix(Ftr2)]).tocsr()
    print("== pair tfidf + hand ==", flush=True)
    quick_cv(Xtr_b, y, "pair_hand")

    # combination C: full (concat tfidf + diff tfidf + hand) + pair tfidf
    print("full features + pair...", flush=True)
    Xtr_full, Xte_full, _ = build_pair_features(train_df, test_df)
    Xtr_c = hstack([Xtr_full, Xtr_p]).tocsr()
    quick_cv(Xtr_c, y, "full+pair")

    print(f"elapsed {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

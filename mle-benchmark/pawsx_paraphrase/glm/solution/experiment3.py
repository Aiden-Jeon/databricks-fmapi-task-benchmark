"""Experiment 3: combine the best feature sets with an LR/SVC ensemble.

Findings so far:
- LR on (pair_tfidf + hand) ~ 70%
- SVC on (full + pair) ~ 67%
- SVC on full features ~ 68.75% (5-fold)

We build a final pipeline that:
1. Uses 5-fold CV for reliable OOF estimates.
2. Combines multiple feature representations.
3. Ensembles LR + SVC probabilities.
"""
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
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import build_pair_features, make_hand_features, build_tfidf, build_diff_tfidf
from pair_features import build_pairwise_tfidf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def cv_oof(Xtr, y, n_splits=5, seed=42, lr_c=4.0, svc_c=0.5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    lr_oof = np.zeros(len(y))
    svc_oof = np.zeros(len(y))
    for fold, (tr_idx, va_idx) in enumerate(skf.split(Xtr, y)):
        lr = LogisticRegression(C=lr_c, max_iter=3000, solver="liblinear")
        lr.fit(Xtr[tr_idx], y[tr_idx])
        lr_oof[va_idx] = lr.predict_proba(Xtr[va_idx])[:, 1]
        base = LinearSVC(C=svc_c, max_iter=5000, dual="auto")
        svc = CalibratedClassifierCV(base, cv=3)
        svc.fit(Xtr[tr_idx], y[tr_idx])
        svc_oof[va_idx] = svc.predict_proba(Xtr[va_idx])[:, 1]
        print(
            f"  fold {fold} lr={accuracy_score(y[va_idx], (lr_oof[va_idx]>0.5).astype(int)):.4f} "
            f"svc={accuracy_score(y[va_idx], (svc_oof[va_idx]>0.5).astype(int)):.4f}",
            flush=True,
        )
    print(
        f"LR  OOF acc={accuracy_score(y, (lr_oof>0.5).astype(int)):.4f} "
        f"SVC OOF acc={accuracy_score(y, (svc_oof>0.5).astype(int)):.4f}",
        flush=True,
    )
    return lr_oof, svc_oof


def main():
    t0 = time.time()
    train_df = pd.read_csv(os.path.join(ROOT, "train.csv"))
    test_df = pd.read_csv(os.path.join(ROOT, "test.csv"))
    y = train_df["label"].values

    print("building features...", flush=True)
    Xtr_full, Xte_full, _ = build_pair_features(train_df, test_df)
    print(f"full: {Xtr_full.shape}", flush=True)
    Xtr_p, Xte_p, cos_tr, cos_te = build_pairwise_tfidf(train_df, test_df)
    print(f"pair: {Xtr_p.shape}", flush=True)
    Ftr = make_hand_features(train_df)
    Fte = make_hand_features(test_df)
    Ftr2 = np.column_stack([Ftr, cos_tr])
    Fte2 = np.column_stack([Fte, cos_te])
    scaler = StandardScaler()
    Ftr2s = scaler.fit_transform(Ftr2)
    Fte2s = scaler.transform(Fte2)

    # Candidate 1: pair + hand (LR strong)
    print("\n== A: pair + hand (5-fold) ==", flush=True)
    Xtr_a = hstack([Xtr_p, csr_matrix(Ftr2s)]).tocsr()
    lr_a, svc_a = cv_oof(Xtr_a, y)

    # Candidate 2: full + pair (SVC strong)
    print("\n== B: full + pair (5-fold) ==", flush=True)
    Xtr_b = hstack([Xtr_full, Xtr_p]).tocsr()
    lr_b, svc_b = cv_oof(Xtr_b, y)

    # Candidate 3: full + pair + hand
    print("\n== C: full + pair + hand (5-fold) ==", flush=True)
    Xtr_c = hstack([Xtr_full, Xtr_p, csr_matrix(Ftr2s)]).tocsr()
    lr_c, svc_c = cv_oof(Xtr_c, y)

    # Try ensembling the best OOFs across candidates
    print("\n== ensemble search ==", flush=True)
    cands = {
        "A_lr": lr_a, "A_svc": svc_a,
        "B_lr": lr_b, "B_svc": svc_b,
        "C_lr": lr_c, "C_svc": svc_c,
    }
    best = None
    for w_a_lr in np.linspace(0, 1, 5):
        for w_b_svc in np.linspace(0, 1, 5):
            for w_c_lr in np.linspace(0, 1, 5):
                tot = w_a_lr + w_b_svc + w_c_lr
                if tot == 0:
                    continue
                e = (w_a_lr * lr_a + w_b_svc * svc_b + w_c_lr * lr_c) / tot
                acc = accuracy_score(y, (e > 0.5).astype(int))
                if best is None or acc > best[0]:
                    best = (acc, w_a_lr, w_b_svc, w_c_lr)
    print(f"best 3-way acc={best[0]:.4f} w_A_lr={best[1]:.2f} w_B_svc={best[2]:.2f} w_C_lr={best[3]:.2f}", flush=True)

    # also try a simple average of A_lr and C_lr and B_svc
    combos = {
        "A_lr": lr_a, "C_lr": lr_c, "B_svc": svc_b, "C_svc": svc_c, "A_svc": svc_a, "B_lr": lr_b,
    }
    best2 = None
    names = list(combos.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            for w in np.linspace(0, 1, 11):
                e = w * combos[names[i]] + (1 - w) * combos[names[j]]
                acc = accuracy_score(y, (e > 0.5).astype(int))
                if best2 is None or acc > best2[0]:
                    best2 = (acc, names[i], names[j], w)
    print(f"best pair acc={best2[0]:.4f} {best2[1]}*{best2[3]:.2f}+{best2[2]}*{1-best2[3]:.2f}", flush=True)

    print(f"elapsed {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

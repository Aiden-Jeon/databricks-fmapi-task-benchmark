"""Train paraphrase detector: improved features + tuned ensemble."""
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import build_pair_features

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "outputs")


def cv_oof(Xtr, y, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    lr_oof = np.zeros(len(y))
    svc_oof = np.zeros(len(y))
    for fold, (tr_idx, va_idx) in enumerate(skf.split(Xtr, y)):
        lr = LogisticRegression(C=4.0, max_iter=3000, solver="liblinear")
        lr.fit(Xtr[tr_idx], y[tr_idx])
        lr_oof[va_idx] = lr.predict_proba(Xtr[va_idx])[:, 1]
        base = LinearSVC(C=1.0, max_iter=5000, dual="auto")
        svc = CalibratedClassifierCV(base, cv=3)
        svc.fit(Xtr[tr_idx], y[tr_idx])
        svc_oof[va_idx] = svc.predict_proba(Xtr[va_idx])[:, 1]
        print(
            f"  fold {fold} lr={accuracy_score(y[va_idx], (lr_oof[va_idx]>0.5).astype(int)):.4f} "
            f"svc={accuracy_score(y[va_idx], (svc_oof[va_idx]>0.5).astype(int)):.4f}",
            flush=True,
        )
    print(f"LR  OOF acc={accuracy_score(y, (lr_oof>0.5).astype(int)):.4f}", flush=True)
    print(f"SVC OOF acc={accuracy_score(y, (svc_oof>0.5).astype(int)):.4f}", flush=True)
    return lr_oof, svc_oof


def main():
    t0 = time.time()
    train_df = pd.read_csv(os.path.join(ROOT, "train.csv"))
    test_df = pd.read_csv(os.path.join(ROOT, "test.csv"))
    print(f"train {len(train_df)} test {len(test_df)}", flush=True)

    y = train_df["label"].values

    print("building features...", flush=True)
    Xtr, Xte, _ = build_pair_features(train_df, test_df)
    print(f"combined: {Xtr.shape} {Xte.shape}", flush=True)

    lr_oof, svc_oof = cv_oof(Xtr, y)

    # search ensemble weight
    best_w, best_acc = 0.5, -1
    for w in np.linspace(0, 1, 21):
        e = w * lr_oof + (1 - w) * svc_oof
        acc = accuracy_score(y, (e > 0.5).astype(int))
        if acc > best_acc:
            best_acc, best_w = acc, w
    print(f"best ensemble w_lr={best_w:.2f} acc={best_acc:.4f}", flush=True)

    # final models on all data
    print("training final models on full data...", flush=True)
    lr_final = LogisticRegression(C=4.0, max_iter=3000, solver="liblinear")
    lr_final.fit(Xtr, y)
    lr_p = lr_final.predict_proba(Xte)[:, 1]

    svc_base = LinearSVC(C=1.0, max_iter=5000, dual="auto")
    svc_final = CalibratedClassifierCV(svc_base, cv=3)
    svc_final.fit(Xtr, y)
    svc_p = svc_final.predict_proba(Xte)[:, 1]

    proba = best_w * lr_p + (1 - best_w) * svc_p
    pred = (proba > 0.5).astype(int)

    sub = pd.DataFrame({"id": test_df["id"], "label": pred})
    os.makedirs(OUT, exist_ok=True)
    sub.to_csv(os.path.join(OUT, "submission.csv"), index=False)
    print(f"saved submission shape={sub.shape} dist={sub['label'].value_counts().to_dict()}", flush=True)
    print(f"elapsed {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

"""Train KoBEST BoolQ classifier with stratified CV (log loss + accuracy),
then fit on the full train set and persist artifacts for predict.py."""
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss

from features import build_matrix, build_vectorizers

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
RANDOM_STATE = 42


def main():
    train_df = pd.read_csv(ROOT / "train.csv")
    test_df = pd.read_csv(ROOT / "test.csv")
    y = train_df["label"].astype(int).values

    print("building features ...", flush=True)
    vec = build_vectorizers(train_df)
    X, _ = build_matrix(train_df, *vec)
    Xt, _ = build_matrix(test_df, *vec)
    print("train", X.shape, "test", Xt.shape, flush=True)

    C_grid = [0.5, 1.0, 2.0, 4.0]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    best = None
    oof_store = {}
    test_proba_store = {}
    for C in C_grid:
        oof = np.zeros(len(y))
        test_p = np.zeros(len(test_df))
        for tr_idx, va_idx in skf.split(X, y):
            m = LogisticRegression(C=C, max_iter=200, solver="lbfgs",
                                   random_state=RANDOM_STATE)
            m.fit(X[tr_idx], y[tr_idx])
            oof[va_idx] = m.predict_proba(X[va_idx])[:, 1]
            test_p += m.predict_proba(Xt)[:, 1] / skf.n_splits
        acc = accuracy_score(y, oof >= 0.5)
        ll = log_loss(y, oof)
        oof_store[C] = oof
        test_proba_store[C] = test_p
        print(f"C={C}: acc={acc:.4f} logloss={ll:.4f}", flush=True)
        if best is None or ll < best[1]:
            best = (C, ll, acc)

    best_C = best[0]
    print(f"best C={best_C} (logloss={best[1]:.4f}, acc={best[2]:.4f})", flush=True)

    # averaged-CV prediction (bagging across folds) for the best C
    sub = pd.DataFrame({"id": test_df["id"],
                        "label": (test_proba_store[best_C] >= 0.5).astype(int)})
    assert sub["id"].is_unique and len(sub) == len(test_df)
    (ROOT / "outputs").mkdir(exist_ok=True)
    sub.to_csv(ROOT / "outputs" / "submission.csv", index=False)
    print("wrote outputs/submission.csv", sub["label"].value_counts().to_dict(),
          flush=True)

    # final model on full train for reproducibility / later use
    final_model = LogisticRegression(C=best_C, max_iter=200, solver="lbfgs",
                                     random_state=RANDOM_STATE)
    final_model.fit(X, y)
    with open(ROOT / "solution" / "model.pkl", "wb") as f:
        pickle.dump({"vectorizers": vec, "model": final_model, "C": best_C,
                     "cv": {"logloss": best[1], "acc": best[2]}}, f)
    print("saved solution/model.pkl", flush=True)


if __name__ == "__main__":
    main()

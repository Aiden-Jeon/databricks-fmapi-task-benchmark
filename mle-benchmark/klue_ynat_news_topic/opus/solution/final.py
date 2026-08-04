"""Final model: stacked ensemble of 8 TF-IDF linear models -> LogisticRegression meta-learner.

Pipeline
--------
Level 0 (see oof.py `build`): 12 diverse sklearn text classifiers over TF-IDF
  word / char_wb / char n-gram spaces (LinearSVC, RidgeClassifier, ComplementNB,
  SGD-modified-huber, NB-SVM reweighting, binary features, cosine-kNN).  For each, 5-fold out-of-fold decision-function matrices
  (n_train x 7) are produced, plus a test matrix from a model refit on all of train.
Level 1: multinomial LogisticRegression (C=0.1) over the concatenated 12*7 = 84 scores.

Honest nested-CV macro-F1 of this stack: 0.8498 (seed 7) / 0.8490 (seed 21)
(best single level-0 model H: 0.84164)

Reproduce with: bash solution/run_all.sh
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

TASK = "/tmp/kmle/M1_t3_ynat_full_20260804_033458/task"
CACHE = "/tmp/opencode"
KEYS = ["A", "B", "C", "E", "F", "G", "H", "I", "K", "M", "N", "P"]
META_C = 0.1


def load(keys):
    O, T = [], []
    for k in keys:
        z = np.load(f"{CACHE}/oof_{k}.npz", allow_pickle=True)
        o, t, cls = z["oof"], z["test"], z["classes"]
        m, s = o.mean(), o.std()          # global standardization, fit on OOF only
        O.append((o - m) / s)
        T.append((t - m) / s)
    return np.concatenate(O, 1), np.concatenate(T, 1), cls


def main():
    tr = pd.read_csv(f"{TASK}/train.csv")
    te = pd.read_csv(f"{TASK}/test.csv")
    F, FT, cls = load(KEYS)
    c2i = {c: i for i, c in enumerate(cls)}
    yi = np.array([c2i[v] for v in tr.label.values])

    # tol tightened so lbfgs converges fully -> bit-reproducible regardless of
    # BLAS thread count (with the default tol, ~5/9136 borderline rows could flip).
    meta = LogisticRegression(C=META_C, max_iter=20000, tol=1e-9)
    meta.fit(F, yi)
    pred = cls[meta.predict(FT)]

    sub = pd.DataFrame({"id": te.id, "label": pred})
    assert len(sub) == len(te) and set(sub.id) == set(te.id) and not sub.id.duplicated().any()
    assert set(sub.label) <= set(cls)
    sub.to_csv(f"{TASK}/outputs/submission.csv", index=False)
    print("wrote", len(sub), "rows")
    print(sub.label.value_counts().to_dict())


if __name__ == "__main__":
    main()

"""Fast baseline: TF-IDF + One-vs-Rest logistic regression, tuned thresholds."""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

import common as C

DATA = os.environ.get("DATA_DIR", ".")
OUT = os.environ.get("OUT_DIR", "outputs")


def main():
    tr, te = C.load(DATA)
    Y = C.labels_to_matrix(tr.labels)
    Xtr, Xte = C.build_features(tr.sentence, te.sentence, verbose=True)
    print("X:", Xtr.shape, Xte.shape)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros((len(tr), C.N_LABELS))
    test_p = np.zeros((len(te), C.N_LABELS))

    for j in range(C.N_LABELS):
        for trn, val in kf.split(Xtr):
            m = LogisticRegression(C=4.0, max_iter=2000, solver="liblinear")
            m.fit(Xtr[trn], Y[trn, j])
            oof[val, j] = m.predict_proba(Xtr[val])[:, 1]
        m = LogisticRegression(C=4.0, max_iter=2000, solver="liblinear")
        m.fit(Xtr, Y[:, j])
        test_p[:, j] = m.predict_proba(Xte)[:, 1]
        print(f"label {j} done")

    print("OOF macro F1 @0.5:", C.macro_f1(Y, C.apply_thresholds(oof, np.full(10, .5))))
    thr, best = C.tune_thresholds(Y, oof)
    print("OOF macro F1 tuned:", best, "thr:", np.round(thr, 2))

    Yte = C.apply_thresholds(test_p, thr)
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame({"id": te.id, "labels": C.matrix_to_labels(Yte)}).to_csv(
        f"{OUT}/submission.csv", index=False)
    np.save("solution/baseline_oof.npy", oof)
    np.save("solution/baseline_test.npy", test_p)
    print("wrote", f"{OUT}/submission.csv")


if __name__ == "__main__":
    main()

"""Final model: train on train.csv, predict test.csv -> outputs/submission.csv.

Ensemble (weights validated by 3 x 5-fold CV on train, see cv.py / exp.py):
  1.0 * z(point-wise HistGradientBoosting + LogReg on option-level features
          with cross-fitted TF-IDF stack scores)
+ 0.4 * z(log-prob of a list-wise RandomForest(min_samples_leaf=10, sqrt))
+ 0.3 * z(log-prob of a list-wise ExtraTrees)
+ 0.4 * z(log-prob of a list-wise RandomForest(min_samples_leaf=5, mf=0.3))
List-wise models see the features of all four options of a question at once and
predict the answer index directly.

CV accuracy: ~0.342 vs. 0.311 for the majority-position baseline (chance 0.25).
Usage:  python solution/train_predict.py
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import build_features
from pipeline import fit_predict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LISTWISE = [
    ("rf10", 0.4, lambda sd: RandomForestClassifier(
        n_estimators=800, min_samples_leaf=10, max_features="sqrt",
        n_jobs=-1, random_state=sd)),
    ("et8", 0.3, lambda sd: ExtraTreesClassifier(
        n_estimators=800, min_samples_leaf=8, max_features="sqrt",
        n_jobs=-1, random_state=sd)),
    ("rf5", 0.4, lambda sd: RandomForestClassifier(
        n_estimators=800, min_samples_leaf=5, max_features=0.3,
        n_jobs=-1, random_state=sd)),
]


def z(M):
    return (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-9)


def listwise_scores(tr, te, seeds=(0, 1, 2)):
    """Return {name: z-scored log-probability matrix} for each list-wise model."""
    Ftr, vec = build_features(tr)
    Fte, _ = build_features(te, vec=vec)
    cols = [c for c in Ftr.columns if c not in ("qidx", "opt")]
    Xq = Ftr[cols].values.reshape(len(tr), -1)
    Xt = Fte[cols].values.reshape(len(te), -1)
    lab = tr["label"].values - 1
    out = {}
    for name, _w, mk in LISTWISE:
        P = np.zeros((len(te), 4))
        for sd in seeds:
            m = mk(sd)
            m.fit(Xq, lab)
            P += m.predict_proba(Xt)
        out[name] = z(np.log(P / len(seeds) + 1e-6))
    return out


def main():
    tr = pd.read_csv(os.path.join(ROOT, "train.csv"))
    te = pd.read_csv(os.path.join(ROOT, "test.csv"))
    print(f"train={tr.shape} test={te.shape}")

    print("[1/3] point-wise pipeline ...")
    S_pw = fit_predict(tr, te)

    print("[2/3] list-wise tree models ...")
    parts = listwise_scores(tr, te)

    print("[3/3] blending & writing submission ...")
    S = z(S_pw)
    for name, w, _mk in LISTWISE:
        S = S + w * parts[name]
    pred = S.argmax(1) + 1

    out_dir = os.path.join(ROOT, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    sub = pd.DataFrame({"id": te["id"].values, "label": pred.astype(int)})

    ss = pd.read_csv(os.path.join(ROOT, "sample_submission.csv"))
    assert list(sub.columns) == list(ss.columns)
    assert len(sub) == len(te) and sub["id"].is_unique
    assert set(sub["id"]) == set(te["id"])
    assert sub["label"].isin([1, 2, 3, 4]).all()
    sub.to_csv(os.path.join(out_dir, "submission.csv"), index=False)
    print("label distribution:\n", sub["label"].value_counts(normalize=True).sort_index())
    print("wrote", os.path.join(out_dir, "submission.csv"))


if __name__ == "__main__":
    main()

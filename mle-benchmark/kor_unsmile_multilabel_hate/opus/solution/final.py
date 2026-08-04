"""Final pipeline: base-model zoo -> per-label logistic stacking -> threshold
tuning -> outputs/submission.csv

Usage:  PYTHONPATH=solution python solution/final.py
"""
import os
import sys
import json
import numpy as np
import pandas as pd

import common as C
import zoo
import blend

OUT = os.environ.get("OUT_DIR", "outputs")

# Base models used in the final ensemble (selected on out-of-fold macro F1).
DEFAULT_MODELS = [
    # jamo-decomposed char n-grams (strongest single feature space)
    "lr_jamo", "lr_jamo_c2", "lr_jamo_c16", "svc_jamo", "svc_jamo_bal", "sgd_jamo",
    "lr_js", "svc_js_bal",
    # raw-character word-boundary n-grams
    "lr_cwb", "svc_cwb_bal",
    # combined char_wb + word + jamo space
    "lr_main_c4", "lr_main_c12", "lr_main_bal", "svc_main", "svc_main_bal",
    "ridge_main",
    # diversity: counts + Naive Bayes, SVD + MLP, jamo+word space
    "cnb_counts", "mlp_svd", "lr_jw", "lr_jw_bal", "svc_jw",
]
STACK_C = 200.0
USE_ALL_LABELS = False


def main(models=None, stack_c=STACK_C, use_all=USE_ALL_LABELS):
    tr, te = C.load(os.environ.get("DATA_DIR", "."))
    Y = C.labels_to_matrix(tr.labels)

    models = models or DEFAULT_MODELS
    zoo.run([m for m in models if m in zoo.MODELS], tr, te, Y)  # build if absent
    names, oofs, tsts = blend.load_cached(models)
    print("ensembling", len(names), "models:", names)

    oof2, tst2 = blend.stack(Y, oofs, tsts, Cval=stack_c, use_all_labels=use_all)
    thr, oof_score = C.tune_thresholds(Y, oof2, rounds=3)
    hs, hstd = blend.holdout_eval(Y, oof2, n_rep=4)
    print(f"OOF macro F1 (tuned on OOF) = {oof_score:.4f}")
    print(f"honest holdout estimate     = {hs:.4f} +/- {hstd:.4f}")
    print("thresholds:", np.round(thr, 3).tolist())

    Yte = C.apply_thresholds(tst2, thr)
    os.makedirs(OUT, exist_ok=True)
    sub = pd.DataFrame({"id": te.id, "labels": C.matrix_to_labels(Yte)})
    sub.to_csv(f"{OUT}/submission.csv", index=False)

    # sanity checks
    assert len(sub) == len(te) and set(sub.id) == set(te.id)
    assert sub.labels.str.fullmatch(r"[01]{10}").all()
    print("\npredicted positives per label:", Yte.sum(0).tolist())
    print("train label rate  :", np.round(Y.mean(0), 4).tolist())
    print("test  label rate  :", np.round(Yte.mean(0), 4).tolist())
    print("wrote", f"{OUT}/submission.csv")

    with open("solution/final_report.json", "w") as f:
        json.dump({"models": names, "stack_C": stack_c,
                   "use_all_labels": use_all,
                   "oof_macro_f1": oof_score, "holdout_macro_f1": hs,
                   "thresholds": thr.tolist()}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main(sys.argv[1:] or None)

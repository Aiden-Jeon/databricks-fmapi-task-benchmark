"""Stacking: base OOF probs + group/aspect target-encoding + numeric feats."""
import os
import sys
import glob
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, numeric_feats, CLASSES, TASK  # noqa

CACHE = os.path.join(TASK, "solution", "cache")
SEED = 42
NFOLD = 5

BASE = ["lr_text2", "lr2_text", "lr_jamo", "lr_win", "svc_win", "svc_jamo",
        "svc_text2", "lgbm_text", "cnb_text", "lr_masked", "ridge_text"]


def group_aspect_feats(fit_df, fit_y, apply_df, prior, alpha=5.0):
    """Target-encode aspect and sentence-group (other aspects' labels)."""
    n = len(apply_df)
    F = np.zeros((n, 9), dtype=np.float32)
    # --- aspect encoding
    d = pd.DataFrame({"a": fit_df.aspect.values, "y": fit_y})
    agg = d.groupby("a").y.apply(lambda s: np.bincount(s, minlength=3)).to_dict()
    # --- sentence group encoding
    d2 = pd.DataFrame({"s": fit_df.sentence.values, "y": fit_y,
                       "asp": fit_df.aspect.values})
    smap = {}
    for s, sub in d2.groupby("s"):
        smap[s] = (np.bincount(sub.y.values, minlength=3), list(sub.asp.values))
    for i, (s, a) in enumerate(zip(apply_df.sentence.values, apply_df.aspect.values)):
        c = agg.get(a)
        if c is None:
            F[i, 0:3] = prior
            F[i, 3] = 0
        else:
            F[i, 0:3] = (c + alpha * prior) / (c.sum() + alpha)
            F[i, 3] = c.sum()
        g = smap.get(s)
        if g is None:
            F[i, 4:7] = prior
            F[i, 7] = 0
        else:
            cnt = g[0].astype(np.float64).copy()
            # remove self if this exact (sentence,aspect) is in the fit set
            tot = cnt.sum()
            F[i, 4:7] = (cnt + alpha * prior) / (tot + alpha)
            F[i, 7] = tot
        F[i, 8] = 1.0 if g is not None else 0.0
    return F


def main():
    tr, te = load()
    le = LabelEncoder().fit(CLASSES)
    y = le.transform(tr.label.values)
    prior = np.bincount(y, minlength=3) / len(y)
    folds = list(StratifiedKFold(NFOLD, shuffle=True, random_state=SEED).split(tr, y))

    names = [b for b in BASE if os.path.exists(f"{CACHE}/{b}_oof.npy")]
    Poof = np.hstack([np.load(f"{CACHE}/{n}_oof.npy") for n in names])
    Pte = np.hstack([np.load(f"{CACHE}/{n}_test.npy") for n in names])
    num_tr, num_te = numeric_feats(tr), numeric_feats(te)

    # group/aspect features: fold-aware for train, full-train for test
    G_oof = np.zeros((len(tr), 9), dtype=np.float32)
    for i_tr, i_va in folds:
        G_oof[i_va] = group_aspect_feats(tr.iloc[i_tr], y[i_tr], tr.iloc[i_va], prior)
    G_te = group_aspect_feats(tr, y, te, prior)

    variants = {
        "P": (Poof, Pte),
        "P+num": (np.c_[Poof, num_tr], np.c_[Pte, num_te]),
        "P+G": (np.c_[Poof, G_oof], np.c_[Pte, G_te]),
        "P+G+num": (np.c_[Poof, G_oof, num_tr], np.c_[Pte, G_te, num_te]),
    }
    import lightgbm as lgb
    results = {}
    for vn, (A, B) in variants.items():
        for mn in ["lr", "lgb"]:
            oof = np.zeros((len(tr), 3))
            tep = np.zeros((len(te), 3))
            for i_tr, i_va in folds:
                if mn == "lr":
                    from sklearn.preprocessing import StandardScaler
                    sc = StandardScaler().fit(A[i_tr])
                    m = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(A[i_tr]), y[i_tr])
                    oof[i_va] = m.predict_proba(sc.transform(A[i_va]))
                    tep += m.predict_proba(sc.transform(B)) / NFOLD
                else:
                    m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03,
                                           num_leaves=15, subsample=0.8, subsample_freq=1,
                                           colsample_bytree=0.7, min_child_samples=30,
                                           reg_lambda=1.0, verbose=-1, random_state=SEED)
                    m.fit(A[i_tr], y[i_tr], eval_set=[(A[i_va], y[i_va])],
                          callbacks=[lgb.early_stopping(50, verbose=False)])
                    oof[i_va] = m.predict_proba(A[i_va])
                    tep += m.predict_proba(B) / NFOLD
            s = f1_score(y, oof.argmax(1), average="macro")
            print(f"stack {vn:9s} {mn:4s} f1={s:.4f}")
            results[f"STK_{vn}_{mn}"] = (s, oof, tep)
    best = max(results.items(), key=lambda kv: kv[1][0])
    print("best stack:", best[0], round(best[1][0], 4))
    for k, (s, oof, tep) in results.items():
        k2 = k.replace("+", "")
        np.save(f"{CACHE}/{k2}_oof.npy", oof)
        np.save(f"{CACHE}/{k2}_test.npy", tep)


if __name__ == "__main__":
    main()

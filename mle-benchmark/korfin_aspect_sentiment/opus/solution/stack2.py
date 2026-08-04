"""Enriched stacking: base OOF probs + rich group/aspect target-encoding features.

Group features are computed fold-aware (fit part only) to avoid self-leakage, which
mirrors the test-time situation (test rows never appear in train).
"""
import os
import sys
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, numeric_feats, CLASSES, TASK  # noqa

CACHE = os.path.join(TASK, "solution", "cache")
SEED = 42
NFOLD = 5
BASE = ["lr_text2", "lr2_text", "lr_jamo", "lr_win", "svc_win", "svc_jamo",
        "svc_text2", "lgbm_text", "cnb_text", "lr_masked", "ridge_text",
        "lr_text", "svc_text", "svc2_text", "sgd_text", "mnb_text", "lr20_text",
        "lr_dir", "svc_dir", "lr_clause", "svc_clause", "lr_strip", "svc_strip",
        "svc_right", "lr_multi", "svc_multi", "lr_multi2", "svc_multi2",
        "lr_multi3", "svc_multi3", "svcA_multi", "lrA_multi", "ridge_multi"]

FEAT_NAMES = None


def _norm_aspect(a):
    return a.strip().lower().replace(" ", "")


def rich_feats(fit_df, fit_soft, apply_df, prior, alpha=4.0):
    """fit_soft: (n_fit, 3) soft label matrix (one-hot for true labels)."""
    global FEAT_NAMES
    cols = []
    names = []

    def push(arr, nm):
        cols.append(np.asarray(arr, dtype=np.float32).reshape(len(apply_df), -1))
        k = cols[-1].shape[1]
        names.extend([f"{nm}{i}" for i in range(k)] if k > 1 else [nm])

    fs = fit_df.sentence.values
    fa = fit_df.aspect.fillna("").values
    asent = apply_df.sentence.values
    aasp = apply_df.aspect.fillna("").values
    n = len(apply_df)

    # ---- aspect target encoding (exact + normalized) ----
    for key_fn, tag in [(lambda x: x, "asp"), (_norm_aspect, "aspn")]:
        acc = defaultdict(lambda: np.zeros(3))
        for k, w in zip(fa, fit_soft):
            acc[key_fn(k)] += w
        te_ = np.zeros((n, 3), np.float32)
        cn = np.zeros(n, np.float32)
        for i, k in enumerate(aasp):
            c = acc.get(key_fn(k))
            if c is None:
                te_[i] = prior
            else:
                te_[i] = (c + alpha * prior) / (c.sum() + alpha)
                cn[i] = c.sum()
        push(te_, f"te_{tag}_")
        push(np.log1p(cn), f"te_{tag}_n")

    # ---- sentence-group (siblings) ----
    gmap = defaultdict(list)  # sentence -> list of (aspect, soft)
    for s, a, w in zip(fs, fa, fit_soft):
        gmap[s].append((a, w))
    sib = np.zeros((n, 3), np.float32)
    sibn = np.zeros(n, np.float32)
    near = np.zeros((n, 3), np.float32)
    neard = np.full(n, -1.0, np.float32)
    for i, (s, a) in enumerate(zip(asent, aasp)):
        g = gmap.get(s)
        if not g:
            sib[i] = prior
            near[i] = prior
            continue
        tot = np.zeros(3)
        best_d, best_w = None, None
        p_self = s.index(a) if a and a in s else -1
        for a2, w in g:
            if a2 == a:
                continue
            tot += w
            if p_self >= 0 and a2 and a2 in s:
                d = abs(s.index(a2) - p_self)
                if best_d is None or d < best_d:
                    best_d, best_w = d, w
        sibn[i] = tot.sum()
        sib[i] = (tot + alpha * prior) / (tot.sum() + alpha)
        if best_w is not None:
            near[i] = best_w
            neard[i] = best_d / max(len(s), 1)
        else:
            near[i] = prior
    push(sib, "sib_")
    push(sibn, "sib_n")
    push(near, "near_")
    push(neard, "near_d")

    # ---- fuzzy aspect match: aspect is substring of / contains a fit aspect ----
    uniq = defaultdict(lambda: np.zeros(3))
    for k, w in zip(fa, fit_soft):
        uniq[k] += w
    keys = [k for k in uniq if len(k) >= 2]
    fz = np.zeros((n, 3), np.float32)
    fzn = np.zeros(n, np.float32)
    for i, a in enumerate(aasp):
        if len(a) < 2:
            fz[i] = prior
            continue
        tot = np.zeros(3)
        for k in keys:
            if k == a:
                continue
            if a in k or k in a:
                tot += uniq[k]
        fzn[i] = tot.sum()
        fz[i] = (tot + alpha * prior) / (tot.sum() + alpha)
    push(fz, "fz_")
    push(np.log1p(fzn), "fz_n")
    F = np.hstack(cols)
    if FEAT_NAMES is None:
        FEAT_NAMES = names
    return F


def run(A, B, y, folds, kind, seeds=(0,), lgb_params=None):
    import lightgbm as lgb
    ntr, nte = len(A), len(B)
    oof = np.zeros((ntr, 3))
    tep = np.zeros((nte, 3))
    for sd in seeds:
        for i_tr, i_va in folds:
            if kind == "lr":
                sc = StandardScaler().fit(A[i_tr])
                m = LogisticRegression(C=1.0, max_iter=4000).fit(sc.transform(A[i_tr]), y[i_tr])
                oof[i_va] += m.predict_proba(sc.transform(A[i_va])) / len(seeds)
                tep += m.predict_proba(sc.transform(B)) / (len(folds) * len(seeds))
            else:
                p = dict(n_estimators=1500, learning_rate=0.03, num_leaves=15,
                         subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                         min_child_samples=30, reg_lambda=1.0, verbose=-1,
                         random_state=SEED + sd)
                if lgb_params:
                    p.update(lgb_params)
                m = lgb.LGBMClassifier(**p)
                m.fit(A[i_tr], y[i_tr], eval_set=[(A[i_va], y[i_va])],
                      callbacks=[lgb.early_stopping(80, verbose=False)])
                oof[i_va] += m.predict_proba(A[i_va]) / len(seeds)
                tep += m.predict_proba(B) / (len(folds) * len(seeds))
    return oof, tep


def build_matrices():
    tr, te = load()
    le = LabelEncoder().fit(CLASSES)
    y = le.transform(tr.label.values)
    prior = np.bincount(y, minlength=3) / len(y)
    folds = list(StratifiedKFold(NFOLD, shuffle=True, random_state=SEED).split(tr, y))
    names = [b for b in BASE if os.path.exists(f"{CACHE}/{b}_oof.npy")]
    Poof = np.hstack([np.load(f"{CACHE}/{n}_oof.npy") for n in names])
    Pte = np.hstack([np.load(f"{CACHE}/{n}_test.npy") for n in names])
    onehot = np.eye(3)[y]
    G_oof = None
    for i_tr, i_va in folds:
        f = rich_feats(tr.iloc[i_tr], onehot[i_tr], tr.iloc[i_va], prior)
        if G_oof is None:
            G_oof = np.zeros((len(tr), f.shape[1]), np.float32)
        G_oof[i_va] = f
    G_te = rich_feats(tr, onehot, te, prior)
    num_tr, num_te = numeric_feats(tr), numeric_feats(te)
    # transductive group size (no labels used)
    allc = pd.concat([tr.sentence, te.sentence]).value_counts()
    gtr = tr.sentence.map(allc).values.reshape(-1, 1).astype(np.float32)
    gte = te.sentence.map(allc).values.reshape(-1, 1).astype(np.float32)
    A = np.c_[Poof, G_oof, num_tr, gtr]
    B = np.c_[Pte, G_te, num_te, gte]
    return tr, te, y, folds, A, B, names, prior, onehot


def main():
    tr, te, y, folds, A, B, names, prior, onehot = build_matrices()
    print("stack matrix", A.shape, "bases:", len(names))
    res = {}
    for kind in ["lr", "lgb"]:
        oof, tep = run(A, B, y, folds, kind, seeds=(0, 1, 2) if kind == "lgb" else (0,))
        s = f1_score(y, oof.argmax(1), average="macro")
        print(f"rich {kind}: {s:.4f}")
        res[f"S2{kind}"] = (s, oof, tep)
    # blend of lr and lgb
    for w in [0.2, 0.3, 0.4, 0.5]:
        o = w * res["S2lr"][1] + (1 - w) * res["S2lgb"][1]
        print(f"  mix w_lr={w}: {f1_score(y, o.argmax(1), average='macro'):.4f}")
    for k, (s, oof, tep) in res.items():
        np.save(f"{CACHE}/{k}_oof.npy", oof)
        np.save(f"{CACHE}/{k}_test.npy", tep)


if __name__ == "__main__":
    main()

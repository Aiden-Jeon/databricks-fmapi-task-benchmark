"""Final pipeline for PAWS-X Korean paraphrase identification.

Everything is learned from train.csv (no internet, no external data, no
pretrained weights).  Components:

  A. dense hand-crafted features            (features.py, 213 dims)
  B. fuzzy token-alignment features         (features2.py, 46 dims)
  C. train-label graph features             (features2.py, 15 dims)
       nodes = sentences, edges = labelled train pairs.  For a train row the
       row's own edge is masked out, which makes the feature distribution
       identical to test rows (never present in the graph).
  D. sparse TF-IDF views -> LogReg, stacked as OOF probabilities (5 dims)

  Model: HistGradientBoosting ensemble over [A|B|C|D], trained with s1<->s2
  symmetry augmentation and averaged over both orientations at inference.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import features as F  # noqa: E402
import features2 as F2  # noqa: E402

SEED = 42
NFOLD = 5
CACHE = os.path.join(HERE, "cache")
os.makedirs(CACHE, exist_ok=True)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def cached(name, fn):
    p = os.path.join(CACHE, name + ".npy")
    if os.path.exists(p):
        return np.load(p, allow_pickle=True)
    v = fn()
    np.save(p, v)
    return v


def swapped(df):
    d = df.copy()
    d["sentence1"], d["sentence2"] = df["sentence2"].values, df["sentence1"].values
    return d


# ------------------------------------------------------------------ text views
def diff_text(s1, s2):
    a = F.stem_tokens(F.tokenize(F.norm(s1)))
    b = F.stem_tokens(F.tokenize(F.norm(s2)))
    sa, sb = set(a), set(b)
    only = sorted((sa - sb) | (sb - sa))
    return " ".join(only) if only else "__none__"


def inter_text(s1, s2):
    a = F.tokenize(F.norm(s1))
    b = F.tokenize(F.norm(s2))
    it = sorted(set(a) & set(b))
    return " ".join(it) if it else "__none__"


def main():
    t0 = time.time()
    tr = pd.read_csv(os.path.join(ROOT, "train.csv"))
    te = pd.read_csv(os.path.join(ROOT, "test.csv"))
    sub_tmpl = pd.read_csv(os.path.join(ROOT, "sample_submission.csv"))
    for c in ("sentence1", "sentence2"):
        tr[c] = tr[c].fillna("")
        te[c] = te[c].fillna("")
    y = tr["label"].values.astype(int)
    log("data", tr.shape, te.shape)

    # ---------------- A. dense features -------------------------------
    dcols = list(np.load(os.path.join(CACHE, "dense_tr_a_cols.npy"), allow_pickle=True)) \
        if os.path.exists(os.path.join(CACHE, "dense_tr_a_cols.npy")) else None

    def mk_dense(df, tag):
        p = os.path.join(CACHE, f"dense_{tag}.npy")
        if os.path.exists(p):
            return np.load(p)
        X = F.build_matrix(df)
        np.save(p, X.values.astype(np.float32))
        np.save(os.path.join(CACHE, f"dense_{tag}_cols.npy"),
                np.array(X.columns, dtype=object))
        return X.values.astype(np.float32)

    log("A. dense features")
    D_tr_a = mk_dense(tr, "tr_a")
    D_tr_b = mk_dense(swapped(tr), "tr_b")
    D_te_a = mk_dense(te, "te_a")
    D_te_b = mk_dense(swapped(te), "te_b")

    # ---------------- B. alignment features ---------------------------
    log("B. alignment features")
    L_tr_a = cached("align_tr_a", lambda: F2.build_align_matrix(tr))
    L_tr_b = cached("align_tr_b", lambda: F2.build_align_matrix(swapped(tr)))
    L_te_a = cached("align_te_a", lambda: F2.build_align_matrix(te))
    L_te_b = cached("align_te_b", lambda: F2.build_align_matrix(swapped(te)))

    # ---------------- C. graph features -------------------------------
    log("C. graph features")
    adj = F2.build_graph(zip(tr["sentence1"], tr["sentence2"], y))

    def graph_mat(s1_list, s2_list, mask_own):
        rows = []
        for a, b in zip(s1_list, s2_list):
            if mask_own and a in adj and b in adj[a]:
                lab = adj[a].pop(b)
                adj[b].pop(a, None)
                rows.append(F2.graph_row(adj, a, b))
                adj[a][b] = lab
                adj[b][a] = lab
            else:
                rows.append(F2.graph_row(adj, a, b))
        return np.asarray(rows, dtype=np.float32)

    G_tr = graph_mat(tr["sentence1"].tolist(), tr["sentence2"].tolist(), True)
    G_te = graph_mat(te["sentence1"].tolist(), te["sentence2"].tolist(), False)
    # graph features are symmetric except for the a/b naming -> swapped version
    gi = {k: i for i, k in enumerate(F2.GRAPH_KEYS)}
    swap_perm = list(range(len(F2.GRAPH_KEYS)))
    for x, z in [("g_deg_a", "g_deg_b"), ("g_pos_deg_a", "g_pos_deg_b"),
                 ("g_neg_deg_a", "g_neg_deg_b")]:
        swap_perm[gi[x]], swap_perm[gi[z]] = swap_perm[gi[z]], swap_perm[gi[x]]
    G_tr_b = G_tr[:, swap_perm]
    G_te_b = G_te[:, swap_perm]
    log("   graph evidence rate train/test: %.3f / %.3f"
        % ((G_tr[:, gi["g_has_evidence"]] > 0).mean(),
           (G_te[:, gi["g_has_evidence"]] > 0).mean()))

    skf = StratifiedKFold(n_splits=NFOLD, shuffle=True, random_state=SEED)
    folds = list(skf.split(np.zeros(len(y)), y))

    # ---------------- D. sparse stacking ------------------------------
    log("D. sparse stacking")
    p1_tr = [F.norm(s) for s in tr["sentence1"]]
    p2_tr = [F.norm(s) for s in tr["sentence2"]]
    p1_te = [F.norm(s) for s in te["sentence1"]]
    p2_te = [F.norm(s) for s in te["sentence2"]]
    dt_tr = [diff_text(a, b) for a, b in zip(tr["sentence1"], tr["sentence2"])]
    dt_te = [diff_text(a, b) for a, b in zip(te["sentence1"], te["sentence2"])]
    it_tr = [inter_text(a, b) for a, b in zip(tr["sentence1"], tr["sentence2"])]
    it_te = [inter_text(a, b) for a, b in zip(te["sentence1"], te["sentence2"])]

    def pair_mat(v, A, B):
        Va, Vb = v.transform(A), v.transform(B)
        return sparse.hstack([abs(Va - Vb), Va.multiply(Vb)], format="csr")

    VIEWS = {
        "diff_word": ("single", dt_tr, dt_te,
                      dict(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        "diff_char": ("single", dt_tr, dt_te,
                      dict(analyzer="char_wb", ngram_range=(2, 4), min_df=3,
                           sublinear_tf=True, max_features=300000)),
        "inter_word": ("single", it_tr, it_te,
                       dict(analyzer="word", ngram_range=(1, 1), min_df=2, sublinear_tf=True)),
        "pair_char": ("pair", None, None,
                      dict(analyzer="char_wb", ngram_range=(2, 4), min_df=3,
                           sublinear_tf=True, max_features=300000)),
        "pair_word": ("pair", None, None,
                      dict(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
    }
    stack_names, S_tr_cols, S_te_cols = [], [], []
    for name, (kind, Atr, Ate, kw) in VIEWS.items():
        cp = os.path.join(CACHE, f"stack_{name}.npz")
        if os.path.exists(cp):
            z = np.load(cp)
            oof, pte = z["oof"], z["pte"]
        else:
            oof = np.zeros(len(y))
            pte = np.zeros(len(te))
            for itr, iva in folds:
                v = TfidfVectorizer(**kw)
                if kind == "single":
                    Mtr = v.fit_transform([Atr[i] for i in itr])
                    Mva = v.transform([Atr[i] for i in iva])
                    Mte = v.transform(Ate)
                else:
                    v.fit([p1_tr[i] for i in itr] + [p2_tr[i] for i in itr])
                    Mtr = pair_mat(v, [p1_tr[i] for i in itr], [p2_tr[i] for i in itr])
                    Mva = pair_mat(v, [p1_tr[i] for i in iva], [p2_tr[i] for i in iva])
                    Mte = pair_mat(v, p1_te, p2_te)
                clf = LogisticRegression(C=1.0, max_iter=2000, solver="liblinear")
                clf.fit(Mtr, y[itr])
                oof[iva] = clf.predict_proba(Mva)[:, 1]
                pte += clf.predict_proba(Mte)[:, 1] / NFOLD
            np.savez(cp, oof=oof, pte=pte)
        log(f"   sparse[{name}] oof acc={((oof>0.5).astype(int)==y).mean():.4f}")
        stack_names.append("stk_" + name)
        S_tr_cols.append(oof)
        S_te_cols.append(pte)
    S_tr = np.column_stack(S_tr_cols).astype(np.float32)
    S_te = np.column_stack(S_te_cols).astype(np.float32)

    # ---------------- assemble ---------------------------------------
    Xtr_a = np.hstack([D_tr_a, L_tr_a, G_tr, S_tr]).astype(np.float32)
    Xtr_b = np.hstack([D_tr_b, L_tr_b, G_tr_b, S_tr]).astype(np.float32)
    Xte_a = np.hstack([D_te_a, L_te_a, G_te, S_te]).astype(np.float32)
    Xte_b = np.hstack([D_te_b, L_te_b, G_te_b, S_te]).astype(np.float32)
    log("design matrix", Xtr_a.shape)

    CONFIGS = [
        ("gbm1", dict(max_iter=1500, learning_rate=0.04, max_leaf_nodes=31,
                      min_samples_leaf=30, l2_regularization=1.0,
                      max_features=0.7)),
        ("gbm2", dict(max_iter=1500, learning_rate=0.06, max_leaf_nodes=63,
                      min_samples_leaf=60, l2_regularization=3.0,
                      max_features=0.5)),
        ("gbm3", dict(max_iter=2000, learning_rate=0.03, max_leaf_nodes=15,
                      min_samples_leaf=20, l2_regularization=0.5,
                      max_features=0.9)),
    ]
    oofs, ptes, names = [], [], []
    for cname, kw in CONFIGS:
        cp = os.path.join(CACHE, f"model_{cname}.npz")
        if os.path.exists(cp):
            z = np.load(cp)
            oof, pte = z["oof"], z["pte"]
        else:
            oof = np.zeros(len(y))
            pte = np.zeros(len(te))
            for k, (itr, iva) in enumerate(folds):
                Xf = np.vstack([Xtr_a[itr], Xtr_b[itr]])
                yf = np.concatenate([y[itr], y[itr]])
                m = HistGradientBoostingClassifier(
                    early_stopping=True, validation_fraction=0.12,
                    n_iter_no_change=60, random_state=SEED + 7 * k, **kw)
                m.fit(Xf, yf)
                oof[iva] = 0.5 * (m.predict_proba(Xtr_a[iva])[:, 1] +
                                  m.predict_proba(Xtr_b[iva])[:, 1])
                pte += 0.5 * (m.predict_proba(Xte_a)[:, 1] +
                              m.predict_proba(Xte_b)[:, 1]) / NFOLD
                log(f"   {cname} fold{k} iters={m.n_iter_} "
                    f"acc={(((oof[iva]>0.5).astype(int))==y[iva]).mean():.4f}")
            np.savez(cp, oof=oof, pte=pte)
        log(f"  {cname} oof acc={((oof>0.5).astype(int)==y).mean():.4f}")
        oofs.append(oof)
        ptes.append(pte)
        names.append(cname)

    # extra-trees for diversity
    cp = os.path.join(CACHE, "model_et.npz")
    if os.path.exists(cp):
        z = np.load(cp)
        oof, pte = z["oof"], z["pte"]
    else:
        oof = np.zeros(len(y))
        pte = np.zeros(len(te))
        Xa = np.nan_to_num(Xtr_a, nan=-999.0)
        Xb = np.nan_to_num(Xtr_b, nan=-999.0)
        Ta = np.nan_to_num(Xte_a, nan=-999.0)
        Tb = np.nan_to_num(Xte_b, nan=-999.0)
        for k, (itr, iva) in enumerate(folds):
            m = ExtraTreesClassifier(n_estimators=600, min_samples_leaf=2,
                                     max_features="sqrt", n_jobs=-1,
                                     random_state=SEED + k)
            m.fit(np.vstack([Xa[itr], Xb[itr]]), np.concatenate([y[itr], y[itr]]))
            oof[iva] = 0.5 * (m.predict_proba(Xa[iva])[:, 1] + m.predict_proba(Xb[iva])[:, 1])
            pte += 0.5 * (m.predict_proba(Ta)[:, 1] + m.predict_proba(Tb)[:, 1]) / NFOLD
        np.savez(cp, oof=oof, pte=pte)
    log(f"  et oof acc={((oof>0.5).astype(int)==y).mean():.4f}")
    oofs.append(oof)
    ptes.append(pte)
    names.append("et")

    # ---------------- blend ------------------------------------------
    O = np.column_stack(oofs)
    P = np.column_stack(ptes)
    # rank-average + logistic meta learner, pick whichever is better on OOF
    best = (0.0, None, None)
    simple = O.mean(1)
    accs = {"mean": ((simple > 0.5).astype(int) == y).mean()}
    meta_oof = np.zeros(len(y))
    meta_pte = np.zeros(len(te))
    Ol = np.log(np.clip(O, 1e-6, 1 - 1e-6) / (1 - np.clip(O, 1e-6, 1 - 1e-6)))
    Pl = np.log(np.clip(P, 1e-6, 1 - 1e-6) / (1 - np.clip(P, 1e-6, 1 - 1e-6)))
    for itr, iva in folds:
        mm = LogisticRegression(C=1.0, max_iter=1000)
        mm.fit(Ol[itr], y[itr])
        meta_oof[iva] = mm.predict_proba(Ol[iva])[:, 1]
        meta_pte += mm.predict_proba(Pl)[:, 1] / NFOLD
    accs["meta"] = ((meta_oof > 0.5).astype(int) == y).mean()
    for k, v in accs.items():
        log(f"  blend[{k}] oof acc={v:.4f}")
    if accs["meta"] >= accs["mean"]:
        oof_final, pte_final = meta_oof, meta_pte
    else:
        oof_final, pte_final = simple, P.mean(1)

    # threshold tuning on OOF
    ths = np.linspace(0.3, 0.7, 81)
    scores = [((oof_final > t).astype(int) == y).mean() for t in ths]
    bi = int(np.argmax(scores))
    th = float(ths[bi])
    log(f"  best threshold {th:.3f} oof acc={scores[bi]:.4f} (0.5 -> {((oof_final>0.5).astype(int)==y).mean():.4f})")
    if scores[bi] - ((oof_final > 0.5).astype(int) == y).mean() < 0.0015:
        th = 0.5
        log("  -> keeping threshold 0.5 (gain not meaningful)")

    pred = (pte_final > th).astype(int)
    out = pd.DataFrame({"id": te["id"].values, "label": pred})
    out = sub_tmpl[["id"]].merge(out, on="id", how="left")
    assert len(out) == len(sub_tmpl) and out["label"].notna().all()
    assert out["id"].is_unique
    out["label"] = out["label"].astype(int)
    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    out.to_csv(os.path.join(ROOT, "outputs", "submission.csv"), index=False)
    np.save(os.path.join(CACHE, "oof_final.npy"), oof_final)
    np.save(os.path.join(CACHE, "pte_final.npy"), pte_final)
    log("wrote outputs/submission.csv", out.shape, "pos_rate=%.3f" % pred.mean(),
        f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

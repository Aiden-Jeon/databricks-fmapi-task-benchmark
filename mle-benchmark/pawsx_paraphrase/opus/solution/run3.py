"""Final model for PAWS-X Korean paraphrase identification (train.csv only).

Feature blocks
  A dense similarity / word-order features        features.py   (213)
  B fuzzy one-to-one token-alignment features     features2.py   (46)
  C sentence-graph features from train labels     features2.py   (15)
      nodes = sentences, edges = labelled train pairs.  A train row's own edge
      is masked while computing its features, so the amount of label evidence
      matches test rows exactly (verified: 24.6% vs 25.3% coverage).
  D symbolic edit-script view (edits.py) -> TF-IDF, reduced with SVD  (+64)
  E OOF stacked probabilities of 7 sparse linear models              (+7)

Model: HistGradientBoosting ensemble (3 configs) trained with s1<->s2 symmetry
augmentation, predictions averaged over both orientations, blended on OOF.
"""
import gc
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import features as F  # noqa: E402
import features2 as F2  # noqa: E402
import edits as ED  # noqa: E402

SEED, NFOLD = 42, 5
CACHE = os.path.join(HERE, "cache")
os.makedirs(CACHE, exist_ok=True)
EDIT_TOK = r"\S+"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def swapped(df):
    d = df.copy()
    d["sentence1"], d["sentence2"] = df["sentence2"].values, df["sentence1"].values
    return d


def npy(name, fn):
    p = os.path.join(CACHE, name + ".npy")
    if os.path.exists(p):
        return np.load(p, allow_pickle=True)
    v = fn()
    np.save(p, v)
    return v


def diff_text(s1, s2):
    a = F.stem_tokens(F.tokenize(F.norm(s1)))
    b = F.stem_tokens(F.tokenize(F.norm(s2)))
    sa, sb = set(a), set(b)
    only = sorted((sa - sb) | (sb - sa))
    return " ".join(only) if only else "__none__"


def inter_text(s1, s2):
    a, b = F.tokenize(F.norm(s1)), F.tokenize(F.norm(s2))
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

    # ---------------- A dense ------------------------------------------
    def mk_dense(df, tag):
        p = os.path.join(CACHE, f"dense_{tag}.npy")
        if os.path.exists(p):
            return np.load(p)
        X = F.build_matrix(df)
        np.save(p, X.values.astype(np.float32))
        np.save(os.path.join(CACHE, f"dense_{tag}_cols.npy"),
                np.array(X.columns, dtype=object))
        return X.values.astype(np.float32)

    log("A dense")
    D_tr_a, D_tr_b = mk_dense(tr, "tr_a"), mk_dense(swapped(tr), "tr_b")
    D_te_a, D_te_b = mk_dense(te, "te_a"), mk_dense(swapped(te), "te_b")

    # ---------------- B alignment --------------------------------------
    log("B alignment")
    L_tr_a = npy("align_tr_a", lambda: F2.build_align_matrix(tr))
    L_tr_b = npy("align_tr_b", lambda: F2.build_align_matrix(swapped(tr)))
    L_te_a = npy("align_te_a", lambda: F2.build_align_matrix(te))
    L_te_b = npy("align_te_b", lambda: F2.build_align_matrix(swapped(te)))

    # ---------------- C graph ------------------------------------------
    log("C graph")
    adj = F2.build_graph(zip(tr["sentence1"], tr["sentence2"], y))

    def graph_mat(s1l, s2l, mask_own):
        rows = []
        for a, b in zip(s1l, s2l):
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
    gi = {k: i for i, k in enumerate(F2.GRAPH_KEYS)}
    perm = list(range(len(F2.GRAPH_KEYS)))
    for x, z in [("g_deg_a", "g_deg_b"), ("g_pos_deg_a", "g_pos_deg_b"),
                 ("g_neg_deg_a", "g_neg_deg_b")]:
        perm[gi[x]], perm[gi[z]] = perm[gi[z]], perm[gi[x]]
    G_tr_b, G_te_b = G_tr[:, perm], G_te[:, perm]
    log("  graph evidence train/test %.3f/%.3f"
        % ((G_tr[:, gi["g_has_evidence"]] > 0).mean(),
           (G_te[:, gi["g_has_evidence"]] > 0).mean()))

    folds = list(StratifiedKFold(NFOLD, shuffle=True, random_state=SEED).split(y, y))

    # ---------------- D edit-script view -------------------------------
    log("D edit script")
    ep = os.path.join(CACHE, "edits.pkl")
    if os.path.exists(ep):
        E = pickle.load(open(ep, "rb"))
    else:
        E = {"tr": ED.build(tr), "te": ED.build(te)}
        pickle.dump(E, open(ep, "wb"))
    # the edit script is direction-dependent (MOVE_p vs MOVE_n); build the
    # swapped version too so symmetry augmentation stays consistent
    ep2 = os.path.join(CACHE, "edits_swap.pkl")
    if os.path.exists(ep2):
        E2 = pickle.load(open(ep2, "rb"))
    else:
        E2 = {"tr": ED.build(swapped(tr)), "te": ED.build(swapped(te))}
        pickle.dump(E2, open(ep2, "wb"))

    def svd_block(name, txt_tr, txt_te, txt_tr_b, txt_te_b, ncomp=64):
        p = os.path.join(CACHE, f"svd_{name}.npz")
        if os.path.exists(p):
            z = np.load(p)
            return z["a"], z["b"], z["c"], z["d"]
        v = TfidfVectorizer(token_pattern=EDIT_TOK, analyzer="word",
                            ngram_range=(1, 2), min_df=3, sublinear_tf=True)
        v.fit(list(txt_tr) + list(txt_te))
        sv = TruncatedSVD(n_components=ncomp, random_state=SEED)
        sv.fit(v.transform(list(txt_tr) + list(txt_te)))
        a = sv.transform(v.transform(txt_tr)).astype(np.float32)
        c = sv.transform(v.transform(txt_te)).astype(np.float32)
        b = sv.transform(v.transform(txt_tr_b)).astype(np.float32)
        d = sv.transform(v.transform(txt_te_b)).astype(np.float32)
        np.savez(p, a=a, b=b, c=c, d=d)
        return a, b, c, d

    V_tr_a, V_tr_b, V_te_a, V_te_b = svd_block(
        "edit", E["tr"], E["te"], E2["tr"], E2["te"], ncomp=64)
    log("  edit SVD", V_tr_a.shape)

    # ---------------- E sparse stacking --------------------------------
    log("E sparse stacking")
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

    VIEWS = [
        ("diff_word", "single", dt_tr, dt_te,
         dict(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True), 1.0),
        ("diff_char", "single", dt_tr, dt_te,
         dict(analyzer="char_wb", ngram_range=(2, 4), min_df=3, sublinear_tf=True,
              max_features=300000), 1.0),
        ("inter_word", "single", it_tr, it_te,
         dict(analyzer="word", ngram_range=(1, 1), min_df=2, sublinear_tf=True), 1.0),
        ("pair_char", "pair", None, None,
         dict(analyzer="char_wb", ngram_range=(2, 4), min_df=3, sublinear_tf=True,
              max_features=300000), 1.0),
        ("pair_word", "pair", None, None,
         dict(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True), 1.0),
        ("edit12", "single", E["tr"], E["te"],
         dict(token_pattern=EDIT_TOK, analyzer="word", ngram_range=(1, 2), min_df=3,
              sublinear_tf=True), 0.5),
        ("edit1", "single", E["tr"], E["te"],
         dict(token_pattern=EDIT_TOK, analyzer="word", ngram_range=(1, 1), min_df=3,
              sublinear_tf=True), 0.5),
    ]
    S_tr_c, S_te_c, snames = [], [], []
    for name, kind, Atr, Ate, kw, Creg in VIEWS:
        cp = os.path.join(CACHE, f"stack3_{name}.npz")
        if os.path.exists(cp):
            z = np.load(cp)
            oof, pte = z["oof"], z["pte"]
        else:
            oof, pte = np.zeros(len(y)), np.zeros(len(te))
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
                clf = LogisticRegression(C=Creg, max_iter=3000, solver="liblinear")
                clf.fit(Mtr, y[itr])
                oof[iva] = clf.predict_proba(Mva)[:, 1]
                pte += clf.predict_proba(Mte)[:, 1] / NFOLD
                del Mtr, Mva, Mte, v, clf
                gc.collect()
            np.savez(cp, oof=oof, pte=pte)
        log(f"  sparse[{name}] oof acc={((oof>0.5).astype(int)==y).mean():.4f}")
        S_tr_c.append(oof)
        S_te_c.append(pte)
        snames.append(name)
    S_tr = np.column_stack(S_tr_c).astype(np.float32)
    S_te = np.column_stack(S_te_c).astype(np.float32)
    del p1_tr, p2_tr, p1_te, p2_te, dt_tr, dt_te, it_tr, it_te, VIEWS, E, E2
    gc.collect()

    # ---------------- assemble -----------------------------------------
    Xtr_a = np.hstack([D_tr_a, L_tr_a, G_tr, V_tr_a, S_tr]).astype(np.float32)
    Xtr_b = np.hstack([D_tr_b, L_tr_b, G_tr_b, V_tr_b, S_tr]).astype(np.float32)
    Xte_a = np.hstack([D_te_a, L_te_a, G_te, V_te_a, S_te]).astype(np.float32)
    Xte_b = np.hstack([D_te_b, L_te_b, G_te_b, V_te_b, S_te]).astype(np.float32)
    log("design matrix", Xtr_a.shape)

    CONFIGS = [
        ("g1", dict(max_iter=2000, learning_rate=0.045, max_leaf_nodes=31,
                    min_samples_leaf=30, l2_regularization=1.0, max_features=0.7)),
        ("g2", dict(max_iter=1800, learning_rate=0.055, max_leaf_nodes=63,
                    min_samples_leaf=60, l2_regularization=3.0, max_features=0.5)),
        ("g3", dict(max_iter=2500, learning_rate=0.04, max_leaf_nodes=15,
                    min_samples_leaf=20, l2_regularization=0.5, max_features=0.9)),
    ]
    oofs, ptes, names = [], [], []
    for cname, kw in CONFIGS:
        cp = os.path.join(CACHE, f"m3_{cname}.npz")
        if os.path.exists(cp):
            z = np.load(cp)
            oof, pte = z["oof"], z["pte"]
        else:
            oof, pte = np.zeros(len(y)), np.zeros(len(te))
            for k, (itr, iva) in enumerate(folds):
                Xf = np.vstack([Xtr_a[itr], Xtr_b[itr]])
                yf = np.concatenate([y[itr], y[itr]])
                m = HistGradientBoostingClassifier(
                    early_stopping=True, validation_fraction=0.12,
                    n_iter_no_change=70, random_state=SEED + 7 * k, **kw)
                m.fit(Xf, yf)
                oof[iva] = 0.5 * (m.predict_proba(Xtr_a[iva])[:, 1] +
                                  m.predict_proba(Xtr_b[iva])[:, 1])
                pte += 0.5 * (m.predict_proba(Xte_a)[:, 1] +
                              m.predict_proba(Xte_b)[:, 1]) / NFOLD
                log(f"  {cname} f{k} it={m.n_iter_} "
                    f"acc={(((oof[iva]>0.5).astype(int))==y[iva]).mean():.4f}")
            np.savez(cp, oof=oof, pte=pte)
        log(f" {cname} oof acc={((oof>0.5).astype(int)==y).mean():.4f}")
        oofs.append(oof)
        ptes.append(pte)
        names.append(cname)
        # keep a valid submission on disk after every finished model
        write_submission(sub_tmpl, te, np.mean(ptes, axis=0), 0.5)

    # ---------------- blend --------------------------------------------
    O, P = np.column_stack(oofs), np.column_stack(ptes)
    cands = {}
    for i, n in enumerate(names):
        cands[n] = (O[:, i], P[:, i])
    cands["mean"] = (O.mean(1), P.mean(1))
    lg = lambda z: np.log(np.clip(z, 1e-6, 1 - 1e-6) / (1 - np.clip(z, 1e-6, 1 - 1e-6)))
    cands["logitmean"] = (1 / (1 + np.exp(-lg(O).mean(1))), 1 / (1 + np.exp(-lg(P).mean(1))))
    mo, mp = np.zeros(len(y)), np.zeros(len(te))
    Ol, Pl = lg(O), lg(P)
    for itr, iva in folds:
        mm = LogisticRegression(C=1.0, max_iter=1000)
        mm.fit(Ol[itr], y[itr])
        mo[iva] = mm.predict_proba(Ol[iva])[:, 1]
        mp += mm.predict_proba(Pl)[:, 1] / NFOLD
    cands["meta"] = (mo, mp)

    best_name, best_acc = None, -1
    for n, (o, p) in cands.items():
        a = ((o > 0.5).astype(int) == y).mean()
        log(f"  blend[{n}] oof acc={a:.4f}")
        if a > best_acc:
            best_acc, best_name = a, n
    log(f"  -> selected {best_name} ({best_acc:.4f})")
    oof_f, pte_f = cands[best_name]

    ths = np.linspace(0.35, 0.65, 61)
    sc = [((oof_f > t).astype(int) == y).mean() for t in ths]
    bi = int(np.argmax(sc))
    th = float(ths[bi]) if sc[bi] - best_acc > 0.002 else 0.5
    log(f"  threshold {th:.3f} (best {ths[bi]:.3f}->{sc[bi]:.4f}, 0.5->{best_acc:.4f})")
    write_submission(sub_tmpl, te, pte_f, th)
    np.save(os.path.join(CACHE, "oof_final3.npy"), oof_f)
    np.save(os.path.join(CACHE, "pte_final3.npy"), pte_f)
    log(f"done {time.time()-t0:.0f}s")


def write_submission(sub_tmpl, te, pte, th):
    pred = (pte > th).astype(int)
    out = pd.DataFrame({"id": te["id"].values, "label": pred})
    out = sub_tmpl[["id"]].merge(out, on="id", how="left")
    assert len(out) == len(sub_tmpl), "row count mismatch"
    assert out["label"].notna().all(), "missing predictions"
    assert out["id"].is_unique, "duplicate ids"
    out["label"] = out["label"].astype(int)
    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    out.to_csv(os.path.join(ROOT, "outputs", "submission.csv"), index=False)
    log(f"  submission written pos_rate={pred.mean():.3f}")


if __name__ == "__main__":
    main()

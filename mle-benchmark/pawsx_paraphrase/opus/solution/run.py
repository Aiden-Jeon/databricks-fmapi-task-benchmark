"""PAWS-X Korean paraphrase identification.

Pipeline (train.csv only, no internet / no pretrained weights):
  1. hand-crafted similarity + word-order features (features.py)
  2. sparse char/word TF-IDF "interaction" views  ->  linear models, OOF stacked
  3. HistGradientBoosting over [dense features | stacked linear probs]
  4. symmetry augmentation (s1<->s2) for training and test-time averaging
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import features as F  # noqa: E402

SEED = 42
NFOLD = 5
CACHE = os.path.join(HERE, "cache")
os.makedirs(CACHE, exist_ok=True)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ------------------------------------------------------------------ dense feats
def dense_features(df, name):
    p = os.path.join(CACHE, f"dense_{name}.npy")
    pc = os.path.join(CACHE, f"dense_{name}_cols.npy")
    if os.path.exists(p):
        return pd.DataFrame(np.load(p), columns=np.load(pc, allow_pickle=True))
    X = F.build_matrix(df)
    np.save(p, X.values)
    np.save(pc, np.array(X.columns, dtype=object))
    return X


def swapped(df):
    d = df.copy()
    d["sentence1"], d["sentence2"] = df["sentence2"].values, df["sentence1"].values
    return d


# ------------------------------------------------------------------ text views
def prep(s):
    return F.norm(s)


def diff_text(s1, s2, stem=True):
    """Tokens that occur in only one of the two sentences (the 'edit')."""
    a = F.tokenize(prep(s1))
    b = F.tokenize(prep(s2))
    if stem:
        a, b = F.stem_tokens(a), F.stem_tokens(b)
    sa, sb = set(a), set(b)
    only = sorted((sa - sb) | (sb - sa))
    return " ".join(only) if only else "__none__"


def both_text(s1, s2):
    a = F.tokenize(prep(s1))
    b = F.tokenize(prep(s2))
    inter = sorted(set(a) & set(b))
    return " ".join(inter) if inter else "__none__"


def build_text_views(df):
    s1 = df["sentence1"].tolist()
    s2 = df["sentence2"].tolist()
    dt = [diff_text(a, b) for a, b in zip(s1, s2)]
    it = [both_text(a, b) for a, b in zip(s1, s2)]
    p1 = [prep(a) for a in s1]
    p2 = [prep(b) for b in s2]
    return dt, it, p1, p2


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

    # --- dense features (original + swapped orientation)
    log("dense features: train")
    Xtr_a = dense_features(tr, "tr_a")
    log("dense features: train swapped")
    Xtr_b = dense_features(swapped(tr), "tr_b")
    log("dense features: test")
    Xte_a = dense_features(te, "te_a")
    log("dense features: test swapped")
    Xte_b = dense_features(swapped(te), "te_b")
    cols = list(Xtr_a.columns)
    Xtr_b = Xtr_b[cols]
    Xte_a, Xte_b = Xte_a[cols], Xte_b[cols]
    log("dense dim", len(cols))

    # --- text views
    log("text views")
    dt_tr, it_tr, p1_tr, p2_tr = build_text_views(tr)
    dt_te, it_te, p1_te, p2_te = build_text_views(te)

    skf = StratifiedKFold(n_splits=NFOLD, shuffle=True, random_state=SEED)
    folds = list(skf.split(np.zeros(len(y)), y))

    # ---------------- sparse stacking models -----------------------------
    stack_tr, stack_te, stack_names = [], [], []

    def add_sparse_model(name, make_train_matrix):
        """make_train_matrix(train_idx-agnostic) -> (Mtr, Mte) fitted on all text.

        Vectorizers are refit inside each fold to avoid leakage of the target;
        text-only fitting on the union of train+test is unsupervised so we fit
        the vectorizer on train folds only for strictness.
        """
        oof = np.zeros(len(y))
        pte = np.zeros(len(te))
        for k, (itr, iva) in enumerate(folds):
            Mtr, Mva, Mte = make_train_matrix(itr, iva)
            clf = LogisticRegression(C=1.0, max_iter=2000, solver="liblinear")
            clf.fit(Mtr, y[itr])
            oof[iva] = clf.predict_proba(Mva)[:, 1]
            pte += clf.predict_proba(Mte)[:, 1] / NFOLD
        acc = ((oof > 0.5).astype(int) == y).mean()
        log(f"  sparse[{name}] oof acc={acc:.4f}")
        stack_tr.append(oof)
        stack_te.append(pte)
        stack_names.append("stk_" + name)

    def view_diff_word(itr, iva):
        v = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                            sublinear_tf=True)
        Mtr = v.fit_transform([dt_tr[i] for i in itr])
        Mva = v.transform([dt_tr[i] for i in iva])
        Mte = v.transform(dt_te)
        return Mtr, Mva, Mte

    def view_diff_char(itr, iva):
        v = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3,
                            sublinear_tf=True, max_features=300000)
        Mtr = v.fit_transform([dt_tr[i] for i in itr])
        Mva = v.transform([dt_tr[i] for i in iva])
        Mte = v.transform(dt_te)
        return Mtr, Mva, Mte

    def _pair_matrix(v, A, B):
        Va, Vb = v.transform(A), v.transform(B)
        return sparse.hstack([abs(Va - Vb), Va.multiply(Vb)], format="csr")

    def view_pair_char(itr, iva):
        v = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3,
                            sublinear_tf=True, max_features=300000)
        v.fit([p1_tr[i] for i in itr] + [p2_tr[i] for i in itr])
        Mtr = _pair_matrix(v, [p1_tr[i] for i in itr], [p2_tr[i] for i in itr])
        Mva = _pair_matrix(v, [p1_tr[i] for i in iva], [p2_tr[i] for i in iva])
        Mte = _pair_matrix(v, p1_te, p2_te)
        return Mtr, Mva, Mte

    def view_pair_word(itr, iva):
        v = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                            sublinear_tf=True)
        v.fit([p1_tr[i] for i in itr] + [p2_tr[i] for i in itr])
        Mtr = _pair_matrix(v, [p1_tr[i] for i in itr], [p2_tr[i] for i in itr])
        Mva = _pair_matrix(v, [p1_tr[i] for i in iva], [p2_tr[i] for i in iva])
        Mte = _pair_matrix(v, p1_te, p2_te)
        return Mtr, Mva, Mte

    def view_inter_word(itr, iva):
        v = TfidfVectorizer(analyzer="word", ngram_range=(1, 1), min_df=2,
                            sublinear_tf=True)
        Mtr = v.fit_transform([it_tr[i] for i in itr])
        Mva = v.transform([it_tr[i] for i in iva])
        Mte = v.transform(it_te)
        return Mtr, Mva, Mte

    log("sparse stacking models")
    add_sparse_model("diff_word", view_diff_word)
    add_sparse_model("diff_char", view_diff_char)
    add_sparse_model("pair_char", view_pair_char)
    add_sparse_model("pair_word", view_pair_word)
    add_sparse_model("inter_word", view_inter_word)

    S_tr = np.column_stack(stack_tr)
    S_te = np.column_stack(stack_te)

    # ---------------- GBM on dense + stacked ------------------------------
    def assemble(Xd, S):
        return np.hstack([Xd.values.astype(np.float64), S])

    all_cols = cols + stack_names

    Atr_a = assemble(Xtr_a, S_tr)
    Atr_b = assemble(Xtr_b, S_tr)
    Ate_a = assemble(Xte_a, S_te)
    Ate_b = assemble(Xte_b, S_te)

    log("GBM training", Atr_a.shape)
    oof = np.zeros(len(y))
    pte = np.zeros(len(te))
    for k, (itr, iva) in enumerate(folds):
        # symmetry augmentation on the training part only
        Xf = np.vstack([Atr_a[itr], Atr_b[itr]])
        yf = np.concatenate([y[itr], y[itr]])
        m = HistGradientBoostingClassifier(
            max_iter=800, learning_rate=0.05, max_leaf_nodes=31,
            min_samples_leaf=40, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=40, random_state=SEED + k,
        )
        m.fit(Xf, yf)
        oof[iva] = 0.5 * (m.predict_proba(Atr_a[iva])[:, 1] +
                          m.predict_proba(Atr_b[iva])[:, 1])
        pte += 0.5 * (m.predict_proba(Ate_a)[:, 1] +
                      m.predict_proba(Ate_b)[:, 1]) / NFOLD
        log(f"  fold {k} iters={m.n_iter_} acc={(((oof[iva]>0.5).astype(int))==y[iva]).mean():.4f}")
    acc = ((oof > 0.5).astype(int) == y).mean()
    log(f"GBM oof acc = {acc:.4f}")

    np.save(os.path.join(CACHE, "oof_gbm.npy"), oof)
    np.save(os.path.join(CACHE, "pte_gbm.npy"), pte)
    np.save(os.path.join(CACHE, "S_tr.npy"), S_tr)
    np.save(os.path.join(CACHE, "S_te.npy"), S_te)
    np.save(os.path.join(CACHE, "y.npy"), y)

    pred = (pte > 0.5).astype(int)
    out = pd.DataFrame({"id": te["id"].values, "label": pred})
    out = sub_tmpl[["id"]].merge(out, on="id", how="left")
    assert out["label"].notna().all() and len(out) == len(sub_tmpl)
    out["label"] = out["label"].astype(int)
    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    out.to_csv(os.path.join(ROOT, "outputs", "submission.csv"), index=False)
    log("wrote submission", out.shape, "pos rate", pred.mean(), f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

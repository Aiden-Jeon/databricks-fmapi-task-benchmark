"""Train models for PAWS-X ko paraphrase identification and write submission."""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE = os.path.join(HERE, "cache")
os.makedirs(CACHE, exist_ok=True)
SEED = 42
NFOLD = 5


def log(*a):
    print("[%7.1fs]" % (time.time() - T0), *a, flush=True)


T0 = time.time()


def load():
    tr = pd.read_csv(os.path.join(ROOT, "train.csv"))
    te = pd.read_csv(os.path.join(ROOT, "test.csv"))
    return tr, te


def numeric_features(tr, te):
    p = os.path.join(CACHE, "num.npz")
    fn = os.path.join(CACHE, "num_cols.npy")
    if os.path.exists(p):
        d = np.load(p)
        cols = np.load(fn, allow_pickle=True)
        return d["a"], d["b"], list(cols)
    Xtr = F.build_numeric(tr)
    Xte = F.build_numeric(te)
    Xte = Xte[Xtr.columns]
    np.savez_compressed(p, a=Xtr.values.astype(np.float32), b=Xte.values.astype(np.float32))
    np.save(fn, np.array(list(Xtr.columns), dtype=object))
    return Xtr.values.astype(np.float32), Xte.values.astype(np.float32), list(Xtr.columns)


def text_docs(tr, te):
    p = os.path.join(CACHE, "docs.npz")
    if os.path.exists(p):
        d = np.load(p, allow_pickle=True)
        return list(d["a"]), list(d["b"])
    a = [F.diff_doc(x, y) for x, y in zip(tr.sentence1.values, tr.sentence2.values)]
    b = [F.diff_doc(x, y) for x, y in zip(te.sentence1.values, te.sentence2.values)]
    np.savez_compressed(p, a=np.array(a, dtype=object), b=np.array(b, dtype=object))
    return a, b


def cross_docs(tr, te):
    """char-ngram documents of the two sentences (for a simple similarity view)."""
    def mk(df):
        return [(F.norm_text(a), F.norm_text(b))
                for a, b in zip(df.sentence1.values, df.sentence2.values)]
    return mk(tr), mk(te)


def main():
    tr, te = load()
    y = tr.label.values.astype(int)
    log("loaded", tr.shape, te.shape)

    Xn_tr, Xn_te, cols = numeric_features(tr, te)
    log("numeric features", Xn_tr.shape)

    dtr, dte = text_docs(tr, te)
    log("diff docs built")

    vec = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", min_df=3,
                          sublinear_tf=True, ngram_range=(1, 1))
    Str = vec.fit_transform(dtr)
    Ste = vec.transform(dte)
    log("diff tfidf", Str.shape)

    # char n-gram tfidf on the two sentences -> |u-v| and u*v style interactions
    ctr, cte = cross_docs(tr, te)
    cvec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=5,
                           sublinear_tf=True, max_features=200000)
    cvec.fit([a for a, b in ctr] + [b for a, b in ctr])
    A1 = cvec.transform([a for a, b in ctr]); B1 = cvec.transform([b for a, b in ctr])
    A2 = cvec.transform([a for a, b in cte]); B2 = cvec.transform([b for a, b in cte])
    Dtr = abs(A1 - B1)
    Dte = abs(A2 - B2)
    log("char diff tfidf", Dtr.shape)

    skf = StratifiedKFold(n_splits=NFOLD, shuffle=True, random_state=SEED)
    folds = list(skf.split(Xn_tr, y))

    def oof_model(name, fit_predict):
        oof = np.zeros(len(y))
        test_p = np.zeros(len(dte))
        for k, (i_tr, i_va) in enumerate(folds):
            pv, pt = fit_predict(i_tr, i_va)
            oof[i_va] = pv
            test_p += pt / NFOLD
        acc = ((oof > 0.5).astype(int) == y).mean()
        log("%s oof acc = %.4f" % (name, acc))
        return oof, test_p, acc

    results = {}

    # 1) sparse LR on diff docs
    def f_lr(i_tr, i_va):
        m = LogisticRegression(C=1.0, max_iter=2000, solver="liblinear")
        m.fit(Str[i_tr], y[i_tr])
        return m.predict_proba(Str[i_va])[:, 1], m.predict_proba(Ste)[:, 1]
    results["lr_diff"] = oof_model("lr_diff", f_lr)

    # 2) sparse LR on char diff
    def f_lrc(i_tr, i_va):
        m = LogisticRegression(C=1.0, max_iter=3000, solver="liblinear")
        m.fit(Dtr[i_tr], y[i_tr])
        return m.predict_proba(Dtr[i_va])[:, 1], m.predict_proba(Dte)[:, 1]
    results["lr_char"] = oof_model("lr_char", f_lrc)

    # 3) GBM on numeric
    def f_gbm(i_tr, i_va):
        m = HistGradientBoostingClassifier(
            max_iter=600, learning_rate=0.06, max_leaf_nodes=31,
            min_samples_leaf=20, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=40,
            random_state=SEED)
        m.fit(Xn_tr[i_tr], y[i_tr])
        return m.predict_proba(Xn_tr[i_va])[:, 1], m.predict_proba(Xn_te)[:, 1]
    results["gbm"] = oof_model("gbm", f_gbm)

    # 4) GBM on numeric + oof preds of sparse models (stacked features)
    extra_tr = np.column_stack([results["lr_diff"][0], results["lr_char"][0]])
    extra_te = np.column_stack([results["lr_diff"][1], results["lr_char"][1]])
    Xs_tr = np.hstack([Xn_tr, extra_tr]).astype(np.float32)
    Xs_te = np.hstack([Xn_te, extra_te]).astype(np.float32)

    def f_gbm2(i_tr, i_va):
        m = HistGradientBoostingClassifier(
            max_iter=600, learning_rate=0.06, max_leaf_nodes=31,
            min_samples_leaf=20, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=40,
            random_state=SEED)
        m.fit(Xs_tr[i_tr], y[i_tr])
        return m.predict_proba(Xs_tr[i_va])[:, 1], m.predict_proba(Xs_te)[:, 1]
    results["gbm_stack"] = oof_model("gbm_stack", f_gbm2)

    # blend search
    keys = list(results)
    P = np.column_stack([results[k][0] for k in keys])
    Pt = np.column_stack([results[k][1] for k in keys])
    meta = LogisticRegression(max_iter=1000)
    # simple in-sample meta (small #features, low overfit risk) evaluated by CV
    oof_meta = np.zeros(len(y))
    test_meta = np.zeros(Pt.shape[0])
    for i_tr, i_va in folds:
        meta.fit(P[i_tr], y[i_tr])
        oof_meta[i_va] = meta.predict_proba(P[i_va])[:, 1]
        test_meta += meta.predict_proba(Pt)[:, 1] / NFOLD
    acc_meta = ((oof_meta > 0.5).astype(int) == y).mean()
    log("meta oof acc = %.4f" % acc_meta)

    best_name, best_acc = max([(k, results[k][2]) for k in keys], key=lambda x: x[1])
    if acc_meta >= best_acc:
        pred_test = test_meta
        log("using meta blend")
    else:
        pred_test = results[best_name][1]
        log("using single model", best_name)

    out = pd.DataFrame({"id": te.id.values, "label": (pred_test > 0.5).astype(int)})
    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    out.to_csv(os.path.join(ROOT, "outputs", "submission.csv"), index=False)
    log("wrote submission", out.shape, "pos rate", out.label.mean())
    np.savez(os.path.join(CACHE, "oof.npz"), **{("oof_" + k): results[k][0] for k in keys},
             **{("te_" + k): results[k][1] for k in keys}, y=y)


if __name__ == "__main__":
    main()

"""Final KorFin-ASC pipeline.

Aspect-based sentiment classification with aspect-aware character/word TF-IDF views,
neighbour/target encodings derived from train only, and a blend of linear models
(+ a nonlinear model on SVD-compressed features for diversity).

Usage:  python solution/final.py          (run from the task root directory)
Writes: outputs/submission.csv
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from scipy import sparse
from scipy.special import softmax

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import build_frame, numeric_feats  # noqa: E402

SEED = 42
LABELS = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
N_SPLITS = 5

VEC_SPECS = [
    ("marked_char", "marked", dict(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                   sublinear_tf=True, max_features=300000)),
    ("marked_word", "marked", dict(analyzer="word", ngram_range=(1, 2), min_df=2,
                                   sublinear_tf=True)),
    ("masked_char", "masked", dict(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                   sublinear_tf=True, max_features=300000)),
    ("ctx_char", "ctx", dict(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                             sublinear_tf=True, max_features=300000)),
    ("ctxs_char", "ctx_s", dict(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                sublinear_tf=True, max_features=200000)),
    ("clause_char", "clause", dict(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                   sublinear_tf=True, max_features=300000)),
    ("clause_word", "clause", dict(analyzer="word", ngram_range=(1, 2), min_df=2,
                                   sublinear_tf=True)),
    ("aspect_char", "aspect", dict(analyzer="char_wb", ngram_range=(2, 4), min_df=2,
                                   sublinear_tf=True)),
    # extra views (round 3)
    ("ctx_word", "ctx", dict(analyzer="word", ngram_range=(1, 2), min_df=2,
                             sublinear_tf=True)),
    ("masked_word", "masked", dict(analyzer="word", ngram_range=(1, 1), min_df=2,
                                   sublinear_tf=True)),
    ("marked_char6", "marked", dict(analyzer="char", ngram_range=(6, 6), min_df=3,
                                    sublinear_tf=True, max_features=200000)),
]
BASE_BLOCKS = [s[0] for s in VEC_SPECS[:8]]
EXTRA_BLOCKS = [s[0] for s in VEC_SPECS[8:]]


def build_text_blocks(Ftr, Fte):
    out = {}
    for name, col, kw in VEC_SPECS:
        v = TfidfVectorizer(**kw)
        out[name] = (v.fit_transform(Ftr[col]), v.transform(Fte[col]))
    return out


def enc_features(fit_df, fit_y, apply_df, k_sent=1.0, k_asp=3.0):
    """Neighbour encodings estimated on `fit_df` only.

    Columns: [sentence-level label dist (3), n_sent, aspect-level label dist (3), n_aspect]
    The exact (sentence, aspect) pair being encoded is always excluded, so a row
    never sees its own label.
    """
    prior = np.bincount(fit_y, minlength=3) / len(fit_y)
    sent_cnt = defaultdict(lambda: np.zeros(3))
    pair_cnt = defaultdict(lambda: np.zeros(3))
    asp_cnt = defaultdict(lambda: np.zeros(3))
    for s, a, lab in zip(fit_df.sentence.values, fit_df.aspect.values, fit_y):
        sent_cnt[s][lab] += 1
        pair_cnt[(s, a)][lab] += 1
        asp_cnt[a][lab] += 1

    rows = []
    for s, a in zip(apply_df.sentence.values, apply_df.aspect.values):
        self_c = pair_cnt.get((s, a))
        c = sent_cnt.get(s)
        c = c.copy() if c is not None else np.zeros(3)
        ca = asp_cnt.get(a)
        ca = ca.copy() if ca is not None else np.zeros(3)
        if self_c is not None:
            c = np.maximum(c - self_c, 0)
            ca = np.maximum(ca - self_c, 0)
        n_s, n_a = c.sum(), ca.sum()
        p_s = (c + k_sent * prior) / (n_s + k_sent)
        p_a = (ca + k_asp * prior) / (n_a + k_asp)
        rows.append(np.concatenate([p_s, [min(n_s, 5) / 5.0], p_a, [min(n_a, 10) / 10.0]]))
    return np.asarray(rows)


def proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    return softmax(model.decision_function(X) * 2.0, axis=1)


def fit_predict_linear(Xa, ya, Xb):
    """Return dict of model-name -> probabilities on Xb."""
    res = {}
    for name, mdl in [("lr_c1", LogisticRegression(C=1, max_iter=1000)),
                      ("svc_c0.1", LinearSVC(C=0.1, dual=True))]:
        mdl.fit(Xa, ya)
        res[name] = proba(mdl, Xb)
    return res


def fit_predict_hgb(Da, ya, Db):
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                       max_leaf_nodes=31, l2_regularization=1.0,
                                       early_stopping=False, random_state=SEED)
    m.fit(Da, ya)
    return m.predict_proba(Db)


def main():
    t0 = time.time()
    tr = pd.read_csv("train.csv")
    te = pd.read_csv("test.csv")
    y = tr.label.map({l: i for i, l in enumerate(LABELS)}).values

    Ftr, Fte = build_frame(tr), build_frame(te)
    blocks = build_text_blocks(Ftr, Fte)
    num_tr, num_te = numeric_feats(tr), numeric_feats(te)
    print(f"[{time.time()-t0:.0f}s] text blocks built", flush=True)

    def stack(names, part):
        return sparse.hstack([blocks[n][part] for n in names]).tocsr()

    variants = {
        "base": BASE_BLOCKS,
        "base+extra": BASE_BLOCKS + EXTRA_BLOCKS,
    }
    Xs = {k: (stack(v, 0), stack(v, 1)) for k, v in variants.items()}
    for k, (a, b) in Xs.items():
        print(f"  {k}: {a.shape[1]} features", flush=True)

    folds = list(StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED).split(y, y))

    # ---- out-of-fold predictions -------------------------------------------
    oof = defaultdict(lambda: np.zeros((len(y), 3)))
    svd_cache = {}
    for fi, (tr_i, va_i) in enumerate(folds):
        Ea = enc_features(tr.iloc[tr_i], y[tr_i], tr.iloc[tr_i])
        Eb = enc_features(tr.iloc[tr_i], y[tr_i], tr.iloc[va_i])
        for vname, (Xtr_v, _) in Xs.items():
            Xa = sparse.hstack([Xtr_v[tr_i], sparse.csr_matrix(Ea)]).tocsr()
            Xb = sparse.hstack([Xtr_v[va_i], sparse.csr_matrix(Eb)]).tocsr()
            for mname, p in fit_predict_linear(Xa, y[tr_i], Xb).items():
                oof[f"{vname}|{mname}"][va_i] = p
        # nonlinear model on SVD of the richest view
        Xtr_v = Xs["base+extra"][0]
        svd = TruncatedSVD(n_components=180, random_state=SEED)
        Za = svd.fit_transform(Xtr_v[tr_i])
        Zb = svd.transform(Xtr_v[va_i])
        Da = np.hstack([Za, Ea, num_tr[tr_i]])
        Db = np.hstack([Zb, Eb, num_tr[va_i]])
        oof["svd|hgb"][va_i] = fit_predict_hgb(Da, y[tr_i], Db)
        print(f"[{time.time()-t0:.0f}s] fold {fi+1}/{N_SPLITS} done", flush=True)

    oof = dict(oof)
    scores = {k: f1_score(y, v.argmax(1), average="macro") for k, v in oof.items()}
    for k in sorted(scores, key=lambda x: -scores[x]):
        print(f"  OOF {k:22s} {scores[k]:.4f}")

    # ---- candidate blends ---------------------------------------------------
    def norm(p):
        return p / p.sum(1, keepdims=True)

    cands = {}
    for k in oof:
        cands[k] = {k: 1.0}
    cands["lin4"] = {f"{v}|{m}": 1.0 for v in variants for m in ("lr_c1", "svc_c0.1")}
    cands["lin4+hgb0.5"] = dict(cands["lin4"], **{"svd|hgb": 0.5})
    cands["lin4+hgb1"] = dict(cands["lin4"], **{"svd|hgb": 1.0})
    cands["extra2"] = {"base+extra|lr_c1": 1.0, "base+extra|svc_c0.1": 1.0}
    cands["base2"] = {"base|lr_c1": 1.0, "base|svc_c0.1": 1.0}
    cands["base2+hgb0.5"] = dict(cands["base2"], **{"svd|hgb": 0.5})

    blend_scores = {}
    for name, w in cands.items():
        p = sum(wt * norm(oof[k]) for k, wt in w.items())
        blend_scores[name] = f1_score(y, p.argmax(1), average="macro")
    for name in sorted(blend_scores, key=lambda x: -blend_scores[x]):
        print(f"  BLEND {name:22s} {blend_scores[name]:.4f}")
    best = max(blend_scores, key=blend_scores.get)
    print(f"SELECTED: {best} (OOF macro F1 = {blend_scores[best]:.4f})", flush=True)
    weights = cands[best]

    # ---- refit on full train and predict test -------------------------------
    E_full = enc_features(tr, y, tr)
    E_test = enc_features(tr, y, te)
    test_p = {}
    for vname, (Xtr_v, Xte_v) in Xs.items():
        Xa = sparse.hstack([Xtr_v, sparse.csr_matrix(E_full)]).tocsr()
        Xb = sparse.hstack([Xte_v, sparse.csr_matrix(E_test)]).tocsr()
        for mname, p in fit_predict_linear(Xa, y, Xb).items():
            test_p[f"{vname}|{mname}"] = p
    if "svd|hgb" in weights:
        Xtr_v, Xte_v = Xs["base+extra"]
        svd = TruncatedSVD(n_components=180, random_state=SEED)
        Za = svd.fit_transform(Xtr_v)
        Zb = svd.transform(Xte_v)
        test_p["svd|hgb"] = fit_predict_hgb(np.hstack([Za, E_full, num_tr]), y,
                                            np.hstack([Zb, E_test, num_te]))
    print(f"[{time.time()-t0:.0f}s] full-train fits done", flush=True)

    P = sum(wt * norm(test_p[k]) for k, wt in weights.items())
    pred = P.argmax(1)
    os.makedirs("outputs", exist_ok=True)
    sub = pd.DataFrame({"id": te.id, "label": [LABELS[i] for i in pred]})
    sub.to_csv("outputs/submission.csv", index=False)
    print(sub.label.value_counts().to_string())
    print(f"wrote outputs/submission.csv ({len(sub)} rows) [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()

"""KorSTS sentence-similarity — final reproducible pipeline.

Approach (no pretrained models / no internet; only train.csv is used for learning):
  1. Hand-crafted lexical/character/jamo overlap + TF-IDF-cosine + LSA features.
  2. Distributional word vectors (PPMI + SVD) learned from the training corpus,
     using within-sentence contexts *and* cross-sentence contexts of each pair
     (pair structure only, no labels) -> soft-alignment features.
  3. BM25 / residual-IDF / content-word features.
  4. A model zoo (ExtraTrees, RandomForest, HistGB, SVR-RBF, KernelRidge, Ridge,
     plus sparse "learned-metric" ridges on element-wise-min TF-IDF pair vectors)
     stacked with non-negative-least-squares weights fit on out-of-fold predictions.

Metric: Pearson correlation.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import nnls
from scipy.stats import pearsonr
from sklearn.ensemble import (ExtraTreesRegressor, HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler, normalize
from sklearn.svm import SVR

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from feats import FeatureBuilder, norm, nopunct, jamo, stoks  # noqa: E402
from feats2 import WordVecs, extra_features  # noqa: E402
from feats3 import extra3  # noqa: E402

DATA = os.path.abspath(os.path.join(HERE, os.pardir))
OUT = os.path.join(DATA, "outputs")
N_FOLDS = 10
SEED = 42
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ------------------------------------------------------------------ data
tr = pd.read_csv(os.path.join(DATA, "train.csv"))
te = pd.read_csv(os.path.join(DATA, "test.csv"))
y = tr.score.values.astype(np.float64)
log("data", tr.shape, te.shape)

# ------------------------------------------------------------------ features
fb = FeatureBuilder(svd_dim=180, seed=0).fit(tr.sentence1.values, tr.sentence2.values)
wv = WordVecs(dim=150, seed=0, cross_w=3.0).fit(tr.sentence1.values, tr.sentence2.values)
log("unsupervised components fitted; wv vocab =", len(wv.vocab))

corpus_toks = [stoks(norm(s)) for s in list(tr.sentence1) + list(tr.sentence2)]
avgdl = float(np.mean([len(t) for t in corpus_toks]))
freq = pd.Series([t for toks_ in corpus_toks for t in toks_]).value_counts()
STOP = set(freq.index[:120])


def build(s1, s2):
    F = fb.transform(s1, s2)
    E = extra_features(s1, s2, wv, fb.idf_, fb.idf_default_)
    G = extra3(s1, s2, wv, fb.idf_, fb.idf_default_, avgdl, STOP)
    return F, E, G


Ftr, Etr, Gtr = build(tr.sentence1.values, tr.sentence2.values)
log("train features", Ftr.shape, Etr.shape, Gtr.shape)
Fte, Ete, Gte = build(te.sentence1.values, te.sentence2.values)
log("test features done")

# symmetrize order-dependent columns (STS similarity is symmetric)
PAIRS = [("n_tok_a", "n_tok_b"), ("clen_a", "clen_b"), ("num_a", "num_b"),
         ("lat_a", "lat_b"), ("salign_ab", "salign_ba"),
         ("ntok_a_only", "ntok_b_only")]


def symmetrize(X):
    X = X.copy()
    for a, b in PAIRS:
        lo, hi = np.minimum(X[a], X[b]), np.maximum(X[a], X[b])
        X[a], X[b] = lo, hi
    return X


Ftr, Fte = symmetrize(Ftr), symmetrize(Fte)

XE_tr = pd.concat([Ftr, Etr], axis=1)
XE_te = pd.concat([Fte, Ete], axis=1)
XA_tr = pd.concat([Ftr, Etr, Gtr], axis=1)
XA_te = pd.concat([Fte, Ete, Gte], axis=1)
keep = XA_tr.columns[(XA_tr.std().values > 0)]
XA_tr, XA_te = XA_tr[keep], XA_te[keep]
keep2 = XE_tr.columns[(XE_tr.std().values > 0)]
XE_tr, XE_te = XE_tr[keep2], XE_te[keep2]
log("feature matrices", XA_tr.shape, XE_tr.shape)

# ------------------------------------------------------------------ sparse pair views
def prep(s, kind):
    s = [norm(x) for x in s]
    if kind == "jamo":
        return [jamo(nopunct(x)) for x in s]
    if kind == "word":
        return [" ".join(stoks(x)) for x in s]
    return s


SPARSE = {}
for kind, ngr, mindf in [("char", (2, 3), 2), ("jamo", (3, 5), 3)]:
    a_tr, b_tr = prep(tr.sentence1, kind), prep(tr.sentence2, kind)
    a_te, b_te = prep(te.sentence1, kind), prep(te.sentence2, kind)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=ngr, sublinear_tf=True,
                          min_df=mindf).fit(a_tr + b_tr)
    Xa, Xb = normalize(vec.transform(a_tr)), normalize(vec.transform(b_tr))
    Ya, Yb = normalize(vec.transform(a_te)), normalize(vec.transform(b_te))
    SPARSE[kind] = (Xa.minimum(Xb).tocsr(), Ya.minimum(Yb).tocsr())
log("sparse pair views built")

# ------------------------------------------------------------------ models
kf = list(KFold(N_FOLDS, shuffle=True, random_state=SEED).split(XA_tr))

qt = lambda: QuantileTransformer(output_distribution="normal", n_quantiles=500,
                                 random_state=0)

MODELS = [
    ("etr_A", "A", lambda: ExtraTreesRegressor(n_estimators=800, min_samples_leaf=2,
                                               max_features=0.5, n_jobs=-1, random_state=0)),
    ("etr_A2", "A", lambda: ExtraTreesRegressor(n_estimators=800, min_samples_leaf=4,
                                                max_features=0.8, n_jobs=-1, random_state=7)),
    ("rf_A", "A", lambda: RandomForestRegressor(n_estimators=600, min_samples_leaf=2,
                                                max_features=0.4, n_jobs=-1, random_state=0)),
    ("gbm_deep_A", "A", lambda: HistGradientBoostingRegressor(
        max_iter=1200, learning_rate=0.03, max_leaf_nodes=63, min_samples_leaf=10,
        l2_regularization=2.0, max_features=0.7, early_stopping=False, random_state=1)),
    ("gbm_shal_A", "A", lambda: HistGradientBoostingRegressor(
        max_iter=1500, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=30,
        l2_regularization=0.5, max_features=0.8, early_stopping=False, random_state=2)),
    ("svr_E", "E", lambda: make_pipeline(qt(), SVR(C=8, gamma="scale", epsilon=0.1))),
    ("svr_E2", "E", lambda: make_pipeline(qt(), SVR(C=3, gamma="scale", epsilon=0.2))),
    ("svr_A", "A", lambda: make_pipeline(qt(), SVR(C=8, gamma="scale", epsilon=0.1))),
    ("krr_E", "E", lambda: make_pipeline(StandardScaler(), KernelRidge(alpha=1.0,
                                                                      kernel="rbf", gamma=0.02))),
    ("ridge_A", "A", lambda: make_pipeline(qt(), RidgeCV(alphas=np.logspace(-2, 3, 20)))),
    ("sp_char", "sp_char", lambda: Ridge(alpha=0.3, solver="sparse_cg", max_iter=3000)),
    ("sp_jamo", "sp_jamo", lambda: Ridge(alpha=0.3, solver="sparse_cg", max_iter=3000)),
]

VIEWS = {"A": (XA_tr, XA_te), "E": (XE_tr, XE_te),
         "sp_char": SPARSE["char"], "sp_jamo": SPARSE["jamo"]}

names, OOF, PRED = [], [], []
for name, view, mk in MODELS:
    Xtr_, Xte_ = VIEWS[view]
    is_sp = view.startswith("sp_")
    oof = np.zeros(len(y))
    pr = np.zeros(len(te))
    t = time.time()
    for trn, val in kf:
        m = mk()
        if is_sp:
            m.fit(Xtr_[trn], y[trn])
            oof[val] = m.predict(Xtr_[val])
        else:
            m.fit(Xtr_.iloc[trn], y[trn])
            oof[val] = m.predict(Xtr_.iloc[val])
        pr += m.predict(Xte_) / len(kf)
    names.append(name)
    OOF.append(oof)
    PRED.append(pr)
    log(f"{name:12s} oof_pearson={pearsonr(oof, y)[0]:.5f}  ({time.time()-t:.0f}s)")

OOF = np.column_stack(OOF)
PRED = np.column_stack(PRED)

# ------------------------------------------------------------------ blending
def score(v):
    return pearsonr(v, y)[0]


log("equal-weight blend:", round(score(OOF.mean(1)), 5))

# non-negative least squares stack (with intercept via centering)
A = np.column_stack([OOF, np.ones(len(y))])
w, _ = nnls(A, y)
blend_nnls = PRED @ w[:-1] + w[-1]
oof_nnls = OOF @ w[:-1] + w[-1]
log("nnls blend:", round(score(oof_nnls), 5), dict(zip(names, np.round(w[:-1], 3))))

# greedy forward selection with replacement (robust, low-variance weights)
sel, cur, best = [], np.zeros(len(y)), -1.0
for _ in range(30):
    cands = [(score((cur * len(sel) + OOF[:, j]) / (len(sel) + 1)), j)
             for j in range(OOF.shape[1])]
    r, j = max(cands)
    if r <= best + 1e-5:
        break
    cur = (cur * len(sel) + OOF[:, j]) / (len(sel) + 1)
    sel.append(j)
    best = r
cnt = np.bincount(sel, minlength=OOF.shape[1]) / max(len(sel), 1)
blend_greedy = PRED @ cnt
log("greedy blend:", round(best, 5), {names[j]: round(c, 3) for j, c in enumerate(cnt) if c})

# average the two stackers (reduces weight-selection variance)
oof_final = 0.5 * (oof_nnls / oof_nnls.std() + cur / cur.std())
pred_final = 0.5 * (blend_nnls / oof_nnls.std() + blend_greedy / cur.std())
log("averaged stack:", round(score(oof_final), 5))

cands = {"nnls": (oof_nnls, blend_nnls), "greedy": (cur, blend_greedy),
         "avg": (oof_final, pred_final)}
bkey = max(cands, key=lambda k: score(cands[k][0]))
oof_b, pred_b = cands[bkey]
log("chosen stacker:", bkey, round(score(oof_b), 5))

np.savez(os.path.join(HERE, "stack_raw.npz"), oof=OOF, pred=PRED,
         names=np.array(names), y=y)

# Map to the training score distribution. Pearson is invariant to affine maps, so
# instead of clipping (which is non-linear and loses information at the borders)
# we shrink the spread just enough for everything to land inside [0, 5].
pred = (pred_b - pred_b.mean()) / (pred_b.std() + 1e-12) * y.std() + y.mean()
mu = float(y.mean())
shrink = 1.0
lo, hi = float(pred.min()), float(pred.max())
if lo < 0.0:
    shrink = min(shrink, (mu - 0.0) / (mu - lo))
if hi > 5.0:
    shrink = min(shrink, (5.0 - mu) / (hi - mu))
pred = mu + (pred - mu) * shrink * 0.999
pred = np.clip(pred, 0.0, 5.0)
log("shrink factor", round(shrink, 4), "range", round(pred.min(), 3), round(pred.max(), 3))

# ------------------------------------------------------------------ submission
os.makedirs(OUT, exist_ok=True)
sub = pd.DataFrame({"id": te.id.values, "score": pred})
assert len(sub) == len(te) and sub.id.is_unique and sub.score.notna().all()
assert sub.score.std() > 1e-6, "constant prediction"
sub.to_csv(os.path.join(OUT, "submission.csv"), index=False)
np.save(os.path.join(HERE, "oof.npy"), oof_b)
log("wrote", os.path.join(OUT, "submission.csv"), sub.shape,
    "pred std", round(float(sub.score.std()), 3))

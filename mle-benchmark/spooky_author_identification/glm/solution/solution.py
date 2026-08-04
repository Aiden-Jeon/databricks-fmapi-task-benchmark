"""Final tuned solution for spooky author ID.

Key improvements over solution.py:
- More TF-IDF configs (char n-grams at multiple ranges, word 1-1/1-2/1-3)
- Lower-alpha MultinomialNB variants
- NBSVM with tuned beta/C
- BernoulliNB on binary features
- Stacked features for LogReg
- Exhaustive ensemble weight search (pairwise + triple over top models)
"""
import warnings
import itertools
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder, Binarizer

warnings.filterwarnings("ignore")

BASE = "/tmp/kmle/M3_t2_spooky_full_20260804_033550/task"
CLASSES = ["EAP", "HPL", "MWS"]
SEED = 42

train = pd.read_csv(f"{BASE}/train.csv")
test = pd.read_csv(f"{BASE}/test.csv")
sub = pd.read_csv(f"{BASE}/sample_submission.csv")

le = LabelEncoder().fit(CLASSES)
y = le.transform(train["author"].values)
X_text = train["text"].fillna("").astype(str).str.lower().values
Xt_text = test["text"].fillna("").astype(str).str.lower().values


def make_vec_word(ngram=(1, 2), min_df=3, max_df=1.0):
    return TfidfVectorizer(
        ngram_range=ngram, min_df=min_df, max_df=max_df,
        sublinear_tf=True, strip_accents="unicode",
        lowercase=False, token_pattern=r"\w+",
    )


def make_vec_char(ngram=(3, 5), min_df=5, max_df=1.0):
    return TfidfVectorizer(
        ngram_range=ngram, min_df=min_df, max_df=max_df,
        sublinear_tf=True, strip_accents="unicode",
        lowercase=False, analyzer="char_wb",
    )


def kfold_proba(est_fn, X, y, Xt, splits=5, seed=SEED):
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    oof = np.zeros((len(y), 3))
    test_pred = np.zeros((Xt.shape[0], 3))
    for tr, va in skf.split(np.zeros(len(y)), y):
        clf = est_fn()
        clf.fit(X[tr], y[tr])
        oof[va] = clf.predict_proba(X[va])
        test_pred += clf.predict_proba(Xt) / splits
    return oof, test_pred


def normalize(p, eps=1e-12):
    p = np.clip(p, eps, None)
    p = p / p.sum(axis=1, keepdims=True)
    return p


def ll(y, p):
    return log_loss(y, p, labels=[0, 1, 2])


def pr(X, y_i, y, alpha=1.0):
    p = np.asarray(X[y == y_i].sum(0)).ravel()
    return (p + alpha) / ((y == y_i).sum() + alpha * X.shape[1])


class NBSVM:
    """One-vs-rest NBSVM with NB log-count ratio features, blended with MNB."""
    def __init__(self, alpha=1.0, C=1.0, beta=0.25):
        self.alpha = alpha
        self.C = C
        self.beta = beta
        self.lrs = []
        self.r = None
        self.nb = None

    def fit(self, X, y):
        self.r = np.zeros((3, X.shape[1]))
        for c in range(3):
            p1 = pr(X, c, y, self.alpha)
            p0 = pr(X, -c, y, self.alpha)
            self.r[c] = np.log(p1 / p0)
        self.nb = MultinomialNB(alpha=self.alpha)
        self.nb.fit(X, y)
        self.lrs = []
        for c in range(3):
            Xc = X.multiply(self.r[c]).tocsr()
            lr = LogisticRegression(
                C=self.C, solver="liblinear", max_iter=3000,
                random_state=SEED, fit_intercept=True,
            )
            yb = (y == c).astype(int)
            lr.fit(Xc, yb)
            self.lrs.append(lr)
        return self

    def predict_proba(self, X):
        scores = np.zeros((X.shape[0], 3))
        for c in range(3):
            Xc = X.multiply(self.r[c]).tocsr()
            scores[:, c] = self.lrs[c].decision_function(Xc)
        scores = scores - scores.max(axis=1, keepdims=True)
        e = np.exp(scores)
        p = e / e.sum(axis=1, keepdims=True)
        nb_p = self.nb.predict_proba(X)
        p = self.beta * p + (1 - self.beta) * nb_p
        return p


print("Building features...")
# word vectors
vec_w11 = make_vec_word((1, 1), 2); Xw11 = vec_w11.fit_transform(X_text); Xtw11 = vec_w11.transform(Xt_text)
vec_w12 = make_vec_word((1, 2), 3); Xw12 = vec_w12.fit_transform(X_text); Xtw12 = vec_w12.transform(Xt_text)
vec_w13 = make_vec_word((1, 3), 3); Xw13 = vec_w13.fit_transform(X_text); Xtw13 = vec_w13.transform(Xt_text)

# char vectors
vec_c13 = make_vec_char((1, 3), 5); Xc13 = vec_c13.fit_transform(X_text); Xtc13 = vec_c13.transform(Xt_text)
vec_c35 = make_vec_char((3, 5), 5); Xc35 = vec_c35.fit_transform(X_text); Xtc35 = vec_c35.transform(Xt_text)
vec_c25 = make_vec_char((2, 5), 5); Xc25 = vec_c25.fit_transform(X_text); Xtc25 = vec_c25.transform(Xt_text)
vec_c36 = make_vec_char((3, 6), 5); Xc36 = vec_c36.fit_transform(X_text); Xtc36 = vec_c36.transform(Xt_text)

# binary count features for BernoulliNB
Xw12_bin = Xw12.copy()
Xw12_bin.data = np.ones_like(Xw12_bin.data)
Xtw12_bin = Xtw12.copy()
Xtw12_bin.data = np.ones_like(Xtw12_bin.data)

oofs = {}
tpreds = {}

def run(name, est_fn, X, Xt, y_):
    oof, tp = kfold_proba(est_fn, X, y_, Xt)
    oof = normalize(oof)
    tp = normalize(tp)
    oofs[name] = oof
    tpreds[name] = tp
    print(f"  {name} OOF: {ll(y_, oof):.5f}")


print("MultinomialNB variants (low alpha):")
for nm, X, Xt, a in [
    ("mnb_w12_a0.02", Xw12, Xtw12, 0.02),
    ("mnb_w12_a0.05", Xw12, Xtw12, 0.05),
    ("mnb_w12_a0.1", Xw12, Xtw12, 0.1),
    ("mnb_w13_a0.05", Xw13, Xtw13, 0.05),
    ("mnb_w11_a0.05", Xw11, Xtw11, 0.05),
    ("mnb_c35_a0.05", Xc35, Xtc35, 0.05),
    ("mnb_c35_a0.1", Xc35, Xtc35, 0.1),
    ("mnb_c25_a0.05", Xc25, Xtc25, 0.05),
    ("mnb_c36_a0.05", Xc36, Xtc36, 0.05),
]:
    run(nm, lambda a=a: MultinomialNB(alpha=a), X, Xt, y)

print("BernoulliNB:")
run("bnb_w12_a0.1", lambda: BernoulliNB(alpha=0.1), Xw12_bin, Xtw12_bin, y)
run("bnb_w12_a0.05", lambda: BernoulliNB(alpha=0.05), Xw12_bin, Xtw12_bin, y)

print("ComplementNB:")
run("cnb_w12_a0.1", lambda: ComplementNB(alpha=0.1), Xw12, Xtw12, y)

print("NBSVM variants:")
for nm, X, Xt, a, c, b in [
    ("nbsvm_w12_a0.02_C4_b0.25", Xw12, Xtw12, 0.02, 4.0, 0.25),
    ("nbsvm_w12_a0.05_C4_b0.25", Xw12, Xtw12, 0.05, 4.0, 0.25),
    ("nbsvm_w12_a0.1_C4_b0.25", Xw12, Xtw12, 0.1, 4.0, 0.25),
    ("nbsvm_w12_a0.1_C8_b0.25", Xw12, Xtw12, 0.1, 8.0, 0.25),
    ("nbsvm_w12_a0.1_C4_b0.5", Xw12, Xtw12, 0.1, 4.0, 0.5),
    ("nbsvm_w13_a0.1_C4_b0.25", Xw13, Xtw13, 0.1, 4.0, 0.25),
    ("nbsvm_w12_a0.1_C4_b0.1", Xw12, Xtw12, 0.1, 4.0, 0.1),
]:
    run(nm, lambda a=a, c=c, b=b: NBSVM(alpha=a, C=c, beta=b), X, Xt, y)

print("LogisticRegression on stacked features:")
X_stk = hstack([Xw12, Xc35]).tocsr()
Xt_stk = hstack([Xtw12, Xtc35]).tocsr()
run("lr_stk_w12c35_c2", lambda: LogisticRegression(C=2.0, solver="liblinear", max_iter=2000, random_state=SEED), X_stk, Xt_stk, y)
run("lr_stk_w12c35_c1", lambda: LogisticRegression(C=1.0, solver="liblinear", max_iter=2000, random_state=SEED), X_stk, Xt_stk, y)

X_stk2 = hstack([Xw12, Xw13, Xc35, Xc25]).tocsr()
Xt_stk2 = hstack([Xtw12, Xtw13, Xtc35, Xtc25]).tocsr()
run("lr_stk_all_c2", lambda: LogisticRegression(C=2.0, solver="liblinear", max_iter=2000, random_state=SEED), X_stk2, Xt_stk2, y)

# -------- Ensemble search --------
from scipy.optimize import minimize

names = list(oofs.keys())
n = len(names)
print(f"\nSearching ensemble weights over {n} models...")

best = None

# Equal-weight subsets (size 1..6)
for r in range(1, min(n, 6) + 1):
    for subset in itertools.combinations(range(n), r):
        oof_avg = np.zeros_like(oofs[names[0]])
        for idx in subset:
            oof_avg += oofs[names[idx]]
        oof_avg = normalize(oof_avg / len(subset))
        llv = ll(y, oof_avg)
        if best is None or llv < best[0]:
            best = (llv, [(names[i], 1.0) for i in subset])

# Pairwise fine grid
for i in range(n):
    for j in range(i + 1, n):
        a = oofs[names[i]]
        b = oofs[names[j]]
        for w in np.arange(0.0, 1.001, 0.025):
            m = normalize(a * w + b * (1 - w))
            llv = ll(y, m)
            if llv < best[0]:
                best = (llv, [(names[i], w), (names[j], 1 - w)])

# Triple coarse grid over top-6 individual models
top = sorted(oofs.items(), key=lambda kv: ll(y, kv[1]))[:6]
print("Top individual OOF:", [(k, round(ll(y, v), 5)) for k, v in top])
for i in range(len(top)):
    for j in range(i + 1, len(top)):
        for k in range(j + 1, len(top)):
            ni, ai = top[i]; nj, aj = top[j]; nk, ak = top[k]
            for wi in np.arange(0.0, 1.001, 0.1):
                for wj in np.arange(0.0, 1.001 - wi, 0.1):
                    wk = 1 - wi - wj
                    m = normalize(ai * wi + aj * wj + ak * wk)
                    llv = ll(y, m)
                    if llv < best[0]:
                        best = (llv, [(ni, wi), (nj, wj), (nk, wk)])

# Continuous SLSQP optimization over top-10 models
top10 = [k for k, _ in sorted(oofs.items(), key=lambda kv: ll(y, kv[1]))[:10]]
A = np.stack([oofs[k] for k in top10], axis=0)  # (10, N, 3)


def loss_fn(w):
    w = np.maximum(w, 0)
    s = w.sum()
    if s < 1e-9:
        return 1e9
    p = np.tensordot(w, A, axes=([0], [0])) / s
    return ll(y, normalize(p))


best_w = None
for _ in range(40):
    w0 = np.random.dirichlet(np.ones(len(top10)))
    res = minimize(loss_fn, w0, method="SLSQP",
                   bounds=[(0, 1)] * len(top10),
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   options={"maxiter": 200, "ftol": 1e-7})
    if res.success and res.fun < (best[0] if best else 1e9):
        if best_w is None or res.fun < best_w[0]:
            best_w = (res.fun, res.x)
if best_w is not None and best_w[0] < best[0]:
    w = np.maximum(best_w[1], 0)
    w = w / w.sum()
    best = (best_w[0], [(top10[i], float(w[i])) for i in range(len(top10)) if w[i] > 1e-4])

print("Best ensemble:", best[0])
for nm, w in best[1]:
    print(f"  {nm}: {w:.4f}")

# Build final test predictions
test_final = np.zeros_like(tpreds[names[0]])
total_w = 0.0
for nm, w in best[1]:
    test_final += tpreds[nm] * w
    total_w += w
test_final = normalize(test_final / total_w)

eps = 1e-6
test_final = np.clip(test_final, eps, 1 - eps)
test_final = test_final / test_final.sum(axis=1, keepdims=True)

out = sub.copy()
out["EAP"] = test_final[:, 0]
out["HPL"] = test_final[:, 1]
out["MWS"] = test_final[:, 2]
out.to_csv(f"{BASE}/outputs/submission.csv", index=False)
print(f"\nWrote submission. OOF logloss: {best[0]:.5f}")
print(out.head())
print("Rows:", len(out))

# -*- coding: utf-8 -*-
"""
v5: 다양성 추가 (TruncatedSVD 임베딩 + LR, char n-gram SVC 등) + 중첩CV 앙상블 재실행.
캐시(oof_v4.npz)에 신규 모델만 추가.
"""
import os, time, warnings
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from scipy.optimize import minimize
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

RANDOM_STATE = 42
N_SPLITS = 5
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES = ["EAP", "HPL", "MWS"]

train = pd.read_csv(os.path.join(BASE, "train.csv"))
test = pd.read_csv(os.path.join(BASE, "test.csv"))
y = LabelEncoder().fit(CLASSES).transform(train["author"])
t0 = time.time()

# 기존 캐시 로드
z = np.load(os.path.join(BASE, "solution", "oof_v4.npz"))
oofs, testps, lls = {}, {}, {}
for k in z.files:
    if k.startswith("oof_"):
        oofs[k[4:]] = z[k]; lls[k[4:]] = log_loss(y, z[k], labels=[0,1,2])
    elif k.startswith("test_"):
        testps[k[5:]] = z[k]
print(f"loaded {len(oofs)} cached models")

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

def run_dense(name, maker, Xtr, Xte):
    oof = np.zeros((Xtr.shape[0], 3))
    for tr_idx, va_idx in skf.split(Xtr, y):
        m = maker(); m.fit(Xtr[tr_idx], y[tr_idx])
        p = m.predict_proba(Xtr[va_idx])
        oof[va_idx] = p[:, [list(m.classes_).index(c) for c in range(3)]]
    ll = log_loss(y, oof, labels=[0,1,2])
    m = maker(); m.fit(Xtr, y)
    tp = m.predict_proba(Xte)
    tp = tp[:, [list(m.classes_).index(c) for c in range(3)]]
    print(f"  {name}: {ll:.5f} ({time.time()-t0:.0f}s)")
    lls[name] = ll; oofs[name] = oof; testps[name] = tp

def run_sparse(name, maker, Xtr, Xte):
    run_dense(name, maker, Xtr, Xte)

# --- 신규 피처 ---
word12 = TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=2, sublinear_tf=True, strip_accents="unicode", max_features=120000)
W12tr, W12te = word12.fit_transform(train["text"]), word12.transform(test["text"])

char26 = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,6), min_df=3, sublinear_tf=True, max_features=160000)
C26tr, C26te = char26.fit_transform(train["text"]), char26.transform(test["text"])

word123 = TfidfVectorizer(analyzer="word", ngram_range=(1,3), min_df=2, sublinear_tf=True, strip_accents="unicode", max_features=160000)
W123tr, W123te = word123.fit_transform(train["text"]), word123.transform(test["text"])

combo_tr = hstack([W123tr, C26tr]).tocsr()
combo_te = hstack([W123te, C26te]).tocsr()

def LR(C): return lambda: LogisticRegression(C=C, solver="saga", multi_class="multinomial", max_iter=600, n_jobs=4, random_state=RANDOM_STATE)
def MNB(a): return lambda: MultinomialNB(alpha=a)

# 1) SVD(word12) + LR (dense 문서 임베딩 대용)
for dim, C in [(200, 4), (300, 8)]:
    svd = TruncatedSVD(n_components=dim, random_state=RANDOM_STATE)
    Ztr = svd.fit_transform(W12tr); Zte = svd.transform(W12te)
    run_dense(f"svd{dim}_w12_lr", LR(C), Ztr, Zte)

# 2) SVD(char26) + LR
for dim, C in [(300, 4)]:
    svd = TruncatedSVD(n_components=dim, random_state=RANDOM_STATE)
    Ztr = svd.fit_transform(C26tr); Zte = svd.transform(C26te)
    run_dense(f"svd{dim}_c26_lr", LR(C), Ztr, Zte)

# 3) high-C LR on word12 sparse (fast)
run_sparse("lr_w12_C20", LR(20), W12tr, W12te)

# 4) MNB on char26 tuned
run_sparse("mnb_c26_a50", MNB(0.5), C26tr, C26te)
run_sparse("mnb_c26_a15", MNB(0.15), C26tr, C26te)

# 5) Calibrated LinearSVC high C on combo
def svc(C): return lambda: CalibratedClassifierCV(LinearSVC(C=C, random_state=RANDOM_STATE, max_iter=4000), cv=3, method="sigmoid")
run_sparse("svc_combo_C1", svc(1.0), combo_tr, combo_te)

# 6) extra seeds LR (bagging diversity) - 2 extra seeds for best model
for seed in [1, 7]:
    def LRs(C, s): return lambda: LogisticRegression(C=C, solver="saga", multi_class="multinomial", max_iter=600, n_jobs=4, random_state=s)
    run_sparse(f"lr_cwc2_C25_s{seed}", LRs(25, seed), combo_tr, combo_te)

print(f"\ntotal models now: {len(oofs)}")

# ---------------- nested-CV ensemble ----------------
def fit_slsqp(O_train, y_train, restarts=5):
    M = O_train.shape[0]
    def obj(w):
        p = np.tensordot(w, O_train, axes=1)
        p = np.clip(p, 1e-9, 1.0); p /= p.sum(1, keepdims=True)
        return log_loss(y_train, p, labels=[0,1,2])
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1})
    bnds = [(0,1)] * M
    inits = [np.eye(M)[0], np.ones(M)/M]
    rng = np.random.default_rng(0)
    inits += [rng.dirichlet(np.ones(M)) for _ in range(restarts)]
    best = (1e9, None)
    for w0 in inits:
        r = minimize(obj, w0, method="SLSQP", bounds=bnds, constraints=cons,
                     options={"maxiter": 400, "ftol": 1e-10})
        if r.fun < best[0]:
            best = (r.fun, r.x)
    return best[1]

def fit_temp(logp, y_train):
    def obj(T):
        q = logp / T[0]; q -= q.max(1, keepdims=True)
        p = np.exp(q); p /= p.sum(1, keepdims=True)
        return log_loss(y_train, p, labels=[0,1,2])
    r = minimize(obj, [1.0], method="Nelder-Mead", options={"xatol":1e-4, "fatol":1e-7})
    return float(r.x[0])

ranked = sorted(lls, key=lls.get)
print("\ntop:", [(n, round(lls[n],4)) for n in ranked[:12]])
TOP = ranked[:14]
names = TOP
O_all = np.stack([oofs[n] for n in names])
P_all = np.stack([testps[n] for n in names])

outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=777)
oof_ens = np.zeros((len(y), 3)); oof_ens_t = np.zeros((len(y), 3))
for tr_idx, va_idx in outer.split(np.zeros(len(y)), y):
    w = fit_slsqp(O_all[:, tr_idx, :], y[tr_idx])
    pv = np.tensordot(w, O_all[:, va_idx, :], axes=1)
    pv = np.clip(pv, 1e-9, 1.0); pv /= pv.sum(1, keepdims=True)
    oof_ens[va_idx] = pv
    T = fit_temp(np.log(np.clip(np.tensordot(w, O_all[:, tr_idx, :], axes=1), 1e-12, 1.0)), y[tr_idx])
    q = np.log(np.clip(pv, 1e-12, 1.0)) / T; q -= q.max(1, keepdims=True)
    p2 = np.exp(q); p2 /= p2.sum(1, keepdims=True)
    oof_ens_t[va_idx] = p2

ll_ens = log_loss(y, oof_ens, labels=[0,1,2])
ll_ens_t = log_loss(y, oof_ens_t, labels=[0,1,2])
print(f"\nnested-CV: ens={ll_ens:.5f}  ens+temp={ll_ens_t:.5f}")

w_full = fit_slsqp(O_all, y)
pf = np.tensordot(w_full, O_all, axes=1)
pf = np.clip(pf, 1e-9, 1.0); pf /= pf.sum(1, keepdims=True)
ll_full = log_loss(y, pf, labels=[0,1,2])
print(f"full-fit slsqp OOF: {ll_full:.5f}")
for n, w in sorted(zip(names, w_full), key=lambda x: -x[1]):
    if w > 0.02: print(f"  {n}: {w:.3f}")

final = np.tensordot(w_full, P_all, axes=1)
final = np.clip(final, 1e-9, 1.0); final /= final.sum(1, keepdims=True)
T_full = None
if ll_ens_t < ll_ens:
    T_full = fit_temp(np.log(np.clip(pf, 1e-12, 1.0)), y)
    q = np.log(np.clip(final, 1e-12, 1.0)) / T_full
    q -= q.max(1, keepdims=True)
    final = np.exp(q); final /= final.sum(1, keepdims=True)
    print(f"applied temperature T={T_full:.3f}")
final = np.clip(final, 1e-6, 1.0); final /= final.sum(1, keepdims=True)

sub = pd.DataFrame({"id": test["id"], "EAP": final[:,0], "HPL": final[:,1], "MWS": final[:,2]})
assert sub["id"].is_unique and set(sub["id"]) == set(test["id"])
sub.to_csv(os.path.join(BASE, "outputs", "submission.csv"), index=False)

np.savez(os.path.join(BASE, "solution", "oof_v5.npz"),
         **{f"oof_{k}": v for k, v in oofs.items()},
         **{f"test_{k}": v for k, v in testps.items()})
import json
json.dump({"lls": lls, "TOP": names, "w_full": w_full.tolist(), "ll_full": ll_full,
           "nested": {"ens": ll_ens, "ens_temp": ll_ens_t}, "T_full": T_full},
          open(os.path.join(BASE, "solution", "oof_v5.json"), "w"), indent=2)
print(f"saved ({time.time()-t0:.0f}s total)")

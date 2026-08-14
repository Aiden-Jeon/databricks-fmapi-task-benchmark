# -*- coding: utf-8 -*-
"""
v4: 중첩 CV 앙상블 (SLSQP 가중치 + 온도를 out-of-fold로 추정해 과적합 억제)
+ high-C LR 모델 추가 (lr_cwc2_C15/C25, liblinear 빠른 추정 포함)
+ 전체 OOF 캐시 기반.
"""
import os, time, warnings
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from scipy.optimize import minimize
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
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
views = {}
specs = {
    "word12":  TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=2, sublinear_tf=True, strip_accents="unicode", max_features=120000),
    "word123": TfidfVectorizer(analyzer="word", ngram_range=(1,3), min_df=2, sublinear_tf=True, strip_accents="unicode", max_features=160000),
    "char26":  TfidfVectorizer(analyzer="char_wb", ngram_range=(2,6), min_df=3, sublinear_tf=True, max_features=160000),
    "char35":  TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=3, sublinear_tf=True, max_features=120000),
}
for name, vec in specs.items():
    views[name] = (vec.fit_transform(train["text"]), vec.transform(test["text"]))

combo_tr = hstack([views["word123"][0], views["char26"][0]]).tocsr()
combo_te = hstack([views["word123"][1], views["char26"][1]]).tocsr()
views["combo_wc2"] = (combo_tr, combo_te)
combo3_tr = hstack([views["word123"][0], views["char26"][0], views["char35"][0]]).tocsr()
combo3_te = hstack([views["word123"][1], views["char26"][1], views["char35"][1]]).tocsr()
views["combo3"] = (combo3_tr, combo3_te)
print(f"views ready ({time.time()-t0:.0f}s)")

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

def run(name, maker, view):
    Xtr, Xte = views[view]
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
    return ll, oof, tp

def LR(C): return lambda: LogisticRegression(C=C, solver="saga", multi_class="multinomial", max_iter=600, n_jobs=4, random_state=RANDOM_STATE)
def LRlib(C): return lambda: LogisticRegression(C=C, solver="liblinear", multi_class="ovr", max_iter=1000, random_state=RANDOM_STATE)
def MNB(a): return lambda: MultinomialNB(alpha=a)

jobs = [
    ("lr_cwc2_C15",  LR(15), "combo_wc2"),
    ("lr_cwc2_C25",  LR(25), "combo_wc2"),
    ("lrlib_cwc2_C4", LRlib(4), "combo_wc2"),
    ("lr_c3_C15",    LR(15), "combo3"),
    ("mnb_w12_a03",  MNB(0.03), "word12"),
    ("mnb_w12_a07",  MNB(0.07), "word12"),
]

oofs, testps, lls = {}, {}, {}
for name, maker, view in jobs:
    ll, oof, tp = run(name, maker, view)
    lls[name] = ll; oofs[name] = oof; testps[name] = tp

# v3 캐시 병합 (v3 안에 v2 포함됨)
cache = os.path.join(BASE, "solution", "oof_v3.npz")
z = np.load(cache)
for k in z.files:
    if k.startswith("oof_"):
        n = k[4:]
        if n not in oofs:
            oofs[n] = z[k]
            lls[n] = log_loss(y, z[k], labels=[0,1,2])
    elif k.startswith("test_"):
        n = k[5:]
        if n not in testps:
            testps[n] = z[k]
print(f"total models: {len(oofs)}")

# ---------------- nested-CV ensemble evaluation ----------------
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
print("\ntop:", [(n, round(lls[n],4)) for n in ranked[:10]])
TOP = ranked[:12]
names = TOP
M = len(names)
O_all = np.stack([oofs[n] for n in names])
P_all = np.stack([testps[n] for n in names])

# outer CV: 앙상블 일반화 성능 추정
outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=777)
oof_ens = np.zeros((len(y), 3))
oof_ens_temp = np.zeros((len(y), 3))
oof_simple = np.zeros((len(y), 3))
for tr_idx, va_idx in outer.split(np.zeros(len(y)), y):
    O_tr, O_va = O_all[:, tr_idx, :], O_all[:, va_idx, :]
    w = fit_slsqp(O_tr, y[tr_idx])
    pv = np.tensordot(w, O_va, axes=1)
    pv = np.clip(pv, 1e-9, 1.0); pv /= pv.sum(1, keepdims=True)
    oof_ens[va_idx] = pv
    # 온도도 inner 학습
    lp_tr = np.log(np.clip(np.tensordot(w, O_tr, axes=1), 1e-12, 1.0))
    T = fit_temp(lp_tr, y[tr_idx])
    q = np.log(np.clip(pv, 1e-12, 1.0)) / T
    q -= q.max(1, keepdims=True)
    p2 = np.exp(q); p2 /= p2.sum(1, keepdims=True)
    oof_ens_temp[va_idx] = p2
    # 단순: 상위2개 평균
    ps = (O_va[0] + O_va[1]) / 2
    oof_simple[va_idx] = ps

ll_ens = log_loss(y, oof_ens, labels=[0,1,2])
ll_ens_t = log_loss(y, oof_ens_temp, labels=[0,1,2])
ll_simple = log_loss(y, oof_simple, labels=[0,1,2])
print(f"\nnested-CV honest estimates:")
print(f"  slsqp-ens:        {ll_ens:.5f}")
print(f"  slsqp-ens+temp:   {ll_ens_t:.5f}")
print(f"  top2-simple-avg:  {ll_simple:.5f}")
print(f"  best single OOF:  {lls[ranked[0]]:.5f}")

# ---------------- 최종: 전체 OOF로 w, T 추정 후 test 예측 ----------------
w_full = fit_slsqp(O_all, y)
pf = np.tensordot(w_full, O_all, axes=1)
pf = np.clip(pf, 1e-9, 1.0); pf /= pf.sum(1, keepdims=True)
ll_full = log_loss(y, pf, labels=[0,1,2])
print(f"\nfull-fit slsqp OOF: {ll_full:.5f}")
for n, w in sorted(zip(names, w_full), key=lambda x: -x[1]):
    if w > 0.02: print(f"  {n}: {w:.3f}")

use_temp = ll_ens_t < ll_ens
print(f"use temperature: {use_temp}")
final = np.tensordot(w_full, P_all, axes=1)
final = np.clip(final, 1e-9, 1.0); final /= final.sum(1, keepdims=True)
T_full = None
if use_temp:
    T_full = fit_temp(np.log(np.clip(pf, 1e-12, 1.0)), y)
    q = np.log(np.clip(final, 1e-12, 1.0)) / T_full
    q -= q.max(1, keepdims=True)
    final = np.exp(q); final /= final.sum(1, keepdims=True)
    print(f"T_full={T_full:.3f}")
final = np.clip(final, 1e-6, 1.0); final /= final.sum(1, keepdims=True)

sub = pd.DataFrame({"id": test["id"], "EAP": final[:,0], "HPL": final[:,1], "MWS": final[:,2]})
assert sub["id"].is_unique and set(sub["id"]) == set(test["id"])
sub.to_csv(os.path.join(BASE, "outputs", "submission.csv"), index=False)

np.savez(os.path.join(BASE, "solution", "oof_v4.npz"),
         **{f"oof_{k}": v for k, v in oofs.items()},
         **{f"test_{k}": v for k, v in testps.items()})
import json
json.dump({"lls": lls, "TOP": names, "w_full": w_full.tolist(),
           "ll_full": ll_full, "nested": {"ens": ll_ens, "ens_temp": ll_ens_t, "simple": ll_simple},
           "T_full": T_full},
          open(os.path.join(BASE, "solution", "oof_v4.json"), "w"), indent=2)
print(f"saved ({time.time()-t0:.0f}s total)")

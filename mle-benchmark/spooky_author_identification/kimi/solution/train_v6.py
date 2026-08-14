# -*- coding: utf-8 -*-
"""
v6: raw count + BM25-like 피처 NB 추가 (고전 spooky 강력 조합),
    중첩CV로 SLSQP vs 간단 혼합 비교, 전체 seed bagging 확대.
"""
import os, time, warnings
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from scipy.optimize import minimize
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder, normalize

RANDOM_STATE = 42
N_SPLITS = 5
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES = ["EAP", "HPL", "MWS"]

train = pd.read_csv(os.path.join(BASE, "train.csv"))
test = pd.read_csv(os.path.join(BASE, "test.csv"))
y = LabelEncoder().fit(CLASSES).transform(train["author"])
t0 = time.time()

z = np.load(os.path.join(BASE, "solution", "oof_v5.npz"))
oofs, testps, lls = {}, {}, {}
for k in z.files:
    if k.startswith("oof_"):
        oofs[k[4:]] = z[k]; lls[k[4:]] = log_loss(y, z[k], labels=[0,1,2])
    elif k.startswith("test_"):
        testps[k[5:]] = z[k]
print(f"loaded {len(oofs)} cached models")

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

def run(name, maker, Xtr, Xte):
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

# --- raw count word 1-2 ---
cv12 = CountVectorizer(analyzer="word", ngram_range=(1,2), min_df=2, strip_accents="unicode", max_features=120000)
R12tr, R12te = cv12.fit_transform(train["text"]), cv12.transform(test["text"])

# --- raw count char 2-6 ---
cv26 = CountVectorizer(analyzer="char_wb", ngram_range=(2,6), min_df=3, max_features=160000)
RC26tr, RC26te = cv26.fit_transform(train["text"]), cv26.transform(test["text"])

# --- BM25-like transform: tf=log(1+tf), idf=smooth, l2 norm ---
def bm25(X):
    Xf = X.astype(np.float64).copy()
    Xf.data = np.log1p(Xf.data)
    df = np.asarray((X > 0).sum(axis=0)).ravel()
    N = X.shape[0]
    idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0)
    Xf = Xf.multiply(idf).tocsr()
    return normalize(Xf, norm="l2")

B12tr, B12te = bm25(R12tr), bm25(R12te)
BC26tr, BC26te = bm25(RC26tr), bm25(RC26te)

def MNB(a): return lambda: MultinomialNB(alpha=a)
def LR(C): return lambda: LogisticRegression(C=C, solver="saga", multi_class="multinomial", max_iter=600, n_jobs=4, random_state=RANDOM_STATE)

# raw counts + NB
run("mnb_raw_w12_a05", MNB(0.05), R12tr, R12te)
run("mnb_raw_w12_a20", MNB(0.2), R12tr, R12te)
run("mnb_raw_c26_a30", MNB(0.3), RC26tr, RC26te)

# bm25 + NB (l2 정규화돼 음수 아님)
run("mnb_bm25_w12_a05", MNB(0.05), B12tr, B12te)
run("mnb_bm25_w12_a20", MNB(0.2), B12tr, B12te)
run("mnb_bm25_c26_a30", MNB(0.3), BC26tr, BC26te)

# bm25 + LR (sparse, high C)
run("lr_bm25_combo_C15", LR(15), hstack([B12tr, BC26tr]).tocsr(), hstack([B12te, BC26te]).tocsr())

print(f"\ntotal models: {len(oofs)}")

# ---------------- nested-CV 비교: SLSQP vs 단순 2-모델 혼합 ----------------
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
print("\ntop:", [(n, round(lls[n],4)) for n in ranked[:14]])
TOP = ranked[:14]
names = TOP
O_all = np.stack([oofs[n] for n in names])
P_all = np.stack([testps[n] for n in names])

outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=777)
evals = {"slsqp": np.zeros((len(y),3)), "slsqp_t": np.zeros((len(y),3)),
         "simple2": np.zeros((len(y),3)), "simple2_t": np.zeros((len(y),3)),
         "simple3": np.zeros((len(y),3))}
for tr_idx, va_idx in outer.split(np.zeros(len(y)), y):
    Otr, Ova = O_all[:, tr_idx, :], O_all[:, va_idx, :]
    ytr = y[tr_idx]
    w = fit_slsqp(Otr, ytr)
    pv = np.tensordot(w, Ova, axes=1); pv = np.clip(pv,1e-9,1); pv /= pv.sum(1,keepdims=True)
    evals["slsqp"][va_idx] = pv
    T = fit_temp(np.log(np.clip(np.tensordot(w,Otr,axes=1),1e-12,1)), ytr)
    q = np.log(np.clip(pv,1e-12,1))/T; q -= q.max(1,keepdims=True)
    p2 = np.exp(q); p2/=p2.sum(1,keepdims=True); evals["slsqp_t"][va_idx]=p2
    # simple2: 상위2 모델 평균
    ps = (Ova[0]+Ova[1])/2; evals["simple2"][va_idx]=ps
    ptr = (Otr[0]+Otr[1])/2
    T2 = fit_temp(np.log(np.clip(ptr,1e-12,1)), ytr)
    q = np.log(np.clip(ps,1e-12,1))/T2; q-=q.max(1,keepdims=True)
    p3 = np.exp(q); p3/=p3.sum(1,keepdims=True); evals["simple2_t"][va_idx]=p3
    evals["simple3"][va_idx]=(Ova[0]+Ova[1]+Ova[2])/3

for k, v in evals.items():
    print(f"  nested {k}: {log_loss(y, v, labels=[0,1,2]):.5f}")

best_method = min(evals, key=lambda k: log_loss(y, evals[k], labels=[0,1,2]))
print(f"best method: {best_method}")

# 최종 test 예측 (best method와 동일 절차, 전체 데이터로 피팅)
w_full = fit_slsqp(O_all, y)
pf = np.tensordot(w_full, O_all, axes=1); pf=np.clip(pf,1e-9,1); pf/=pf.sum(1,keepdims=True)
print(f"full slsqp OOF: {log_loss(y,pf,labels=[0,1,2]):.5f}")
for n, w in sorted(zip(names, w_full), key=lambda x: -x[1]):
    if w > 0.02: print(f"  {n}: {w:.3f}")

if best_method == "slsqp":
    final = np.tensordot(w_full, P_all, axes=1)
elif best_method == "slsqp_t":
    final = np.tensordot(w_full, P_all, axes=1)
    final = np.clip(final,1e-9,1); final/=final.sum(1,keepdims=True)
    T = fit_temp(np.log(np.clip(pf,1e-12,1)), y)
    q = np.log(np.clip(final,1e-12,1))/T; q-=q.max(1,keepdims=True)
    final = np.exp(q); final/=final.sum(1,keepdims=True)
    print(f"T={T:.3f}")
elif best_method == "simple2":
    final = (P_all[0]+P_all[1])/2
elif best_method == "simple2_t":
    final = (P_all[0]+P_all[1])/2
    T = fit_temp(np.log(np.clip((O_all[0]+O_all[1])/2,1e-12,1)), y)
    q = np.log(np.clip(final,1e-12,1))/T; q-=q.max(1,keepdims=True)
    final = np.exp(q); final/=final.sum(1,keepdims=True)
    print(f"T={T:.3f}")
else:
    final = (P_all[0]+P_all[1]+P_all[2])/3

final = np.clip(final,1e-6,1); final/=final.sum(1,keepdims=True)
sub = pd.DataFrame({"id": test["id"], "EAP": final[:,0], "HPL": final[:,1], "MWS": final[:,2]})
assert sub["id"].is_unique and set(sub["id"]) == set(test["id"])
sub.to_csv(os.path.join(BASE, "outputs", "submission.csv"), index=False)

np.savez(os.path.join(BASE, "solution", "oof_v6.npz"),
         **{f"oof_{k}": v for k, v in oofs.items()},
         **{f"test_{k}": v for k, v in testps.items()})
import json
json.dump({"lls": lls, "TOP": names, "w_full": w_full.tolist(), "best_method": best_method},
          open(os.path.join(BASE, "solution", "oof_v6.json"), "w"), indent=2)
print(f"saved ({time.time()-t0:.0f}s total)")

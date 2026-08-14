# -*- coding: utf-8 -*-
"""
v7: 스태킹 메타 러너(Ridge on logits) + SLSQP(전수 44모델/상위N) + 스태킹LR 비교.
중첩CV로 공정 평가 후 최적 방법으로 최종 제출.
"""
import os, time, warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.linear_model import Ridge, LogisticRegression

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES = ["EAP", "HPL", "MWS"]
train = pd.read_csv(os.path.join(BASE, "train.csv"))
test = pd.read_csv(os.path.join(BASE, "test.csv"))
y = pd.Categorical(train["author"], categories=CLASSES).codes

t0 = time.time()
z = np.load(os.path.join(BASE, "solution", "oof_v6.npz"))
oofs, testps, lls = {}, {}, {}
for k in z.files:
    if k.startswith("oof_"):
        oofs[k[4:]] = z[k]; lls[k[4:]] = log_loss(y, z[k], labels=[0,1,2])
    elif k.startswith("test_"):
        testps[k[5:]] = z[k]
print(f"loaded {len(oofs)} models")

# 쓸모없는 모델 제외 (OOF > 0.55)
good = [n for n in lls if lls[n] < 0.55]
ranked = sorted(good, key=lambda n: lls[n])
print("ranking:", [(n, round(lls[n],4)) for n in ranked])

def to_logits(P):
    return np.log(np.clip(P, 1e-9, 1.0))

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

def fit_ridge(Ltr, ytr, alpha):
    """Ltr: (N, M*3) logits. 각 클래스에 ridge 회귀 후 softmax."""
    Ws = []
    for c in range(3):
        r = Ridge(alpha=alpha)
        r.fit(Ltr, (ytr == c).astype(float))
        Ws.append((r.coef_.copy(), r.intercept_))
    return Ws

def pred_ridge(Ws, L):
    S = np.column_stack([L @ w + b for w, b in Ws])
    S -= S.max(1, keepdims=True)
    P = np.exp(S); P /= P.sum(1, keepdims=True)
    return P

def fit_stacklr(Ltr, ytr, C):
    lr = LogisticRegression(C=C, solver="lbfgs", multi_class="multinomial", max_iter=2000)
    lr.fit(Ltr, ytr)
    return lr

# ----- 중첩 CV 비교 -----
outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=777)
results = {}

def nested_eval(builder, name):
    oof = np.zeros((len(y), 3))
    for tr_idx, va_idx in outer.split(np.zeros(len(y)), y):
        oof[va_idx] = builder(tr_idx, va_idx)
    ll = log_loss(y, oof, labels=[0,1,2])
    results[name] = ll
    print(f"  nested {name}: {ll:.5f} ({time.time()-t0:.0f}s)")

# (a) SLSQP top14 + temp
names14 = ranked[:14]
O14 = np.stack([oofs[n] for n in names14])
def b_slsqp_t(tr, va):
    w = fit_slsqp(O14[:, tr, :], y[tr])
    pv = np.tensordot(w, O14[:, va, :], axes=1); pv=np.clip(pv,1e-9,1); pv/=pv.sum(1,keepdims=True)
    T = fit_temp(np.log(np.clip(np.tensordot(w, O14[:, tr, :], axes=1),1e-12,1)), y[tr])
    q = np.log(np.clip(pv,1e-12,1))/T; q-=q.max(1,keepdims=True)
    p = np.exp(q); p/=p.sum(1,keepdims=True)
    return p
nested_eval(b_slsqp_t, "slsqp14_t")

# (b) SLSQP all good + temp
namesAll = ranked
OA = np.stack([oofs[n] for n in namesAll])
def b_slsqp_all_t(tr, va):
    w = fit_slsqp(OA[:, tr, :], y[tr], restarts=3)
    pv = np.tensordot(w, OA[:, va, :], axes=1); pv=np.clip(pv,1e-9,1); pv/=pv.sum(1,keepdims=True)
    T = fit_temp(np.log(np.clip(np.tensordot(w, OA[:, tr, :], axes=1),1e-12,1)), y[tr])
    q = np.log(np.clip(pv,1e-12,1))/T; q-=q.max(1,keepdims=True)
    p = np.exp(q); p/=p.sum(1,keepdims=True)
    return p
nested_eval(b_slsqp_all_t, "slsqpAll_t")

# (c) Ridge stacking on logits (top14)
def make_L(names):
    mats = []
    for n in names:
        mats.append(to_logits(oofs[n]))
    return np.concatenate(mats, axis=1)  # (N, M*3)
L14 = make_L(names14)
for alpha in [10, 100, 1000]:
    def b_ridge(tr, va, alpha=alpha):
        Ws = fit_ridge(L14[tr], y[tr], alpha)
        return pred_ridge(Ws, L14[va])
    nested_eval(b_ridge, f"ridge14_a{alpha}")

# (d) LR stacking on logits (top14)
for C in [0.1, 1.0]:
    def b_sl(tr, va, C=C):
        m = fit_stacklr(L14[tr], y[tr], C)
        return m.predict_proba(L14[va])
    nested_eval(b_sl, f"stacklr14_C{C}")

# (e) Ridge stacking all-good
LA = make_L(namesAll)
def b_ridge_all(tr, va):
    Ws = fit_ridge(LA[tr], y[tr], 100)
    return pred_ridge(Ws, LA[va])
nested_eval(b_ridge_all, "ridgeAll_a100")

best = min(results, key=results.get)
print(f"\nbest method: {best} ({results[best]:.5f})")

# ----- 최종 test 예측 -----
def test_logits(names):
    return np.concatenate([to_logits(testps[n]) for n in names], axis=1)

if best == "slsqp14_t":
    w = fit_slsqp(O14, y)
    pf = np.tensordot(w, O14, axes=1); pf=np.clip(pf,1e-9,1); pf/=pf.sum(1,keepdims=True)
    T = fit_temp(np.log(np.clip(pf,1e-12,1)), y)
    P14 = np.stack([testps[n] for n in names14])
    final = np.tensordot(w, P14, axes=1); final=np.clip(final,1e-9,1); final/=final.sum(1,keepdims=True)
    q = np.log(np.clip(final,1e-12,1))/T; q-=q.max(1,keepdims=True)
    final = np.exp(q); final/=final.sum(1,keepdims=True)
    print(f"slsqp14_t full: ll={log_loss(y,pf,labels=[0,1,2]):.5f} T={T:.3f}")
    for n, wi in sorted(zip(names14, w), key=lambda x:-x[1]):
        if wi > 0.02: print(f"  {n}: {wi:.3f}")
elif best == "slsqpAll_t":
    w = fit_slsqp(OA, y, restarts=3)
    pf = np.tensordot(w, OA, axes=1); pf=np.clip(pf,1e-9,1); pf/=pf.sum(1,keepdims=True)
    T = fit_temp(np.log(np.clip(pf,1e-12,1)), y)
    PA = np.stack([testps[n] for n in namesAll])
    final = np.tensordot(w, PA, axes=1); final=np.clip(final,1e-9,1); final/=final.sum(1,keepdims=True)
    q = np.log(np.clip(final,1e-12,1))/T; q-=q.max(1,keepdims=True)
    final = np.exp(q); final/=final.sum(1,keepdims=True)
    print(f"slsqpAll_t full: ll={log_loss(y,pf,labels=[0,1,2]):.5f} T={T:.3f}")
elif best.startswith("ridge14"):
    alpha = float(best.split("a")[1])
    Ws = fit_ridge(L14, y, alpha)
    final = pred_ridge(Ws, test_logits(names14))
elif best.startswith("stacklr14"):
    C = float(best.split("C")[1])
    m = fit_stacklr(L14, y, C)
    final = m.predict_proba(test_logits(names14))
else:  # ridgeAll
    Ws = fit_ridge(LA, y, 100)
    final = pred_ridge(Ws, test_logits(namesAll))

final = np.clip(final, 1e-6, 1.0); final /= final.sum(1, keepdims=True)
sub = pd.DataFrame({"id": test["id"], "EAP": final[:,0], "HPL": final[:,1], "MWS": final[:,2]})
assert sub["id"].is_unique and set(sub["id"]) == set(test["id"])
sub.to_csv(os.path.join(BASE, "outputs", "submission.csv"), index=False)
import json
json.dump({"nested": results, "best": best}, open(os.path.join(BASE,"solution","oof_v7.json"),"w"), indent=2)
print(f"saved ({time.time()-t0:.0f}s total)")

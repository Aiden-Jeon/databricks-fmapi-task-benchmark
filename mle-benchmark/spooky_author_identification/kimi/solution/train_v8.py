# -*- coding: utf-8 -*-
"""
v8: 스태킹 LR 메타러너 미세조정 (모델 수/C/피처 그룹) + stacklr vs slsqp 혼합.
중첩CV로만 판단.
"""
import os, time, warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression

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

good = [n for n in lls if lls[n] < 0.55]
ranked = sorted(good, key=lambda n: lls[n])
print(f"{len(good)} usable models")

def to_logits(P, eps=1e-9):
    return np.log(np.clip(P, eps, 1.0))

def make_L(names, src):
    return np.concatenate([to_logits(src[n]) for n in names], axis=1)

outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=777)
results = {}

def nested_eval(builder, name):
    oof = np.zeros((len(y), 3))
    for tr_idx, va_idx in outer.split(np.zeros(len(y)), y):
        oof[va_idx] = builder(tr_idx, va_idx)
    ll = log_loss(y, oof, labels=[0,1,2])
    results[name] = ll
    print(f"  nested {name}: {ll:.5f} ({time.time()-t0:.0f}s)")

def fit_stacklr(Ltr, ytr, C, cw=None):
    lr = LogisticRegression(C=C, solver="lbfgs", multi_class="multinomial", max_iter=3000, class_weight=cw)
    lr.fit(Ltr, ytr)
    return lr

# 모델 수 변화 + C 변화
for K in [6, 10, 14, 20, 30]:
    names = ranked[:K]
    L = make_L(names, oofs)
    for C in ([0.5, 1.0, 2.0] if K in (10, 14, 20) else [1.0]):
        def b(tr, va, L=L, C=C):
            m = fit_stacklr(L[tr], y[tr], C)
            return m.predict_proba(L[va])
        nested_eval(b, f"stk_K{K}_C{C}")

# 확률 자체 concat (log 대신)
def make_P(names, src):
    return np.concatenate([src[n] for n in names], axis=1)
names14 = ranked[:14]
P14 = make_P(names14, oofs)
def bP(tr, va):
    m = fit_stacklr(P14[tr], y[tr], 1.0)
    return m.predict_proba(P14[va])
nested_eval(bP, "stk_prob_K14_C1")

# saga (L1) - 희소 가중치
def b_l1(tr, va):
    lr = LogisticRegression(C=0.5, penalty="l1", solver="saga", multi_class="multinomial", max_iter=3000, n_jobs=4)
    lr.fit(P14[tr], y[tr])
    return lr.predict_proba(P14[va])
nested_eval(b_l1, "stkL1_prob_K14_C0.5")

best = min(results, key=results.get)
print(f"\nbest: {best} ({results[best]:.5f})")

# 최종
def parse(best):
    if best.startswith("stk_K"):
        K = int(best.split("_")[1][1:]); C = float(best.split("_")[2][1:])
        return ("logit", K, C)
    raise ValueError(best)

kind, K, C = parse(best)
names = ranked[:K]
Ltr = make_L(names, oofs); Lte = make_L(names, testps)
m = fit_stacklr(Ltr, y, C)
final = m.predict_proba(Lte)
# 클래스 순서 확인
order = [list(m.classes_).index(c) for c in range(3)]
final = final[:, order]

final = np.clip(final, 1e-6, 1.0); final /= final.sum(1, keepdims=True)
sub = pd.DataFrame({"id": test["id"], "EAP": final[:,0], "HPL": final[:,1], "MWS": final[:,2]})
assert sub["id"].is_unique and set(sub["id"]) == set(test["id"])
sub.to_csv(os.path.join(BASE, "outputs", "submission.csv"), index=False)
import json
json.dump({"nested": results, "best": best, "names": names},
          open(os.path.join(BASE,"solution","oof_v8.json"),"w"), indent=2)
print(f"saved ({time.time()-t0:.0f}s)")

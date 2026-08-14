# -*- coding: utf-8 -*-
"""
v9: 스태킹 K 확대 (전체), C 탐색, 외부 seed 안정성 체크.
"""
import os, time, warnings
import numpy as np
import pandas as pd
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
        oofs[k[4:]] = z[k]
        lls[k[4:]] = log_loss(y, z[k], labels=[0, 1, 2])
    elif k.startswith("test_"):
        testps[k[5:]] = z[k]

good = [n for n in lls if lls[n] < 0.55]
ranked = sorted(good, key=lambda n: lls[n])

def to_logits(P, eps=1e-9):
    return np.log(np.clip(P, eps, 1.0))

def make_L(names, src):
    return np.concatenate([to_logits(src[n]) for n in names], axis=1)

def fit_stacklr(Ltr, ytr, C):
    lr = LogisticRegression(C=C, solver="lbfgs", multi_class="multinomial", max_iter=3000)
    lr.fit(Ltr, ytr)
    return lr

def nested_eval_seed(builder, seed):
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros((len(y), 3))
    for tr_idx, va_idx in outer.split(np.zeros(len(y)), y):
        oof[va_idx] = builder(tr_idx, va_idx)
    return log_loss(y, oof, labels=[0, 1, 2])

namesAll = ranked
LA = make_L(namesAll, oofs)

# C 탐색 (seed 777)
for C in [0.5, 1.0, 2.0, 4.0]:
    def b(tr, va, C=C):
        m = fit_stacklr(LA[tr], y[tr], C)
        return m.predict_proba(LA[va])
    ll = nested_eval_seed(b, 777)
    print(f"  stkAll_C{C}: {ll:.5f} ({time.time()-t0:.0f}s)")

# seed 안정성 (C=1)
lls_seeds = []
for s in [777, 123, 2024]:
    def b(tr, va):
        m = fit_stacklr(LA[tr], y[tr], 1.0)
        return m.predict_proba(LA[va])
    ll = nested_eval_seed(b, s)
    lls_seeds.append(ll)
    print(f"  seed{s}: {ll:.5f}")
print(f"  mean={np.mean(lls_seeds):.5f} std={np.std(lls_seeds):.5f}")

# 최종: 전체 데이터 학습 (C=1)
m = fit_stacklr(LA, y, 1.0)
LAte = make_L(namesAll, testps)
final = m.predict_proba(LAte)
order = [list(m.classes_).index(c) for c in range(3)]
final = final[:, order]

final = np.clip(final, 1e-6, 1.0)
final /= final.sum(1, keepdims=True)
sub = pd.DataFrame({"id": test["id"], "EAP": final[:, 0], "HPL": final[:, 1], "MWS": final[:, 2]})
assert sub["id"].is_unique and set(sub["id"]) == set(test["id"])
sub.to_csv(os.path.join(BASE, "outputs", "submission.csv"), index=False)
import json
json.dump({"seed_stability": lls_seeds}, open(os.path.join(BASE, "solution", "oof_v9.json"), "w"))
print(f"saved ({time.time()-t0:.0f}s)")

# -*- coding: utf-8 -*-
"""
v3: 하이퍼파라미터 튜닝 + 추가 모델 + SLSQP 가중치 최적화 + 온도 스케일링.
v2의 OOF/테스트 예측을 캐시에서 불러와 함께 사용.
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
    "word13b": TfidfVectorizer(analyzer="word", ngram_range=(1,3), min_df=2, binary=True, strip_accents="unicode", max_features=120000),
    "char35":  TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=3, sublinear_tf=True, max_features=120000),
    "char26":  TfidfVectorizer(analyzer="char_wb", ngram_range=(2,6), min_df=3, sublinear_tf=True, max_features=160000),
    "char48":  TfidfVectorizer(analyzer="char_wb", ngram_range=(4,8), min_df=5, sublinear_tf=True, max_features=120000),
}
for name, vec in specs.items():
    views[name] = (vec.fit_transform(train["text"]), vec.transform(test["text"]))
    print(f"  {name}: {views[name][0].shape}")

# 메타 피처 (문장 통계)
def meta_feats(texts):
    arr = []
    for t in texts:
        words = t.split()
        n = len(t); nw = len(words)
        awl = np.mean([len(w) for w in words]) if words else 0
        punc = sum(ch in ".,;:!?-'\"“”()…" for ch in t) / max(n,1)
        digit = sum(ch.isdigit() for ch in t) / max(n,1)
        upper = sum(ch.isupper() for ch in t) / max(n,1)
        arr.append([n, nw, awl, punc, digit, upper, n/max(nw,1)])
    return np.asarray(arr, dtype=np.float64)

from sklearn.preprocessing import StandardScaler
msc = StandardScaler()
Mtr = msc.fit_transform(meta_feats(train["text"]))
Mte = msc.transform(meta_feats(test["text"]))

combo_wc2_tr = hstack([views["word123"][0], views["char26"][0]]).tocsr()
combo_wc2_te = hstack([views["word123"][1], views["char26"][1]]).tocsr()
views["combo_wc2"] = (combo_wc2_tr, combo_wc2_te)
combo_meta_tr = hstack([combo_wc2_tr, Mtr * 2.0]).tocsr()
combo_meta_te = hstack([combo_wc2_te, Mte * 2.0]).tocsr()
views["combo_meta"] = (combo_meta_tr, combo_meta_te)
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

candidates = {}

def LR(C): return lambda: LogisticRegression(C=C, solver="saga", multi_class="multinomial", max_iter=500, n_jobs=4, random_state=RANDOM_STATE)
def MNB(a): return lambda: MultinomialNB(alpha=a)

jobs = [
    ("mnb_w12_a02",  MNB(0.02), "word12"),
    ("mnb_w12_a05",  MNB(0.05), "word12"),
    ("mnb_w12_a10",  MNB(0.10), "word12"),
    ("mnb_w123_a10", MNB(0.10), "word123"),
    ("mnb_w13b_a20", MNB(0.20), "word13b"),
    ("mnb_c26_a30",  MNB(0.30), "char26"),
    ("mnb_c48_a10",  MNB(0.10), "char48"),
    ("lr_cwc2_C6",   LR(6),  "combo_wc2"),
    ("lr_cwc2_C10",  LR(10), "combo_wc2"),
    ("lr_cmeta_C6",  LR(6),  "combo_meta"),
    ("lr_w123_C8",   LR(8),  "word123"),
    ("lr_c48_C4",    LR(4),  "char48"),
]

oofs, testps, lls = {}, {}, {}
for name, maker, view in jobs:
    ll, oof, tp = run(name, maker, view)
    lls[name] = ll; oofs[name] = oof; testps[name] = tp

# v2 캐시 병합
cache = os.path.join(BASE, "solution", "oof_v2.npz")
if os.path.exists(cache):
    z = np.load(cache)
    for k in z.files:
        if k.startswith("oof_"):
            n = "v2_" + k[4:]
            oofs[n] = z[k]
            lls[n] = log_loss(y, z[k], labels=[0,1,2])
        elif k.startswith("test_"):
            testps["v2_" + k[5:]] = z[k]
    print(f"merged {len([k for k in z.files if k.startswith('oof_')])} v2 models")

# ---- SLSQP 가중치 최적화 (상위 모델만 사용해 과적합 억제) ----
ranked = sorted(lls, key=lls.get)
print("\nranking:", [(n, round(lls[n],4)) for n in ranked[:8]])
TOP = ranked[:14]
O = np.stack([oofs[n] for n in ranked[:len(TOP)]])  # (M, N, 3)
M = O.shape[0]

def ens_ll(w):
    p = np.tensordot(w, O, axes=1)
    p = np.clip(p, 1e-9, 1.0); p /= p.sum(1, keepdims=True)
    return log_loss(y, p, labels=[0,1,2])

best_w, best_ll = None, 1e9
inits = []
inits.append(np.eye(M)[0])                      # best single
inits.append(np.ones(M)/M)                      # uniform
greedy = np.zeros(M); greedy[0]=0.5; greedy[1]=0.5
inits.append(greedy)
rng = np.random.default_rng(0)
for _ in range(6):
    d = rng.dirichlet(np.ones(M)); inits.append(d)

cons = ({"type": "eq", "fun": lambda w: w.sum() - 1})
bnds = [(0, 1)] * M
for w0 in inits:
    r = minimize(ens_ll, w0, method="SLSQP", bounds=bnds, constraints=cons,
                 options={"maxiter": 500, "ftol": 1e-10})
    if r.fun < best_ll:
        best_ll, best_w = r.fun, r.x
print(f"\nSLSQP ensemble OOF: {best_ll:.5f}")
for n, w in sorted(zip(TOP, best_w), key=lambda x: -x[1]):
    if w > 0.01: print(f"  {n}: {w:.3f}")

# ---- 온도 스케일링 (1-param, log(p)/T 후 softmax) ----
oof_ens = np.tensordot(best_w, O, axes=1)
logp = np.log(np.clip(oof_ens, 1e-12, 1.0))
def temp_ll(T):
    q = logp / T
    q -= q.max(1, keepdims=True)
    p = np.exp(q); p /= p.sum(1, keepdims=True)
    return log_loss(y, p, labels=[0,1,2])
rt = minimize(temp_ll, [1.0], method="Nelder-Mead", options={"xatol":1e-4, "fatol":1e-7})
T = float(rt.x[0])
print(f"temperature T={T:.3f}  ll {best_ll:.5f} -> {rt.fun:.5f}")

# ---- 최종 test 예측 ----
Tstack = np.stack([testps[n] for n in TOP])
final = np.tensordot(best_w, Tstack, axes=1)
lf = np.log(np.clip(final, 1e-12, 1.0)) / T
lf -= lf.max(1, keepdims=True)
final = np.exp(lf); final /= final.sum(1, keepdims=True)
final = np.clip(final, 1e-6, 1.0); final /= final.sum(1, keepdims=True)

sub = pd.DataFrame({"id": test["id"], "EAP": final[:,0], "HPL": final[:,1], "MWS": final[:,2]})
assert sub["id"].is_unique and set(sub["id"]) == set(test["id"])
sub.to_csv(os.path.join(BASE, "outputs", "submission.csv"), index=False)

np.savez(os.path.join(BASE, "solution", "oof_v3.npz"),
         **{f"oof_{k}": v for k, v in oofs.items()},
         **{f"test_{k}": v for k, v in testps.items()})
import json
json.dump({"lls": lls, "TOP": TOP, "weights": best_w.tolist(), "ens_ll": best_ll, "T": T, "ens_ll_temp": float(rt.fun)},
          open(os.path.join(BASE, "solution", "oof_v3.json"), "w"), indent=2)
print(f"saved submission ({time.time()-t0:.0f}s total)")

# -*- coding: utf-8 -*-
"""
v2: 피처/모델 확장 + OOF log loss로 greedy 앙상블 가중치 탐색.
목표: OOF multi-class log loss 최소화.
"""
import os, time, itertools, warnings
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
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
print(f"train {train.shape}, test {test.shape}")

t0 = time.time()
print("vectorizing...")
views = {}
specs = {
    "word12":  TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=2, sublinear_tf=True, strip_accents="unicode", max_features=120000),
    "word123": TfidfVectorizer(analyzer="word", ngram_range=(1,3), min_df=2, sublinear_tf=True, strip_accents="unicode", max_features=160000),
    "char35":  TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=3, sublinear_tf=True, max_features=120000),
    "char26":  TfidfVectorizer(analyzer="char_wb", ngram_range=(2,6), min_df=3, sublinear_tf=True, max_features=160000),
}
for name, vec in specs.items():
    Xtr = vec.fit_transform(train["text"])
    Xte = vec.transform(test["text"])
    views[name] = (Xtr, Xte)
    print(f"  {name}: {Xtr.shape}")

combo_tr = hstack([views["word12"][0], views["char35"][0]]).tocsr()
combo_te = hstack([views["word12"][1], views["char35"][1]]).tocsr()
views["combo_wc"] = (combo_tr, combo_te)
combo2_tr = hstack([views["word123"][0], views["char26"][0]]).tocsr()
combo2_te = hstack([views["word123"][1], views["char26"][1]]).tocsr()
views["combo_wc2"] = (combo2_tr, combo2_te)
print(f"vectorizing done ({time.time()-t0:.1f}s)")

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

def oof_predict(make_model, X, name):
    oof = np.zeros((X.shape[0], 3))
    for tr_idx, va_idx in skf.split(X, y):
        m = make_model()
        m.fit(X[tr_idx], y[tr_idx])
        p = m.predict_proba(X[va_idx])
        order = [list(m.classes_).index(c) for c in range(3)]
        oof[va_idx] = p[:, order]
    ll = log_loss(y, oof, labels=[0,1,2])
    print(f"  {name}: {ll:.5f}")
    return ll, oof

def full_predict(make_model, Xtr, Xte):
    m = make_model()
    m.fit(Xtr, y)
    p = m.predict_proba(Xte)
    order = [list(m.classes_).index(c) for c in range(3)]
    return p[:, order]

models = {
    "lr_combo_C6":   (lambda: LogisticRegression(C=6, solver="saga", multi_class="multinomial", max_iter=400, n_jobs=4, random_state=RANDOM_STATE), "combo_wc"),
    "lr_combo_C3":   (lambda: LogisticRegression(C=3, solver="saga", multi_class="multinomial", max_iter=400, n_jobs=4, random_state=RANDOM_STATE), "combo_wc"),
    "lr_word12_C6":  (lambda: LogisticRegression(C=6, solver="saga", multi_class="multinomial", max_iter=400, n_jobs=4, random_state=RANDOM_STATE), "word12"),
    "lr_char35_C6":  (lambda: LogisticRegression(C=6, solver="saga", multi_class="multinomial", max_iter=400, n_jobs=4, random_state=RANDOM_STATE), "char35"),
    "lr_char26_C4":  (lambda: LogisticRegression(C=4, solver="saga", multi_class="multinomial", max_iter=400, n_jobs=4, random_state=RANDOM_STATE), "char26"),
    "lr_combo2_C6":  (lambda: LogisticRegression(C=6, solver="saga", multi_class="multinomial", max_iter=400, n_jobs=4, random_state=RANDOM_STATE), "combo_wc2"),
    "mnb_word12":    (lambda: MultinomialNB(alpha=0.05), "word12"),
    "mnb_word123":   (lambda: MultinomialNB(alpha=0.1), "word123"),
    "cnb_word12":    (lambda: ComplementNB(alpha=0.3), "word12"),
    "mnb_char35":    (lambda: MultinomialNB(alpha=0.2), "char35"),
    "svc_combo":     (lambda: CalibratedClassifierCV(LinearSVC(C=0.5, random_state=RANDOM_STATE, max_iter=3000), cv=3, method="sigmoid"), "combo_wc"),
}

oofs, lls, testps = {}, {}, {}
for name, (maker, view) in models.items():
    Xtr, Xte = views[view]
    ll, oof = oof_predict(maker, Xtr, name)
    lls[name] = ll
    oofs[name] = oof
    testps[name] = full_predict(maker, Xtr, Xte)

# --- greedy weighted ensemble on OOF (forward selection with replacement) ---
names = list(oofs.keys())
best_ll = min(lls.values())
best_name = min(lls, key=lls.get)
print(f"\nbest single: {best_name} {best_ll:.5f}")

sel = [best_name]
cur = oofs[best_name].copy()
improved = True
it = 0
while improved and it < 30:
    improved = False
    it += 1
    for n in names:
        cand = (cur * len(sel) + oofs[n]) / (len(sel) + 1)
        ll = log_loss(y, cand, labels=[0,1,2])
        if ll < best_ll - 1e-6:
            best_ll = ll
            sel.append(n)
            cur = cand
            improved = True
    print(f"  greedy iter{it}: ll={best_ll:.5f} sel={sel}")

from collections import Counter
cnt = Counter(sel)
total = sum(cnt.values())
weights = {k: v/total for k, v in cnt.items()}
print("ensemble weights:", {k: round(v,3) for k,v in weights.items()})
print(f"greedy ensemble OOF logloss: {best_ll:.5f}")

final = sum(testps[k] * w for k, w in weights.items())
final = np.clip(final, 1e-6, 1.0)
final /= final.sum(axis=1, keepdims=True)

sub = pd.DataFrame({"id": test["id"], "EAP": final[:,0], "HPL": final[:,1], "MWS": final[:,2]})
assert sub["id"].is_unique and set(sub["id"]) == set(test["id"])
out = os.path.join(BASE, "outputs", "submission.csv")
sub.to_csv(out, index=False)
print(f"saved {out} ({time.time()-t0:.0f}s total)")

# OOF 디버깅용 저장
np.savez(os.path.join(BASE, "solution", "oof_v2.npz"), y=y, **{f"oof_{k}": v for k,v in oofs.items()}, **{f"test_{k}": v for k,v in testps.items()})
import json
json.dump({"lls": lls, "weights": weights, "ens_ll": best_ll}, open(os.path.join(BASE,"solution","oof_v2.json"),"w"), indent=2)

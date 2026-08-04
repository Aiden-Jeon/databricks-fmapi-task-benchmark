import sys, time
import numpy as np, pandas as pd
from scipy import sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold
from feats import build_features, OPTS

RS = 42
tr = pd.read_csv("../train.csv")
y = tr.label.values - 1
n = len(tr)
Xd, texts, qtexts = build_features(tr)
print("dense feats", Xd.shape, flush=True)

yb = np.zeros(n * 4, dtype=int)
yb[np.arange(n) * 4 + y] = 1


def acc_from_scores(s):
    """s: (n*4,) scores -> accuracy"""
    return (s.reshape(n, 4).argmax(1) == y).mean()


kf = KFold(5, shuffle=True, random_state=RS)
folds = list(kf.split(np.arange(n)))


def expand(idx):
    return (idx[:, None] * 4 + np.arange(4)).ravel()


results = {}

# ---- model 1: dense features + HGB
oof = np.zeros(n * 4)
for f, (tri, vai) in enumerate(folds):
    m = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=15,
        min_samples_leaf=40, l2_regularization=1.0, random_state=RS)
    m.fit(Xd[expand(tri)], yb[expand(tri)])
    oof[expand(vai)] = m.predict_proba(Xd[expand(vai)])[:, 1]
results["hgb_dense"] = oof
print("hgb_dense acc", acc_from_scores(oof), flush=True)

# ---- model 2: option-text tfidf (char) logistic
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3,
                      sublinear_tf=True, max_features=300000)
Xt_all = vec.fit_transform(texts)
oof2 = np.zeros(n * 4)
for f, (tri, vai) in enumerate(folds):
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3,
                        sublinear_tf=True, max_features=300000)
    A = v.fit_transform([texts[i] for i in expand(tri)])
    B = v.transform([texts[i] for i in expand(vai)])
    m = LogisticRegression(C=1.0, max_iter=2000)
    m.fit(A, yb[expand(tri)])
    oof2[expand(vai)] = m.decision_function(B)
results["lr_opttext"] = oof2
print("lr_opttext acc", acc_from_scores(oof2), flush=True)

# ---- model 3: word-level tfidf on option text
oof3 = np.zeros(n * 4)
for f, (tri, vai) in enumerate(folds):
    v = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                        sublinear_tf=True, token_pattern=r"\S+")
    A = v.fit_transform([texts[i] for i in expand(tri)])
    B = v.transform([texts[i] for i in expand(vai)])
    m = LogisticRegression(C=1.0, max_iter=2000)
    m.fit(A, yb[expand(tri)])
    oof3[expand(vai)] = m.decision_function(B)
results["lr_optword"] = oof3
print("lr_optword acc", acc_from_scores(oof3), flush=True)

# ---- combine: dense + tfidf scores into HGB
for name, s in results.items():
    print(name, round(acc_from_scores(s), 4))

Z = np.column_stack([results[k] for k in results])
oof4 = np.zeros(n * 4)
for f, (tri, vai) in enumerate(folds):
    XX = np.hstack([Xd, Z])
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
        max_leaf_nodes=15, min_samples_leaf=40, random_state=RS)
    m.fit(XX[expand(tri)], yb[expand(tri)])
    oof4[expand(vai)] = m.predict_proba(XX[expand(vai)])[:, 1]
print("stack(leaky) acc", acc_from_scores(oof4))

# simple rank-average blend
def z(s):
    s = s.reshape(n, 4)
    return ((s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-9)).ravel()

for w in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
    s = z(results["hgb_dense"]) * (1 - w) + w * (z(oof2) + z(oof3)) / 2
    print("blend w=", w, round(acc_from_scores(s), 4))
np.save("/tmp/oof.npy", Z)

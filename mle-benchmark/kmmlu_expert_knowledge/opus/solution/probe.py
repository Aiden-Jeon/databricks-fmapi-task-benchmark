"""Quick OOF probes: how much signal does each model family carry?"""
import sys, os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import build_features, option_texts

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tr = pd.read_csv(os.path.join(ROOT, "train.csv"))
n = len(tr)
lab = tr["label"].values - 1
y = np.zeros(n * 4)
y[np.arange(n) * 4 + lab] = 1
groups = np.repeat(np.arange(n), 4)

F, vec = build_features(tr)
X = F.drop(columns=["qidx", "opt"]).values
opt_flat, qo_flat = option_texts(tr)
opt_flat = np.array(opt_flat, dtype=object)
qo_flat = np.array(qo_flat, dtype=object)


def qacc(scores):
    return (scores.reshape(n, 4).argmax(1) == lab).mean()


prior = np.bincount(lab, minlength=4) / n
print("prior      :", np.round(prior, 3), "acc(argmax prior) =", round(prior.max(), 4))

gkf = GroupKFold(n_splits=5)
folds = list(gkf.split(X, y, groups))

# 1) handcrafted GBM
s = np.zeros(n * 4)
for trn, val in folds:
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                       max_leaf_nodes=15, min_samples_leaf=40,
                                       l2_regularization=1.0, random_state=0)
    m.fit(X[trn], y[trn])
    s[val] = m.predict_proba(X[val])[:, 1]
print("HGB feats  : acc =", round(qacc(s), 4))
np.save(os.path.join(ROOT, "solution", "_oof_hgb.npy"), s)

# 2) option-text tfidf LR
s2 = np.zeros(n * 4)
for trn, val in folds:
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), min_df=3,
                        sublinear_tf=True)
    A = v.fit_transform(opt_flat[trn]); B = v.transform(opt_flat[val])
    m = LogisticRegression(C=0.5, max_iter=2000)
    m.fit(A, y[trn])
    s2[val] = m.decision_function(B)
print("LR opt-text: acc =", round(qacc(s2), 4))
np.save(os.path.join(ROOT, "solution", "_oof_lropt.npy"), s2)

# 3) question+option tfidf LR
s3 = np.zeros(n * 4)
for trn, val in folds:
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3,
                        sublinear_tf=True)
    A = v.fit_transform(qo_flat[trn]); B = v.transform(qo_flat[val])
    m = LogisticRegression(C=0.3, max_iter=2000)
    m.fit(A, y[trn])
    s3[val] = m.decision_function(B)
print("LR q+opt   : acc =", round(qacc(s3), 4))
np.save(os.path.join(ROOT, "solution", "_oof_lrqo.npy"), s3)

# 4) word-level option LR
s4 = np.zeros(n * 4)
for trn, val in folds:
    v = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                        sublinear_tf=True)
    A = v.fit_transform(opt_flat[trn]); B = v.transform(opt_flat[val])
    m = LogisticRegression(C=1.0, max_iter=2000)
    m.fit(A, y[trn])
    s4[val] = m.decision_function(B)
print("LR opt word: acc =", round(qacc(s4), 4))
np.save(os.path.join(ROOT, "solution", "_oof_lrw.npy"), s4)

# blends
for w in [0.1, 0.3, 0.5]:
    z = lambda a: (a - a.mean()) / (a.std() + 1e-9)
    print(f"blend hgb+{w}*lropt:", round(qacc(z(s) + w * z(s2)), 4),
          f" hgb+{w}*lrqo:", round(qacc(z(s) + w * z(s3)), 4))

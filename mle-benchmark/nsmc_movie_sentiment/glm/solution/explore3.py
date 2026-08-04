import pandas as pd, numpy as np, re, sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack

tr = pd.read_csv("train.csv")
te = pd.read_csv("test.csv")

def norm(s):
    s = str(s).lower()
    s = re.sub(r"[^가-힣a-z0-9]", " ", s)
    return s

tr["d"] = tr.document.map(norm)
te["d"] = te.document.map(norm)
y = tr.label.values

# word (1,2) + char (3,5)
vw = TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True, max_features=80000, analyzer="word")
Xw_tr = vw.fit_transform(tr.d)
vc = TfidfVectorizer(ngram_range=(3,5), min_df=3, sublinear_tf=True, max_features=150000, analyzer="char_wb")
Xc_tr = vc.fit_transform(tr.d)
X_tr = hstack([Xw_tr, Xc_tr]).tocsr()
print("combo", X_tr.shape, file=sys.stderr)

for C in [1.0, 3.0, 6.0]:
    clf = LogisticRegression(C=C, max_iter=1000, n_jobs=1)
    scores = cross_val_score(clf, X_tr, y, cv=5, scoring="accuracy", n_jobs=4)
    print(f"combo C={C} cv", scores.mean(), scores, file=sys.stderr)

for C in [0.5, 1.0, 3.0]:
    clf = LinearSVC(C=C, max_iter=2000)
    scores = cross_val_score(clf, X_tr, y, cv=5, scoring="accuracy", n_jobs=4)
    print(f"svc C={C} cv", scores.mean(), scores, file=sys.stderr)

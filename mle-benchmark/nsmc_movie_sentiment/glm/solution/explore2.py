import pandas as pd, numpy as np, re, sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, cross_val_predict
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

# char n-grams (3-5 is typical for Korean char-level)
vchar = TfidfVectorizer(ngram_range=(3,5), min_df=3, sublinear_tf=True, max_features=150000, analyzer="char_wb")
Xc_tr = vchar.fit_transform(tr.d)
print("char", Xc_tr.shape, file=sys.stderr)

clf = LogisticRegression(C=3.0, max_iter=1000, n_jobs=1)
scores = cross_val_score(clf, Xc_tr, y, cv=5, scoring="accuracy", n_jobs=4)
print("char3-5 cv", scores.mean(), scores, file=sys.stderr)

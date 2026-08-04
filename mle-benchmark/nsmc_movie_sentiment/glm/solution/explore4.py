import pandas as pd, numpy as np, re, sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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

# char ranges
for ng in [(2,5),(3,6),(2,6),(3,5)]:
    vec = TfidfVectorizer(ngram_range=ng, min_df=3, sublinear_tf=True, max_features=200000, analyzer="char_wb")
    Xtr = vec.fit_transform(tr.d)
    for C in [2.0, 4.0, 6.0]:
        clf = LogisticRegression(C=C, max_iter=1000, n_jobs=1)
        scores = cross_val_score(clf, Xtr, y, cv=5, scoring="accuracy", n_jobs=4)
        print(f"char{ng} C={C} cv {scores.mean():.4f} {scores}", file=sys.stderr)

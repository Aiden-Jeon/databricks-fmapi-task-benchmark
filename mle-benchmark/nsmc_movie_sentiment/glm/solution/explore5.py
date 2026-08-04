import pandas as pd, numpy as np, re, sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

tr = pd.read_csv("train.csv")
te = pd.read_csv("test.csv")

def norm(s):
    s = str(s).lower()
    s = re.sub(r"[^가-힣a-z0-9]", " ", s)
    return s

tr["d"] = tr.document.map(norm)
te["d"] = te.document.map(norm)
y = tr.label.values

configs = [
    dict(ngram_range=(2,5), min_df=2, max_features=300000),
    dict(ngram_range=(2,5), min_df=4, max_features=200000),
    dict(ngram_range=(2,5), min_df=5, max_features=200000),
    dict(ngram_range=(2,5), min_df=3, max_features=300000),
]
for cfg in configs:
    vec = TfidfVectorizer(sublinear_tf=True, analyzer="char_wb", **cfg)
    Xtr = vec.fit_transform(tr.d)
    for C in [3.0, 4.0, 5.0]:
        clf = LogisticRegression(C=C, max_iter=1000, n_jobs=1)
        scores = cross_val_score(clf, Xtr, y, cv=5, scoring="accuracy", n_jobs=4)
        print(f"{cfg} C={C} cv {scores.mean():.4f} {scores}", file=sys.stderr)

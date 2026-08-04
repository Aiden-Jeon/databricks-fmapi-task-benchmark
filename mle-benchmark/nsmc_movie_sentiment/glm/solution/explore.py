import pandas as pd, numpy as np, re, sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

tr = pd.read_csv("train.csv")
te = pd.read_csv("test.csv")

def norm(s):
    s = str(s).lower()
    s = re.sub(r"[^가-힣a-z0-9]", " ", s)
    return s

tr["d"] = tr.document.map(norm)
te["d"] = te.document.map(norm)

# char n-grams are strong for Korean
vec = TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True, max_features=80000, analyzer="char_wb")
Xtr = vec.fit_transform(tr.d)
Xte = vec.transform(te.d)
y = tr.label.values
print("char Xtr", Xtr.shape, file=sys.stderr)

clf = LogisticRegression(C=3.0, max_iter=1000, n_jobs=1)
scores = cross_val_score(clf, Xtr, y, cv=5, scoring="accuracy", n_jobs=4)
print("char cv acc", scores.mean(), scores, file=sys.stderr)

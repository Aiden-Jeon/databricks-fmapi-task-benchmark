import pandas as pd, numpy as np, re, sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

tr = pd.read_csv("train.csv")
te = pd.read_csv("test.csv")

def norm(s):
    s = str(s).lower()
    s = re.sub(r"[^가-힣a-z0-9]", " ", s)
    return s

tr["d"] = tr.document.map(norm)
te["d"] = te.document.map(norm)
y = tr.label.values

vec = TfidfVectorizer(ngram_range=(3,5), min_df=3, sublinear_tf=True, max_features=150000, analyzer="char_wb")
Xtr = vec.fit_transform(tr.d)
Xte = vec.transform(te.d)

clf = LogisticRegression(C=3.0, max_iter=1000, n_jobs=1)
clf.fit(Xtr, y)
pred = clf.predict(Xte)

out = pd.DataFrame({"id": te.id.values, "label": pred.astype(int)})
out.to_csv("outputs/submission.csv", index=False)
print("wrote", out.shape, file=sys.stderr)
print(out.label.value_counts(normalize=True).to_string(), file=sys.stderr)

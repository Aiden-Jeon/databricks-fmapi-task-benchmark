"""Quick baseline: word TF-IDF on (premise [SEP] hypothesis) + LogisticRegression."""
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

train = pd.read_csv(os.path.join(ROOT, "train.csv"))
test = pd.read_csv(os.path.join(ROOT, "test.csv"))

combo_tr = (train["sentence1"].fillna("") + " [SEP] " + train["sentence2"].fillna(""))
combo_te = (test["sentence1"].fillna("") + " [SEP] " + test["sentence2"].fillna(""))
y = train["label"].values

vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
X = vec.fit_transform(combo_tr)
Xt = vec.transform(combo_te)

tr, va = train_test_split(range(len(y)), test_size=0.05, random_state=0, stratify=y)
clf = LogisticRegression(C=4.0, max_iter=1000, n_jobs=4)
clf.fit(X[tr], y[tr])
acc = clf.score(X[va], y[va])
print(f"baseline val acc = {acc:.4f}")

clf.fit(X, y)
pred = clf.predict(Xt)
out = pd.DataFrame({"id": test["id"], "label": pred})
os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
out.to_csv(os.path.join(ROOT, "outputs", "submission.csv"), index=False)
print("saved outputs/submission.csv", out.shape)

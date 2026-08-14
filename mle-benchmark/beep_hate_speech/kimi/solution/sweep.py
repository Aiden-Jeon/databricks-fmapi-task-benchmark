"""Wider sweep of vectorizers / hyperparams."""
import pandas as pd
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import f1_score

train = pd.read_csv("train.csv")
y = train["label"]
X_text = train["comment"].astype(str)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

vecs = {
    "char(2,5)": TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
    "char(2,6)": TfidfVectorizer(analyzer="char", ngram_range=(2, 6), min_df=2, sublinear_tf=True),
    "char(3,5)": TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, sublinear_tf=True),
    "char_wb(2,5)": TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
    "char_wb(3,5)": TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True),
    "char(2,5)mindf1": TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=1, sublinear_tf=True),
}

results = {}
for vname, vec in vecs.items():
    X = vec.fit_transform(X_text)
    for cname, clf in {
        "LR C4": LogisticRegression(max_iter=2000, class_weight="balanced", C=4.0),
        "LR C1": LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0),
        "LR C8": LogisticRegression(max_iter=2000, class_weight="balanced", C=8.0),
        "SVC C1": LinearSVC(C=1.0, class_weight="balanced", dual="auto"),
        "SVC C0.3": LinearSVC(C=0.3, class_weight="balanced", dual="auto"),
    }.items():
        pred = cross_val_predict(clf, X, y, cv=skf, n_jobs=5)
        f1 = f1_score(y, pred, average="macro")
        results[f"{vname} / {cname}"] = f1

for k, v in sorted(results.items(), key=lambda x: -x[1]):
    print(f"{k}: {v:.4f}")

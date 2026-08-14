"""CV evaluation of candidate models."""
import pandas as pd
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import f1_score, classification_report


def build(mode):
    if mode == "word+char":
        vec = [
            TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True),
            TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, sublinear_tf=True),
        ]
    elif mode == "char2-5":
        vec = [TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2, sublinear_tf=True)]
    return vec


def featurize(vec, train_text, test_text):
    trs = hstack([v.fit_transform(train_text) for v in vec])
    tes = hstack([v.transform(test_text) for v in vec])
    return trs, tes


train = pd.read_csv("train.csv")
y = train["label"]
X_text = train["comment"].astype(str)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

configs = {
    "word+char / LR C4 bal": ("word+char", LogisticRegression(max_iter=2000, class_weight="balanced", C=4.0)),
    "word+char / SVC C1 bal": ("word+char", LinearSVC(C=1.0, class_weight="balanced")),
    "char2-5 / LR C4 bal": ("char2-5", LogisticRegression(max_iter=2000, class_weight="balanced", C=4.0)),
}

for name, (mode, clf) in configs.items():
    vec = build(mode)
    X, _ = featurize(vec, X_text, X_text)
    pred = cross_val_predict(clf, X, y, cv=skf, n_jobs=5)
    f1 = f1_score(y, pred, average="macro")
    print(f"{name}: macro F1 = {f1:.4f}")

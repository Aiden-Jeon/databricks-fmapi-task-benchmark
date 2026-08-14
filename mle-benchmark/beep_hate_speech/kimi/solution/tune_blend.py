"""Tune LR/CNB blend weight and CNB alpha via repeated CV."""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

CLASSES = np.array(["hate", "none", "offensive"])
SEEDS = [0, 1, 2]

train = pd.read_csv("train.csv")
y = train["label"].values

vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True)
X = vec.fit_transform(train["comment"].astype(str))

oof = {m: np.zeros((len(y), 3)) for m in ["lr", "cnb03", "cnb05", "cnb10"]}
for seed in SEEDS:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr_idx, va_idx in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.75)
        clf.fit(X[tr_idx], y[tr_idx])
        oof["lr"][va_idx] += clf.predict_proba(X[va_idx]) / len(SEEDS)
        for alpha, key in [(0.3, "cnb03"), (0.5, "cnb05"), (1.0, "cnb10")]:
            nb = ComplementNB(alpha=alpha)
            nb.fit(X[tr_idx], y[tr_idx])
            oof[key][va_idx] += nb.predict_proba(X[va_idx]) / len(SEEDS)

for key in ["cnb03", "cnb05", "cnb10"]:
    for w in [0.6, 0.65, 0.7, 0.75, 0.8]:
        b = oof["lr"] * w + oof[key] * (1 - w)
        print(f"lr {w} + {key} {1-w:.2f}: {f1_score(y, CLASSES[b.argmax(axis=1)], average='macro'):.4f}")

"""Blend LR (char_wb) with MultinomialNB and SVC-calibrated probs; also seed-ensembled.

Evaluate via repeated CV, then write submission if better than final_model.
"""
import pandas as pd
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

CLASSES = np.array(["hate", "none", "offensive"])
SEEDS = [0, 1, 2]

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
y = train["label"].values

vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True)
Xtr = vec.fit_transform(train["comment"].astype(str))
Xte = vec.transform(test["comment"].astype(str))


def models():
    return {
        "lr": LogisticRegression(max_iter=2000, class_weight="balanced", C=0.75),
        "cnb": ComplementNB(alpha=0.5),
        "svc": CalibratedClassifierCV(LinearSVC(C=0.3, class_weight="balanced", dual="auto"), cv=3),
    }


# ---- repeated-CV per model + blend ----
oof = {m: np.zeros((len(y), 3)) for m in models()}
for seed in SEEDS:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr_idx, va_idx in skf.split(Xtr, y):
        for name, clf in models().items():
            clf.fit(Xtr[tr_idx], y[tr_idx])
            oof[name][va_idx] += clf.predict_proba(Xtr[va_idx]) / len(SEEDS)

for name, p in oof.items():
    print(f"{name}: {f1_score(y, CLASSES[p.argmax(axis=1)], average='macro'):.4f}")

blend = oof["lr"] * 0.5 + oof["cnb"] * 0.2 + oof["svc"] * 0.3
print(f"blend(0.5/0.2/0.3): {f1_score(y, CLASSES[blend.argmax(axis=1)], average='macro'):.4f}")
blend = oof["lr"] * 0.6 + oof["svc"] * 0.4
print(f"blend(lr0.6/svc0.4): {f1_score(y, CLASSES[blend.argmax(axis=1)], average='macro'):.4f}")
blend = oof["lr"] * 0.7 + oof["cnb"] * 0.3
print(f"blend(lr0.7/cnb0.3): {f1_score(y, CLASSES[blend.argmax(axis=1)], average='macro'):.4f}")

# ---- full-data fit & predict with best blend ----
full = {}
for name, clf in models().items():
    clf.fit(Xtr, y)
    full[name] = clf.predict_proba(Xte)
proba = full["lr"] * 0.5 + full["cnb"] * 0.2 + full["svc"] * 0.3
pred = CLASSES[proba.argmax(axis=1)]
sub = pd.DataFrame({"id": test["id"], "label": pred})
sub.to_csv("outputs/submission_blend.csv", index=False)
print(sub["label"].value_counts())

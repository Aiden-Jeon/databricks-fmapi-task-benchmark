"""Fine-tune char_wb LR C around 1, and char_wb ngram ranges."""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import f1_score

train = pd.read_csv("train.csv")
y = train["label"]
X_text = train["comment"].astype(str)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {}
for name, vec in {
    "wb(2,4)": TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True),
    "wb(2,5)": TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
    "wb(3,6)": TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 6), min_df=2, sublinear_tf=True),
    "wb(2,5)m3": TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True),
}.items():
    X = vec.fit_transform(X_text)
    for C in [0.5, 0.75, 1.0, 1.5]:
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=C)
        oof = cross_val_predict(clf, X, y, cv=skf, n_jobs=5, method="predict_proba")
        classes = np.array(sorted(y.unique()))
        lab = classes[oof.argmax(axis=1)]
        results[f"{name} / LR C{C}"] = f1_score(y, lab, average="macro")

for k, v in sorted(results.items(), key=lambda x: -x[1]):
    print(f"{k}: {v:.4f}")

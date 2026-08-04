"""Quick baseline: word TF-IDF + Logistic Regression."""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.metrics import log_loss

CLASSES = ["EAP", "HPL", "MWS"]

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
y = train["author"].values

pipe = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, strip_accents="unicode"),
    LogisticRegression(C=10, max_iter=2000),
)
cv = StratifiedKFold(5, shuffle=True, random_state=42)
oof = cross_val_predict(pipe, train["text"], y, cv=cv, method="predict_proba", n_jobs=-1)
print("baseline CV logloss:", log_loss(y, oof, labels=CLASSES))

pipe.fit(train["text"], y)
p = pipe.predict_proba(test["text"])
sub = pd.DataFrame(p, columns=list(pipe.classes_))
sub.insert(0, "id", test["id"].values)
sub = sub[["id"] + CLASSES]
sub.to_csv("outputs/submission.csv", index=False)
print(sub.shape)

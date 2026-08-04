"""Quick baseline: TF-IDF word n-gram + Logistic Regression for spooky author ID."""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

BASE = "/tmp/kmle/M3_t2_spooky_full_20260804_033550/task"
CLASSES = ["EAP", "HPL", "MWS"]
SEED = 42

train = pd.read_csv(f"{BASE}/train.csv")
test = pd.read_csv(f"{BASE}/test.csv")
sub = pd.read_csv(f"{BASE}/sample_submission.csv")

y = train["author"].values
X_text = train["text"].fillna("").values
Xt_text = test["text"].fillna("").values

vec = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=3,
    sublinear_tf=True,
    strip_accents="unicode",
    lowercase=True,
    token_pattern=r"\w+",
)
X = vec.fit_transform(X_text)
Xt = vec.transform(Xt_text)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof = np.zeros((len(train), 3))
test_pred = np.zeros((len(test), 3))
for fi, (tr, va) in enumerate(skf.split(X, y)):
    clf = LogisticRegression(
        C=4.0, solver="liblinear", max_iter=1000, class_weight=None, random_state=SEED
    )
    clf.fit(X[tr], y[tr])
    oof[va] = clf.predict_proba(X[va])
    test_pred += clf.predict_proba(Xt) / skf.n_splits

print("OOF logloss:", log_loss(y, oof, labels=CLASSES))
# clip
eps = 1e-6
oof = np.clip(oof, eps, 1 - eps)
test_pred = np.clip(test_pred, eps, 1 - eps)

out = sub.copy()
out["EAP"] = test_pred[:, 0]
out["HPL"] = test_pred[:, 1]
out["MWS"] = test_pred[:, 2]
out.to_csv(f"{BASE}/outputs/submission.csv", index=False)
print("Wrote baseline submission.")
print(out.head())

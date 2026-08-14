"""Feature combos and hierarchical stacking vs plain multiclass."""
import pandas as pd
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import f1_score, classification_report

train = pd.read_csv("train.csv")
y = train["label"]
X_text = train["comment"].astype(str)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- Feature combo ---
combo = hstack([
    TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True).fit_transform(X_text),
    TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2, sublinear_tf=True).fit_transform(X_text),
    TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True).fit_transform(X_text),
])
X = combo.tocsr()

classes = np.array(["hate", "none", "offensive"])  # sorted order used by sklearn

for C in [1.0, 2.0]:
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=C)
    oof = cross_val_predict(clf, X, y, cv=skf, n_jobs=5, method="predict_proba")
    lab = classes[oof.argmax(axis=1)]
    print(f"combo / LR C{C} macro F1: {f1_score(y, lab, average='macro'):.4f}")

# --- Hierarchical stacking on OOF probs ---
clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
oof = cross_val_predict(clf, X, y, cv=skf, n_jobs=5, method="predict_proba")
idx = {c: i for i, c in enumerate(classes)}
p_none = oof[:, idx["none"]]

# stage 1: none vs rest
y_bin = (y != "none").astype(int)
oof_bin = cross_val_predict(
    LogisticRegression(C=1.0), p_none.reshape(-1, 1), y_bin, cv=skf, n_jobs=5, method="predict_proba"
)[:, 1]

# stage 2: offensive vs hate, trained on non-none rows
y_oh = (y == "hate").astype(int)
mask = y != "none"
X_oh = np.hstack([oof, oof_bin.reshape(-1, 1)])
oof_oh = np.zeros(len(y))
oof_oh[mask] = cross_val_predict(
    LogisticRegression(C=1.0), X_oh[mask], y_oh[mask], cv=skf, n_jobs=5, method="predict_proba"
)[:, 1]

final = np.where(oof_bin < 0.5, "none", np.where(oof_oh >= 0.5, "hate", "offensive"))
print(f"hierarchical macro F1: {f1_score(y, final, average='macro'):.4f}")

# tune threshold on binary stage
best_t, best_f = 0.5, 0.0
for t in np.arange(0.2, 0.8, 0.05):
    f = np.where(oof_bin < t, "none", np.where(oof_oh >= 0.5, "hate", "offensive"))
    sc = f1_score(y, f, average="macro")
    if sc > best_f:
        best_f, best_t = sc, t
print(f"hierarchical tuned t={best_t:.2f}: {best_f:.4f}")
print(classification_report(y, np.where(oof_bin < best_t, "none", np.where(oof_oh >= 0.5, "hate", "offensive"))))

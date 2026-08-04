"""Compare different TF-IDF configurations and models for KoBEST BoolQ."""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, PassiveAggressiveClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANDOM_STATE = 42

train = pd.read_csv(os.path.join(ROOT, "train.csv"))
test = pd.read_csv(os.path.join(ROOT, "test.csv"))

train["text"] = train["paragraph"].astype(str) + " " + train["question"].astype(str)
test["text"] = test["paragraph"].astype(str) + " " + test["question"].astype(str)
train["text"] = train["text"].str.replace("\n", " ")
test["text"] = test["text"].str.replace("\n", " ")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

configs = []

# word/char variations
for wf in [30000, 50000, 80000]:
    for ngram in [(1,1),(1,2),(1,3)]:
        word_vec = TfidfVectorizer(sublinear_tf=True, ngram_range=ngram, min_df=2, max_df=0.95, max_features=wf, analyzer="word", token_pattern=r"(?u)\b\w+\b")
        char_vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(2,4), min_df=2, max_df=0.95, max_features=wf, analyzer="char_wb")
        features = FeatureUnion([("word", word_vec), ("char", char_vec)])
        for clf_name, clf in [
            ("logreg_c2", LogisticRegression(C=2.0, max_iter=2000, solver="liblinear", random_state=RANDOM_STATE)),
            ("logreg_c4", LogisticRegression(C=4.0, max_iter=2000, solver="liblinear", random_state=RANDOM_STATE)),
            ("logreg_c8", LogisticRegression(C=8.0, max_iter=2000, solver="liblinear", random_state=RANDOM_STATE)),
            ("logreg_c1", LogisticRegression(C=1.0, max_iter=2000, solver="liblinear", random_state=RANDOM_STATE)),
            ("logreg_l1", LogisticRegression(C=4.0, max_iter=2000, solver="liblinear", penalty="l1", random_state=RANDOM_STATE)),
            ("linearsvc", LinearSVC(C=1.0, max_iter=5000, random_state=RANDOM_STATE)),
            ("mnb", MultinomialNB(alpha=0.3)),
            ("cnb", ComplementNB(alpha=0.3)),
        ]:
            configs.append((f"wf={wf} ng={ngram} {clf_name}", Pipeline([("f", features), ("c", clf)])))

best = (0, None, None)
for name, pipe in configs:
    try:
        s = cross_val_score(pipe, train["text"], train["label"], cv=skf, scoring="accuracy", n_jobs=-1)
        m = s.mean()
        if m > best[0]:
            best = (m, name, pipe)
        print(f"{m:.4f} +/- {s.std():.4f}  {name}", file=sys.stderr)
    except Exception as e:
        print(f"ERR {name}: {e}", file=sys.stderr)

print("\nBEST: %.4f %s" % (best[0], best[1]), file=sys.stderr)

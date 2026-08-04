"""Quick baseline: TF-IDF char/word + LinearSVC / LogReg, CV macro-F1."""
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, classification_report

TASK = "/tmp/kmle/M1_t23_korfin_asc_full_20260804_033519/task"
tr = pd.read_csv(f"{TASK}/train.csv")
te = pd.read_csv(f"{TASK}/test.csv")

MASK = " @ASP@ "


def mask_sent(s, a):
    if isinstance(a, str) and a and a in s:
        return s.replace(a, MASK)
    return s


def window(s, a, w=30):
    if not (isinstance(a, str) and a and a in s):
        return s
    out = []
    for m in re.finditer(re.escape(a), s):
        lo = max(0, m.start() - w)
        hi = min(len(s), m.end() + w)
        out.append(s[lo:m.start()] + MASK + s[m.end():hi])
    return " || ".join(out)


for df in (tr, te):
    df["masked"] = [mask_sent(s, a) for s, a in zip(df.sentence, df.aspect)]
    df["win"] = [window(s, a) for s, a in zip(df.sentence, df.aspect)]
    df["text"] = df["masked"] + " ~~ " + df["win"] + " ~~ ASPECT " + df.aspect.fillna("")

y = tr.label.values


def build(kind):
    feats = FeatureUnion([
        ("cw", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                               sublinear_tf=True, max_features=400000)),
        ("w", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                              sublinear_tf=True)),
    ])
    if kind == "svc":
        clf = LinearSVC(C=0.5)
    else:
        clf = LogisticRegression(C=5, max_iter=2000)
    return Pipeline([("f", feats), ("c", clf)])


cv = StratifiedKFold(5, shuffle=True, random_state=42)
for col in ["sentence", "masked", "text"]:
    for kind in ["svc", "lr"]:
        p = cross_val_predict(build(kind), tr[col], y, cv=cv, n_jobs=5)
        print(f"{col:9s} {kind:4s} macro_f1={f1_score(y, p, average='macro'):.4f}")

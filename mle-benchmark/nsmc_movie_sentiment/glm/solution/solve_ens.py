import pandas as pd, numpy as np, re, sys, time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from scipy.sparse import hstack
from scipy.special import expit

t0 = time.time()
tr = pd.read_csv("train.csv")
te = pd.read_csv("test.csv")

def norm(s):
    s = str(s).lower()
    s = re.sub(r"[^가-힣a-z0-9]", " ", s)
    return s

tr["d"] = tr.document.map(norm)
te["d"] = te.document.map(norm)
y = tr.label.values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def run(vec, C, tag):
    Xtr = vec.fit_transform(tr.d)
    Xte = vec.transform(te.d)
    clf = LogisticRegression(C=C, max_iter=1000, n_jobs=1)
    oof = cross_val_predict(clf, Xtr, y, cv=skf, method="decision_function", n_jobs=4)
    acc = ((oof > 0).astype(int) == y).mean()
    print(f"[{tag}] oof acc {acc:.4f} shape {Xtr.shape} time {time.time()-t0:.0f}s", file=sys.stderr)
    clf.fit(Xtr, y)
    test_df = clf.decision_function(Xte)
    return oof, test_df

# Model A: char (2,5) min_df=4
oof_a, te_a = run(
    TfidfVectorizer(ngram_range=(2,5), min_df=4, sublinear_tf=True, max_features=200000, analyzer="char_wb"),
    C=3.0, tag="char25_mdf4")

# Model B: char (2,5) min_df=2
oof_b, te_b = run(
    TfidfVectorizer(ngram_range=(2,5), min_df=2, sublinear_tf=True, max_features=300000, analyzer="char_wb"),
    C=3.0, tag="char25_mdf2")

# Model C: word (1,2) + char
vw = TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True, max_features=80000, analyzer="word")
Xw_tr = vw.fit_transform(tr.d); Xw_te = vw.transform(te.d)
vc = TfidfVectorizer(ngram_range=(2,5), min_df=4, sublinear_tf=True, max_features=200000, analyzer="char_wb")
Xc_tr = vc.fit_transform(tr.d); Xc_te = vc.transform(te.d)
Xcc_tr = hstack([Xw_tr, Xc_tr]).tocsr(); Xcc_te = hstack([Xw_te, Xc_te]).tocsr()
clf = LogisticRegression(C=2.0, max_iter=1000, n_jobs=1)
oof_c = cross_val_predict(clf, Xcc_tr, y, cv=skf, method="decision_function", n_jobs=4)
acc_c = ((oof_c > 0).astype(int) == y).mean()
print(f"[word+char] oof acc {acc_c:.4f} shape {Xcc_tr.shape} time {time.time()-t0:.0f}s", file=sys.stderr)
clf.fit(Xcc_tr, y)
te_c = clf.decision_function(Xcc_te)

# Ensemble: average decision functions
for wa, wb, wc in [(1,0,0),(1,1,0),(1,1,1),(2,1,1),(1,2,1),(1,1,2)]:
    oof = (wa*oof_a + wb*oof_b + wc*oof_c) / (wa+wb+wc)
    acc = ((oof > 0).astype(int) == y).mean()
    print(f"ens a{wa}b{wb}c{wc} oof acc {acc:.4f}", file=sys.stderr)

oof = (oof_a + oof_b + oof_c) / 3
acc = ((oof > 0).astype(int) == y).mean()
print(f"FINAL mean3 oof acc {acc:.4f}", file=sys.stderr)

te_df = (te_a + te_b + te_c) / 3
pred = (te_df > 0).astype(int)
out = pd.DataFrame({"id": te.id.values, "label": pred})
out.to_csv("outputs/submission.csv", index=False)
print("wrote", out.shape, "dist\n", out.label.value_counts(normalize=True).to_string(), file=sys.stderr)
print("total time", time.time()-t0, file=sys.stderr)

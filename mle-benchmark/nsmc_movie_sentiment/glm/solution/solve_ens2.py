import pandas as pd, numpy as np, re, sys, time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from scipy.sparse import hstack

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

def oof_pred(vec, clf, tag, method="decision_function"):
    Xtr = vec.fit_transform(tr.d)
    Xte = vec.transform(te.d)
    oof = cross_val_predict(clf, Xtr, y, cv=skf, method=method, n_jobs=4)
    acc = ((oof > 0).astype(int) == y).mean()
    print(f"[{tag}] oof acc {acc:.4f} shape {Xtr.shape} t {time.time()-t0:.0f}s", file=sys.stderr)
    clf.fit(Xtr, y)
    te_s = clf.decision_function(Xte) if hasattr(clf,"decision_function") else clf.predict_proba(Xte)[:,1]
    return oof, te_s

# A: char (2,5) min_df=4 C=3
oof_a, te_a = oof_pred(
    TfidfVectorizer(ngram_range=(2,5), min_df=4, sublinear_tf=True, max_features=200000, analyzer="char_wb"),
    LogisticRegression(C=3.0, max_iter=1000, n_jobs=1), "char25_mdf4")
# B: char (2,5) min_df=2 C=3
oof_b, te_b = oof_pred(
    TfidfVectorizer(ngram_range=(2,5), min_df=2, sublinear_tf=True, max_features=300000, analyzer="char_wb"),
    LogisticRegression(C=3.0, max_iter=1000, n_jobs=1), "char25_mdf2")
# C: word(1,2)+char(2,5) C=2
vw = TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True, max_features=80000, analyzer="word")
Xw_tr = vw.fit_transform(tr.d); Xw_te = vw.transform(te.d)
vc = TfidfVectorizer(ngram_range=(2,5), min_df=4, sublinear_tf=True, max_features=200000, analyzer="char_wb")
Xc_tr = vc.fit_transform(tr.d); Xc_te = vc.transform(te.d)
Xcc_tr = hstack([Xw_tr, Xc_tr]).tocsr(); Xcc_te = hstack([Xw_te, Xc_te]).tocsr()
clf_c = LogisticRegression(C=2.0, max_iter=1000, n_jobs=1)
oof_c = cross_val_predict(clf_c, Xcc_tr, y, cv=skf, method="decision_function", n_jobs=4)
acc_c = ((oof_c > 0).astype(int) == y).mean()
print(f"[word+char] oof acc {acc_c:.4f} t {time.time()-t0:.0f}s", file=sys.stderr)
clf_c.fit(Xcc_tr, y); te_c = clf_c.decision_function(Xcc_te)
# D: ComplementNB on char counts (2,5)
cv = CountVectorizer(ngram_range=(2,5), min_df=4, max_features=200000, analyzer="char_wb")
Xcv_tr = cv.fit_transform(tr.d); Xcv_te = cv.transform(te.d)
clf_d = ComplementNB()
oof_d = cross_val_predict(clf_d, Xcv_tr, y, cv=skf, method="predict_proba", n_jobs=4)[:,1]
oof_d = oof_d - 0.5
acc_d = ((oof_d > 0).astype(int) == y).mean()
print(f"[cnb] oof acc {acc_d:.4f} t {time.time()-t0:.0f}s", file=sys.stderr)
clf_d.fit(Xcv_tr, y); te_d = clf_d.predict_proba(Xcv_te)[:,1] - 0.5
# E: char (3,5) min_df=3 C=3
oof_e, te_e = oof_pred(
    TfidfVectorizer(ngram_range=(3,5), min_df=3, sublinear_tf=True, max_features=200000, analyzer="char_wb"),
    LogisticRegression(C=3.0, max_iter=1000, n_jobs=1), "char35_mdf3")

oofs = {"a":oof_a,"b":oof_b,"c":oof_c,"d":oof_d,"e":oof_e}
tes = {"a":te_a,"b":te_b,"c":te_c,"d":te_d,"e":te_e}
# search weights
best=(0,None)
import itertools
names=list(oofs.keys())
for r in [2,3]:
    for combo in itertools.combinations_with_replacement(names, r):
        oof = np.mean([oofs[n] for n in combo], axis=0)
        acc = ((oof>0).astype(int)==y).mean()
        if acc>best[0]: best=(acc,combo)
print("best combo", best, "t", time.time()-t0, file=sys.stderr)
# all-5 mean
oof_all = np.mean(list(oofs.values()), axis=0)
acc_all = ((oof_all>0).astype(int)==y).mean()
print("all5 mean acc", acc_all, file=sys.stderr)

combo = best[1]
te_df = np.mean([tes[n] for n in combo], axis=0)
pred = (te_df > 0).astype(int)
out = pd.DataFrame({"id": te.id.values, "label": pred})
out.to_csv("outputs/submission.csv", index=False)
print("wrote", out.shape, "combo", combo, "dist\n", out.label.value_counts(normalize=True).to_string(), file=sys.stderr)
print("total", time.time()-t0, file=sys.stderr)

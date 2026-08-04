import pandas as pd, numpy as np, re, sys, time, itertools
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB, BernoulliNB, MultinomialNB
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

oofs, tes = {}, {}
def add(tag, vec, clf, method="decision_function"):
    Xtr = vec.fit_transform(tr.d); Xte = vec.transform(te.d)
    oof = cross_val_predict(clf, Xtr, y, cv=skf, method=method, n_jobs=4)
    if method == "predict_proba": oof = oof[:,1]-0.5
    acc = ((oof > 0).astype(int) == y).mean()
    print(f"[{tag}] oof {acc:.4f} {Xtr.shape} t {time.time()-t0:.0f}s", file=sys.stderr)
    clf.fit(Xtr, y)
    s = clf.decision_function(Xte) if hasattr(clf,"decision_function") else clf.predict_proba(Xte)[:,1]-0.5
    oofs[tag]=oof; tes[tag]=s

LR = lambda C: LogisticRegression(C=C, max_iter=1000, n_jobs=1)
add("a", TfidfVectorizer(ngram_range=(2,5), min_df=4, sublinear_tf=True, max_features=200000, analyzer="char_wb"), LR(3.0))
add("b", TfidfVectorizer(ngram_range=(2,5), min_df=2, sublinear_tf=True, max_features=300000, analyzer="char_wb"), LR(3.0))
add("c", TfidfVectorizer(ngram_range=(3,5), min_df=3, sublinear_tf=True, max_features=200000, analyzer="char_wb"), LR(3.0))
# word+char combo
vw = TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True, max_features=80000, analyzer="word")
Xw_tr=vw.fit_transform(tr.d); Xw_te=vw.transform(te.d)
vc = TfidfVectorizer(ngram_range=(2,5), min_df=4, sublinear_tf=True, max_features=200000, analyzer="char_wb")
Xc_tr=vc.fit_transform(tr.d); Xc_te=vc.transform(te.d)
Xcc_tr=hstack([Xw_tr,Xc_tr]).tocsr(); Xcc_te=hstack([Xw_te,Xc_te]).tocsr()
clfc=LR(2.0)
oof_c2=cross_val_predict(clfc,Xcc_tr,y,cv=skf,method="decision_function",n_jobs=4)
print(f"[wc] oof {((oof_c2>0).astype(int)==y).mean():.4f} t {time.time()-t0:.0f}s", file=sys.stderr)
clfc.fit(Xcc_tr,y); oofs["d"]=oof_c2; tes["d"]=clfc.decision_function(Xcc_te)
# NB models on char counts
cv = CountVectorizer(ngram_range=(2,5), min_df=4, max_features=200000, analyzer="char_wb")
Xcv_tr=cv.fit_transform(tr.d); Xcv_te=cv.transform(te.d)
for nb_tag, nb in [("e",ComplementNB()),("f",MultinomialNB())]:
    oof=cross_val_predict(nb,Xcv_tr,y,cv=skf,method="predict_proba",n_jobs=4)[:,1]-0.5
    print(f"[{nb_tag}] oof {((oof>0).astype(int)==y).mean():.4f} t {time.time()-t0:.0f}s", file=sys.stderr)
    nb.fit(Xcv_tr,y); oofs[nb_tag]=oof; tes[nb_tag]=nb.predict_proba(Xcv_te)[:,1]-0.5

names=list(oofs.keys())
print("models", names, file=sys.stderr)
# weight search via coordinate-style: try simple averages of subsets
best=(0,None)
for r in range(2,len(names)+1):
    for combo in itertools.combinations(names,r):
        oof=np.mean([oofs[n] for n in combo],axis=0)
        acc=((oof>0).astype(int)==y).mean()
        if acc>best[0]: best=(acc,combo)
print("best subset", best, file=sys.stderr)
oof_all=np.mean(list(oofs.values()),axis=0)
print("all mean", ((oof_all>0).astype(int)==y).mean(), file=sys.stderr)

# also fine weight grid on best 4 models
best4 = best[1] if len(best[1])>=3 else names[:4]
import itertools as it
bw=(0,None)
ws=[0.5,1,1.5,2]
for ws_ in it.product(ws, repeat=len(best4)):
    if sum(ws_)==0: continue
    oof=sum(w*oofs[n] for w,n in zip(ws_,best4))/sum(ws_)
    acc=((oof>0).astype(int)==y).mean()
    if acc>bw[0]: bw=(acc,ws_)
print("best weights", best4, bw, file=sys.stderr)

combo=best[1]; te_df=np.mean([tes[n] for n in combo],axis=0)
pred=(te_df>0).astype(int)
out=pd.DataFrame({"id":te.id.values,"label":pred})
out.to_csv("outputs/submission.csv",index=False)
print("wrote",out.shape,"combo",combo,"dist",dict(out.label.value_counts()),file=sys.stderr)
print("total",time.time()-t0,file=sys.stderr)

import pandas as pd, numpy as np, re, sys, time, itertools
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.calibration import CalibratedClassifierCV
from scipy.sparse import hstack

t0=time.time()
tr=pd.read_csv("train.csv"); te=pd.read_csv("test.csv")

def norm(s):
    s=str(s).lower(); s=re.sub(r"[^가-힣a-z0-9]"," ",s); return s
def norm2(s):
    s=str(s).lower()
    s=re.sub(r"([ㅋㅎㅠㅜ])\1+","\\1",s)
    s=re.sub(r"([가-힣a-z0-9!?~^])\1{2,}","\\1\\1",s)
    s=re.sub(r"[^가-힣a-z0-9!?~]"," ",s); return s

tr["d1"]=tr.document.map(norm); te["d1"]=te.document.map(norm)
tr["d2"]=tr.document.map(norm2); te["d2"]=te.document.map(norm2)
y=tr.label.values
skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

oofs, tes = {}, {}
def add(tag, dcol, vec, clf, method="decision_function", calibrate=False):
    Xtr=vec.fit_transform(tr[dcol]); Xte=vec.transform(te[dcol])
    if calibrate:
        # use cv-calibrated SGD for probability-like scores
        base=clf
        oof=cross_val_predict(base,Xtr,y,cv=skf,method="decision_function",n_jobs=4)
        base.fit(Xtr,y); s=base.decision_function(Xte)
    else:
        oof=cross_val_predict(clf,Xtr,y,cv=skf,method=method,n_jobs=4)
        if method=="predict_proba": oof=oof[:,1]-0.5
        clf.fit(Xtr,y)
        s=clf.decision_function(Xte) if hasattr(clf,"decision_function") else clf.predict_proba(Xte)[:,1]-0.5
    acc=((oof>0).astype(int)==y).mean()
    print(f"[{tag}] oof {acc:.4f} {Xtr.shape} t {time.time()-t0:.0f}s",file=sys.stderr)
    oofs[tag]=oof; tes[tag]=s

LR=lambda C:LogisticRegression(C=C,max_iter=1000,n_jobs=1)
add("a","d1",TfidfVectorizer(ngram_range=(2,5),min_df=4,sublinear_tf=True,max_features=200000,analyzer="char_wb"),LR(3.0))
add("b","d1",TfidfVectorizer(ngram_range=(2,5),min_df=2,sublinear_tf=True,max_features=300000,analyzer="char_wb"),LR(3.0))
add("g","d2",TfidfVectorizer(ngram_range=(2,5),min_df=3,sublinear_tf=True,max_features=300000,analyzer="char_wb"),LR(3.0))
add("h","d2",TfidfVectorizer(ngram_range=(2,5),min_df=4,sublinear_tf=True,max_features=200000,analyzer="char_wb"),LR(3.0))
# word+char combo (norm2)
vw=TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True,max_features=80000,analyzer="word")
Xw_tr=vw.fit_transform(tr["d2"]); Xw_te=vw.transform(te["d2"])
vc=TfidfVectorizer(ngram_range=(2,5),min_df=4,sublinear_tf=True,max_features=200000,analyzer="char_wb")
Xc_tr=vc.fit_transform(tr["d2"]); Xc_te=vc.transform(te["d2"])
Xcc_tr=hstack([Xw_tr,Xc_tr]).tocsr(); Xcc_te=hstack([Xw_te,Xc_te]).tocsr()
clfd=LR(2.0)
oof_d=cross_val_predict(clfd,Xcc_tr,y,cv=skf,method="decision_function",n_jobs=4)
print(f"[d] oof {((oof_d>0).astype(int)==y).mean():.4f} t {time.time()-t0:.0f}s",file=sys.stderr)
clfd.fit(Xcc_tr,y); oofs["d"]=oof_d; tes["d"]=clfd.decision_function(Xcc_te)
# NB on char counts (norm2)
cv=CountVectorizer(ngram_range=(2,5),min_df=4,max_features=200000,analyzer="char_wb")
Xcv_tr=cv.fit_transform(tr["d2"]); Xcv_te=cv.transform(te["d2"])
for nb_tag,nb in [("e",ComplementNB()),("f",MultinomialNB())]:
    oof=cross_val_predict(nb,Xcv_tr,y,cv=skf,method="predict_proba",n_jobs=4)[:,1]-0.5
    print(f"[{nb_tag}] oof {((oof>0).astype(int)==y).mean():.4f} t {time.time()-t0:.0f}s",file=sys.stderr)
    nb.fit(Xcv_tr,y); oofs[nb_tag]=oof; tes[nb_tag]=nb.predict_proba(Xcv_te)[:,1]-0.5
# SGD hinge on char (norm2) for diversity
sgd=SGDClassifier(loss="hinge",alpha=1e-5,max_iter=5,tol=1e-3,random_state=42,n_jobs=1)
add("s","d2",TfidfVectorizer(ngram_range=(2,5),min_df=4,sublinear_tf=True,max_features=300000,analyzer="char_wb"),sgd,calibrate=True)

names=list(oofs.keys())
print("models",names,file=sys.stderr)
best=(0,None)
for r in range(2,len(names)+1):
    for combo in itertools.combinations(names,r):
        oof=np.mean([oofs[n] for n in combo],axis=0)
        acc=((oof>0).astype(int)==y).mean()
        if acc>best[0]: best=(acc,combo)
print("best subset",best,file=sys.stderr)
oof_all=np.mean(list(oofs.values()),axis=0)
print("all mean",((oof_all>0).astype(int)==y).mean(),file=sys.stderr)
sub=best[1]
ws=[0.5,1,1.5,2,2.5,3]
bw=(0,None)
for ws_ in itertools.product(ws,repeat=len(sub)):
    if sum(ws_)==0: continue
    oof=sum(w*oofs[n] for w,n in zip(ws_,sub))/sum(ws_)
    acc=((oof>0).astype(int)==y).mean()
    if acc>bw[0]: bw=(acc,ws_)
print("best weights",sub,bw,file=sys.stderr)

combo=best[1]; ws_=bw[1]
te_df=sum(w*tes[n] for w,n in zip(ws_,combo))/sum(ws_)
pred=(te_df>0).astype(int)
out=pd.DataFrame({"id":te.id.values,"label":pred})
out.to_csv("outputs/submission.csv",index=False)
print("wrote",out.shape,"combo",combo,"weights",ws_,"dist",dict(out.label.value_counts()),file=sys.stderr)
print("total",time.time()-t0,file=sys.stderr)

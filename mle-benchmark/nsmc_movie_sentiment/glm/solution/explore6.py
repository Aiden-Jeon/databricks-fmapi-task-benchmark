import pandas as pd, numpy as np, re, sys, time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

t0=time.time()
tr=pd.read_csv("train.csv"); te=pd.read_csv("test.csv")

# norm2: keep punctuation that signals sentiment (!, ?), collapse repeated chars
def norm2(s):
    s=str(s).lower()
    s=re.sub(r"([ㅋㅎㅠㅜ])\1+","\\1",s)  # collapse repeated emoticon chars
    s=re.sub(r"([가-힣a-z0-9!?~^])\1{2,}","\\1\\1",s)  # limit repeats to 2
    s=re.sub(r"[^가-힣a-z0-9!?~]"," ",s)
    return s

tr["d"]=tr.document.map(norm2); te["d"]=te.document.map(norm2)
y=tr.label.values
skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

for ng,mf in [((2,5),200000),((2,5),300000),((3,5),200000)]:
    vec=TfidfVectorizer(ngram_range=ng,min_df=3,sublinear_tf=True,max_features=mf,analyzer="char_wb")
    Xtr=vec.fit_transform(tr.d)
    for C in [3.0,4.0]:
        clf=LogisticRegression(C=C,max_iter=1000,n_jobs=1)
        oof=cross_val_predict(clf,Xtr,y,cv=skf,method="decision_function",n_jobs=4)
        acc=((oof>0).astype(int)==y).mean()
        print(f"norm2 char{ng} C={C} oof {acc:.4f} {Xtr.shape} t {time.time()-t0:.0f}s",file=sys.stderr)

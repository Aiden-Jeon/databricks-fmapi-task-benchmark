import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

DATA = "/tmp/kmle/M3_t23_korfin_asc_full_20260804_033756/task"
RS = 42


def aw(sentence, aspect, b=40, a=40):
    idx = sentence.find(aspect)
    if idx == -1:
        return sentence, "", ""
    s = max(0, idx - b); e = min(len(sentence), idx + len(aspect) + a)
    return sentence[s:e], sentence[s:idx], sentence[idx + len(aspect):e]


def addf(df):
    df = df.copy()
    df["sentence"] = df["sentence"].astype(str)
    df["aspect"] = df["aspect"].astype(str)
    c, l, r = [], [], []
    for _, row in df.iterrows():
        x = aw(row["sentence"], row["aspect"])
        c.append(x[0]); l.append(x[1]); r.append(x[2])
    df["ctx"] = c; df["left"] = l; df["right"] = r
    df["combo"] = df["ctx"] + " [ASP] " + df["aspect"] + " [ASP] " + df["sentence"]
    df["la"] = df["left"] + " [ASP] " + df["aspect"]
    df["ar"] = df["aspect"] + " [ASP] " + df["right"]
    return df


def bm(tr, te, col, an="char_wb", ng=(1,5), md=2):
    v = TfidfVectorizer(analyzer=an, ngram_range=ng, min_df=md, max_df=0.95,
                        sublinear_tf=True, lowercase=False)
    return v.fit_transform(tr[col]), v.transform(te[col])


def cv5(Xtr, y, model):
    skf = StratifiedKFold(5, shuffle=True, random_state=RS)
    oof = np.zeros(len(y))
    for tr, va in skf.split(Xtr, y):
        import copy
        m = copy.deepcopy(model)
        m.fit(Xtr[tr], y[tr])
        oof[va] = m.predict(Xtr[va])
    return f1_score(y, oof, average="macro")


tr = pd.read_csv(f"{DATA}/train.csv")
te = pd.read_csv(f"{DATA}/test.csv")
tr = addf(tr); te = addf(te)
le = LabelEncoder(); y = le.fit_transform(tr["label"])
lr = lambda C: LogisticRegression(C=C, max_iter=3000, class_weight="balanced",
                                   n_jobs=-1, random_state=RS)

print("=== feature ablation ===")
configs = [
    [("combo","char_wb",(1,5),2)],
    [("combo","char_wb",(1,5),2),("combo","word",(1,2),2)],
    [("combo","char_wb",(1,5),2),("combo","word",(1,2),2),("la","char_wb",(1,4),2)],
    [("combo","char_wb",(1,5),2),("combo","word",(1,2),2),("la","char_wb",(1,4),2),("ar","char_wb",(1,4),2)],
    [("combo","char_wb",(1,5),2),("la","char_wb",(1,4),2),("ar","char_wb",(1,4),2)],
    [("combo","char_wb",(1,6),2),("combo","word",(1,2),2),("la","char_wb",(1,4),2),("ar","char_wb",(1,4),2)],
    [("combo","char_wb",(1,5),2),("combo","word",(1,2),2),("la","char_wb",(1,4),2),("ar","char_wb",(1,4),2),("left","char_wb",(1,4),1),("right","char_wb",(1,4),1)],
    [("combo","char_wb",(2,5),2),("combo","word",(1,2),2),("la","char_wb",(2,4),2),("ar","char_wb",(2,4),2)],
]
for cfg in configs:
    Xp = [bm(tr, te, *c) for c in cfg]
    Xtr = hstack([x[0] for x in Xp]).tocsr()
    sc = cv5(Xtr, y, lr(1.0))
    print(f"{cfg[0][0]} n={len(cfg)} {cfg[0][2]} -> {sc:.4f}")

print("=== C values ===")
Xp = [bm(tr, te, *c) for c in [("combo","char_wb",(1,5),2),("combo","word",(1,2),2),("la","char_wb",(1,4),2),("ar","char_wb",(1,4),2)]]
Xtr = hstack([x[0] for x in Xp]).tocsr()
for C in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
    print(f"C={C}: {cv5(Xtr, y, lr(C)):.4f}")

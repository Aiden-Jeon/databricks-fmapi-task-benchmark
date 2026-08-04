"""Shared data loading / feature engineering for KorFin-ASC."""
import os
import re
import numpy as np
import pandas as pd

TASK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
MASK = " @ASP@ "

# ---- Hangul jamo decomposition (pure python, no external data) ----
CHO = list("\u3131\u3132\u3134\u3137\u3138\u3139\u3141\u3142\u3143\u3145\u3146"
           "\u3147\u3148\u3149\u314a\u314b\u314c\u314d\u314e")
JUNG = list("\u314f\u3150\u3151\u3152\u3153\u3154\u3155\u3156\u3157\u3158\u3159"
            "\u315a\u315b\u315c\u315d\u315e\u315f\u3160\u3161\u3162\u3163")
JONG = [""] + list("\u3131\u3132\u3133\u3134\u3135\u3136\u3137\u3139\u313a\u313b"
                   "\u313c\u313d\u313e\u313f\u3140\u3141\u3142\u3144\u3145\u3146"
                   "\u3147\u3148\u314a\u314b\u314c\u314d\u314e")


def to_jamo(text):
    out = []
    for ch in text:
        o = ord(ch)
        if 0xAC00 <= o <= 0xD7A3:
            i = o - 0xAC00
            out.append(CHO[i // 588])
            out.append(JUNG[(i % 588) // 28])
            j = JONG[i % 28]
            out.append(j if j else "-")
        else:
            out.append(ch)
    return "".join(out)


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


def add_feats(df):
    df = df.copy()
    df["masked"] = [mask_sent(s, a) for s, a in zip(df.sentence, df.aspect)]
    df["win"] = [window(s, a, 30) for s, a in zip(df.sentence, df.aspect)]
    df["win15"] = [window(s, a, 15) for s, a in zip(df.sentence, df.aspect)]
    df["text"] = (df["masked"] + " ~~ " + df["win"]
                  + " ~~ ASPECT " + df.aspect.fillna(""))
    df["text2"] = (df["masked"] + " ~~ " + df["win"] + " ~~ " + df["win15"]
                   + " ~~ ASPECT " + df.aspect.fillna(""))
    df["jamo"] = [to_jamo(t) for t in df["text"]]
    return df


def load():
    tr = pd.read_csv(f"{TASK}/train.csv")
    te = pd.read_csv(f"{TASK}/test.csv")
    return add_feats(tr), add_feats(te)


def numeric_feats(df):
    """Dense hand-crafted numeric features."""
    s = df.sentence.fillna("")
    a = df.aspect.fillna("")
    n = len(df)
    pos = np.zeros(n)
    cnt = np.zeros(n)
    for i, (ss, aa) in enumerate(zip(s, a)):
        if aa and aa in ss:
            pos[i] = ss.index(aa) / max(len(ss), 1)
            cnt[i] = ss.count(aa)
        else:
            pos[i] = -1.0
    X = np.c_[
        s.str.len().values,
        a.str.len().values,
        pos,
        cnt,
        s.str.count("%").values,
        s.str.count(r"\d").values,
        df.id.str.split("_").str[1].astype(int).values,  # aspect index in sentence
        s.str.count(",").values,
    ].astype(np.float32)
    return X

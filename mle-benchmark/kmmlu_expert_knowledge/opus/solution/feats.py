"""Feature engineering for KMMLU 4-choice MCQ (no pretrained models)."""
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

OPTS = ["A", "B", "C", "D"]
NUMRE = re.compile(r"-?\d+(?:\.\d+)?")


def first_num(s):
    s = str(s).replace(",", "")
    m = NUMRE.search(s)
    return float(m.group()) if m else np.nan


def is_numeric_opt(s):
    s = str(s).strip().replace(",", "")
    return bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:\s*[^\s]{0,8})?", s))


def norm(s):
    return re.sub(r"\s+", "", str(s))


def char_set(s, n=3):
    s = norm(s)
    return set(s[i : i + n] for i in range(max(len(s) - n + 1, 1)))


def jac(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def cover(a, b):
    """fraction of a covered by b"""
    if not a:
        return 0.0
    return len(a & b) / len(a)


CUES = [
    "모두", "없다", "있다", "않다", "아니", "이다", "된다", "한다", "크다", "작다",
    "높다", "낮다", "많다", "적다", "관계", "무관", "필요", "가능", "불가", "증가",
    "감소", "동일", "반드시", "항상", "일반", "각각", "이상", "이하", "때문",
]


def build_features(df):
    """Return (X dense feature array of shape (n*4, F), texts list of n*4 option strings,
    qtexts list of n*4 question strings)."""
    n = len(df)
    q = df["question"].astype(str).tolist()
    opts = [[str(df[c].iloc[i]) for c in OPTS] for i in range(n)]

    rows = []
    texts = []
    qtexts = []
    for i in range(n):
        oo = opts[i]
        qq = q[i]
        qn = norm(qq)
        qs3 = char_set(qq, 3)
        qs2 = char_set(qq, 2)
        osets3 = [char_set(o, 3) for o in oo]
        osets2 = [char_set(o, 2) for o in oo]
        L = np.array([len(norm(o)) for o in oo], dtype=float)
        nw = np.array([len(str(o).split()) for o in oo], dtype=float)
        Lrank = np.argsort(np.argsort(L))
        isnum = np.array([is_numeric_opt(o) for o in oo], dtype=float)
        vals = np.array([first_num(o) for o in oo])
        allnum = isnum.all()
        vok = allnum and (~np.isnan(vals)).all()
        if vok:
            vrank = np.argsort(np.argsort(vals))
            sorted_asc = bool((np.diff(vals) > 0).all())
        else:
            vrank = np.zeros(4)
            sorted_asc = False
        # pairwise similarity among options
        sim3 = np.zeros((4, 4))
        for a in range(4):
            for b in range(4):
                if a != b:
                    sim3[a, b] = jac(osets3[a], osets3[b])
        cent = sim3.sum(1) / 3.0
        maxsim = sim3.max(1)
        # digits
        ndig = np.array([sum(ch.isdigit() for ch in o) for o in oo], dtype=float)
        for j in range(4):
            o = oo[j]
            on = norm(o)
            f = []
            # position one-hot
            f += [1.0 if j == k else 0.0 for k in range(4)]
            # length features
            f += [
                L[j],
                np.log1p(L[j]),
                L[j] - L.mean(),
                L[j] / (L.max() + 1e-9),
                float(Lrank[j]),
                float(L[j] == L.max()),
                float(L[j] == L.min()),
                nw[j],
                L.std(),
                L.mean(),
            ]
            # numeric
            f += [
                isnum[j],
                float(allnum),
                float(vrank[j]) if vok else -1.0,
                float(vok and vrank[j] in (1, 2)),
                float(vok and vrank[j] == 0),
                float(vok and vrank[j] == 3),
                float(sorted_asc),
                ndig[j],
                float(ndig[j] > 0),
            ]
            # question-option similarity
            f += [
                jac(qs3, osets3[j]),
                jac(qs2, osets2[j]),
                cover(osets3[j], qs3),
                cover(osets2[j], qs2),
            ]
            # option-option similarity
            f += [cent[j], maxsim[j], cent[j] - cent.mean(), maxsim[j] - maxsim.mean(),
                  float(cent[j] == cent.max()), float(cent[j] == cent.min())]
            # cue flags
            f += [float(c in on) for c in CUES]
            # polarity-interacted cues (neg question: find the FALSE statement)
            polarity = 1.0 if re.search("옳지|않은|아닌|틀린|먼 것|없는 것", qn) else (
                -1.0 if re.search("옳은|맞는|올바른", qn) else 0.0)
            for c in ["있다", "없다", "않다", "아니", "모두", "항상", "반드시",
                      "증가", "감소", "무관", "관계없", "이다", "된다"]:
                f.append(polarity * float(c in on))
            f.append(polarity)
            # ending
            f += [
                float(on.endswith("다")),
                float(on.endswith("다.")),
                float(on.endswith("것")),
                float(on.endswith("음")),
                float(on.endswith("함")),
                float(bool(re.search(r"[%℃㎜㎝㎡㎥]", o))),
            ]
            # question meta (same for all opts but interacts in trees)
            f += [
                float(len(qn)),
                float("옳지" in qn or "않은" in qn or "아닌" in qn or "틀린" in qn),
                float("옳은" in qn or "맞는" in qn),
                float("가장" in qn),
                float("모두" in qn),
                float(bool(re.search(r"\d", qq))),
            ]
            rows.append(f)
            texts.append(o)
            qtexts.append(qq)
    X = np.asarray(rows, dtype=np.float32)
    X = np.nan_to_num(X, nan=-1.0, posinf=0.0, neginf=0.0)
    return X, texts, qtexts

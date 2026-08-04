"""Feature building for KLUE-NLI (scikit-learn only, CPU).

Sparse text blocks:
  h_w / h_c   : hypothesis, particle-stripped word n-grams + char n-grams
  p_w / p_c   : premise, idem
  nov_w/nov_c : the *novel* part of the hypothesis (tokens that do not appear in
                the premise) -- this is where negation / "only" / new entities
                that decide the NLI label live
  shd_w       : the part of the hypothesis shared with the premise
Plus 31 dense hand-crafted overlap / negation / length features.
"""
import re
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

LABELS = ["entailment", "neutral", "contradiction"]
L2I = {l: i for i, l in enumerate(LABELS)}

# crude Korean particle/ending stripper (no external morph analyser available)
PARTICLES = [
    "으로는", "에서는", "에서도", "에게는", "으로써", "이라고", "라고는",
    "에서", "에게", "으로", "까지", "부터", "보다", "처럼", "이라", "라고", "마다",
    "만큼", "조차", "이나", "든지", "하고",
    "은", "는", "이", "가", "을", "를", "의", "에", "도", "만", "과", "와", "로", "야",
    "뿐", "라", "께", "님",
]

NEG_PAT = re.compile(r"않|없|못|아니|안\s|아무|전혀|절대|결코|반대|무시|부정|거짓|틀리|다르")
ONLY_PAT = re.compile(r"만\b|뿐|오직|유일|only|단지|밖에")
MODAL_PAT = re.compile(r"싶|것이다|일까|추측|아마|듯|같다|예정|계획|바란|원한|희망|생각한|모르")
NUM_PAT = re.compile(r"\d+")


def strip_particles(tok):
    for p in PARTICLES:
        if len(tok) > len(p) + 1 and tok.endswith(p):
            return tok[: -len(p)]
    return tok


def norm_text(s):
    return re.sub(r"\s+", " ", str(s).strip())


def tokens(s, stem=True):
    s = re.sub(r"[^0-9A-Za-z\uac00-\ud7a3\u3131-\u318e ]", " ", str(s))
    toks = [t for t in s.split() if t]
    return [strip_particles(t) for t in toks] if stem else toks


def char_ngrams(s, n=3):
    s = re.sub(r"\s+", "", str(s))
    return {s[i : i + n] for i in range(max(0, len(s) - n + 1))}


def diff_text(p, h):
    """Split hypothesis tokens into (novel wrt premise, shared with premise)."""
    pt = set(tokens(p))
    pc = set()
    for t in pt:
        pc |= char_ngrams(t, 2)
    novel, shared = [], []
    for t in tokens(h):
        if t in pt:
            shared.append(t)
        else:
            g = char_ngrams(t, 2)
            ov = len(g & pc) / max(1, len(g))
            (shared if ov >= 0.75 else novel).append(t)
    return " ".join(novel), " ".join(shared)


def numeric_feats(df):
    rows = []
    for p, h in zip(df["premise"].values, df["hypothesis"].values):
        pt, ht = tokens(p), tokens(h)
        spt, sht = set(pt), set(ht)
        p3, h3 = char_ngrams(p, 3), char_ngrams(h, 3)
        p2, h2 = char_ngrams(p, 2), char_ngrams(h, 2)
        inter_w, inter3, inter2 = len(spt & sht), len(p3 & h3), len(p2 & h2)
        lp, lh = len(str(p)), len(str(h))
        pnum, hnum = set(NUM_PAT.findall(str(p))), set(NUM_PAT.findall(str(h)))
        nov, sh = diff_text(p, h)
        nov_t = nov.split()
        rows.append([
            inter_w / max(1, len(sht)), inter_w / max(1, len(spt)),
            inter3 / max(1, len(h3)), inter3 / max(1, len(p3)),
            inter2 / max(1, len(h2)),
            len(p3 & h3) / max(1, len(p3 | h3)),
            len(spt & sht) / max(1, len(spt | sht)),
            lh, lp, lh / max(1, lp), lh - lp,
            len(ht), len(pt), len(ht) - len(pt),
            len(nov_t), len(nov_t) / max(1, len(ht)),
            len(sh.split()) / max(1, len(ht)),
            len(NEG_PAT.findall(str(h))), len(NEG_PAT.findall(str(p))),
            len(NEG_PAT.findall(str(h))) - len(NEG_PAT.findall(str(p))),
            len(ONLY_PAT.findall(str(h))), len(ONLY_PAT.findall(str(p))),
            len(MODAL_PAT.findall(str(h))), len(MODAL_PAT.findall(str(p))),
            len(hnum - pnum), len(hnum & pnum), len(hnum), len(pnum),
            1.0 if str(h)[-2:] == str(p)[-2:] else 0.0,
            len(NEG_PAT.findall(nov)), len(ONLY_PAT.findall(nov)),
        ])
    return np.asarray(rows, dtype=np.float32)


class FeatureBuilder:
    SPECS = [
        ("h_w", "hstem", dict(analyzer="word", ngram_range=(1, 3), min_df=2, sublinear_tf=True, max_features=300000)),
        ("h_c", "h", dict(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=400000)),
        ("p_w", "pstem", dict(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=300000)),
        ("p_c", "p", dict(analyzer="char_wb", ngram_range=(2, 4), min_df=3, sublinear_tf=True, max_features=400000)),
        ("nov_w", "nov", dict(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("nov_c", "nov", dict(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=400000)),
        ("shd_w", "shd", dict(analyzer="word", ngram_range=(1, 1), min_df=2, sublinear_tf=True)),
    ]

    def __init__(self):
        self.vecs = {}

    def _prep(self, df):
        p = df["premise"].map(norm_text).values
        h = df["hypothesis"].map(norm_text).values
        nov, shd = [], []
        for a, b in zip(p, h):
            n, s = diff_text(a, b)
            nov.append(n)
            shd.append(s)
        return dict(p=p, h=h, nov=nov, shd=shd,
                    hstem=[" ".join(tokens(x)) for x in h],
                    pstem=[" ".join(tokens(x)) for x in p])

    def fit_transform(self, df):
        d = self._prep(df)
        blocks = []
        for name, key, kw in self.SPECS:
            v = TfidfVectorizer(**kw)
            blocks.append(v.fit_transform(d[key]))
            self.vecs[name] = v
        num = numeric_feats(df)
        self.scaler = StandardScaler().fit(num)
        blocks.append(sparse.csr_matrix(self.scaler.transform(num)))
        return sparse.hstack(blocks).tocsr()

    def transform(self, df):
        d = self._prep(df)
        blocks = [self.vecs[name].transform(d[key]) for name, key, _ in self.SPECS]
        blocks.append(sparse.csr_matrix(self.scaler.transform(numeric_feats(df))))
        return sparse.hstack(blocks).tocsr()

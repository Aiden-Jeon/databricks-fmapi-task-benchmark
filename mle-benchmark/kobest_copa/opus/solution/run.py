"""KoBEST COPA (causal reasoning) solution.

Two complementary families of pairwise linear models, all trained on train.csv:

1) "answer-only" models: TF-IDF (char / word-stem n-grams) of the two
   alternatives, features = vec(alt2) - vec(alt1), logistic regression.
   Captures distractor style artefacts (negation, phrasing, ...).

2) "association" models: a cause->effect co-occurrence matrix over sub-word
   units is estimated from the *correct* (premise, alternative) pairs of the
   training split, converted to PPMI and factorised with SVD.  A candidate
   alternative is scored by the association between its units and the premise
   units (with the direction given by the `question` column).  This is the
   commonsense-plausibility part of the model.

Scores of all members are z-normalised and averaged; the sign of the blend
gives the predicted label.  Everything is fold-safe: any label-dependent
statistic (co-occurrence counts, model weights) is estimated on the training
part of each CV split only.

Usage:  python run.py [data_dir] [--cv]
"""
import sys, os, re, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import RepeatedStratifiedKFold
from scipy import sparse as sp
from joblib import Parallel, delayed
from common import load, all_sentences, tokens, char_ngrams

DATA = ".."
args = [a for a in sys.argv[1:]]
DO_CV = "--cv" in args
args = [a for a in args if not a.startswith("--")]
if args:
    DATA = args[0]
OUT = os.path.join(DATA, "outputs", "submission.csv")
SEED = 0

tr, te = load(DATA)
y = tr.label.values
n_tr, n_te = len(tr), len(te)
N = n_tr + n_te
sents = all_sentences(tr, te)
prem = np.concatenate([tr.premise.values, te.premise.values])
a1 = np.concatenate([tr.alternative_1.values, te.alternative_1.values])
a2 = np.concatenate([tr.alternative_2.values, te.alternative_2.values])
qc = np.concatenate([tr.question.values, te.question.values]) == "원인"  # premise is the effect

# ===================================================================== part 1
def tfidf_block(**kw):
    v = TfidfVectorizer(**kw).fit(sents)
    return v.transform(prem), v.transform(a1), v.transform(a2)


BLK = {
    "cwb24": tfidf_block(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True),
    "c24": tfidf_block(analyzer="char", ngram_range=(2, 4), min_df=2, sublinear_tf=True),
    "cwb14": tfidf_block(analyzer="char_wb", ngram_range=(1, 4), min_df=1, sublinear_tf=True),
    "wstem": tfidf_block(analyzer=lambda s: tokens(s), min_df=1, sublinear_tf=True),
}
ANSDIFF = {k: sp.csr_matrix(v[2] - v[1]).tocsr() for k, v in BLK.items()}

NEG = re.compile(r"(안 |못 |않|없|아니|말았|그만|실패|아직)")


def cos_rows(X, Y):
    X, Y = normalize(X), normalize(Y)
    return np.asarray(X.multiply(Y).sum(axis=1)).ravel()


def jacc(a, b, n):
    sa, sb = set(char_ngrams(a, n, n)), set(char_ngrams(b, n, n))
    return len(sa & sb) / len(sa | sb) if sa and sb else 0.0


def alt_feats(i):
    A, Aw = BLK["cwb24"][i], BLK["wstem"][i]
    P, Pw = BLK["cwb24"][0], BLK["wstem"][0]
    alts = a1 if i == 1 else a2
    return {
        "cos_c": cos_rows(P, A),
        "cos_w": cos_rows(Pw, Aw),
        "jac2": np.array([jacc(p, a, 2) for p, a in zip(prem, alts)]),
        "jac3": np.array([jacc(p, a, 3) for p, a in zip(prem, alts)]),
        "len": np.array([len(a) for a in alts], float),
        "ntok": np.array([len(a.split()) for a in alts], float),
        "neg": np.array([1.0 if NEG.search(a) else 0.0 for a in alts]),
        "endda": np.array([1.0 if a.endswith("다.") else 0.0 for a in alts]),
    }


_f1, _f2 = alt_feats(1), alt_feats(2)
_d = np.column_stack([_f2[k] - _f1[k] for k in _f1])
DENSE = sp.csr_matrix(StandardScaler().fit_transform(
    np.hstack([_d, _d * qc[:, None], qc[:, None].astype(float)])))

# ===================================================================== part 2
UNIT_FN = {
    "char2": lambda s: [s[i:i + 2] for i in range(len(s) - 1)],
    "char2_stem2": lambda s: [s[i:i + 2] for i in range(len(s) - 1)]
    + [t[:2] if len(t) > 2 else t for t in tokens(s)],
    "char23": lambda s: [s[i:i + 2] for i in range(len(s) - 1)]
    + [s[i:i + 3] for i in range(len(s) - 2)],
    "stem2": lambda s: [t[:2] if len(t) > 2 else t for t in tokens(s)],
    "syl": lambda s: list(re.sub(r"[^가-힣]", "", s)),
}


def build_units(mode, min_count):
    f = UNIT_FN[mode]
    PT, A1T, A2T = [f(s) for s in prem], [f(s) for s in a1], [f(s) for s in a2]
    c = Counter()
    for L in (PT, A1T, A2T):
        for t in L:
            c.update(set(t))
    voc = {w: i for i, w in enumerate(w for w, k in c.items() if k >= min_count)}
    V = len(voc)
    bags = lambda L: [[voc[t] for t in set(x) if t in voc] for x in L]
    PB, A1B, A2B = bags(PT), bags(A1T), bags(A2T)

    def spm(bs):
        rows, cols = [], []
        for i, b in enumerate(bs):
            rows += [i] * len(b)
            cols += b
        return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(bs), V))

    return dict(PB=PB, A1B=A1B, A2B=A2B, V=V, SP=spm(PB), SA1=spm(A1B), SA2=spm(A2B))


UNITS = {}


def ppmi(M):
    tot = M.sum()
    rs = np.asarray(M.sum(1)).ravel() + 1e-9
    cs = np.asarray(M.sum(0)).ravel() + 1e-9
    D = M.tocoo()
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.log(np.maximum(D.data, 1e-12) * tot / (rs[D.row] * cs[D.col]))
    v = np.nan_to_num(np.maximum(v, 0.0))
    return sp.csr_matrix((v, (D.row, D.col)), shape=M.shape)


def assoc_features(tri, mode="char2", min_count=2, dim=100, oriented=True):
    """Fold-safe association features (cause->effect PPMI + SVD similarity)."""
    if (mode, min_count) not in UNITS:
        UNITS[(mode, min_count)] = build_units(mode, min_count)
    U_ = UNITS[(mode, min_count)]
    PB, A1B, A2B, V = U_["PB"], U_["A1B"], U_["A2B"], U_["V"]
    SP, SA1, SA2 = U_["SP"], U_["SA1"], U_["SA2"]
    rows, cols = [], []
    for i in tri:
        alt = A2B[i] if y[i] == 1 else A1B[i]
        p = PB[i]
        cause, eff = (alt, p) if (oriented and qc[i]) else (p, alt)
        for cc in cause:
            for ee in eff:
                rows.append(cc)
                cols.append(ee)
    M = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(V, V))
    if not oriented:
        M = M + M.T
    W = ppmi(M)

    def sides(SA):
        if oriented:
            C = sp.csr_matrix(np.where(qc[:, None], SA.toarray(), SP.toarray()))
            E = sp.csr_matrix(np.where(qc[:, None], SP.toarray(), SA.toarray()))
        else:
            C, E = SP, SA
        return C, E

    feats = []
    raws = []
    for SA in (SA1, SA2):
        C, E = sides(SA)
        num = np.asarray((C @ W).multiply(E).sum(1)).ravel()
        nc = np.maximum(np.asarray(C.sum(1)).ravel(), 1.0)
        ne = np.maximum(np.asarray(E.sum(1)).ravel(), 1.0)
        raws.append((num / (nc * ne), num / ne))
    feats += [raws[1][0] - raws[0][0], raws[1][1] - raws[0][1]]
    svd = TruncatedSVD(n_components=dim, random_state=SEED)
    Uc = normalize(svd.fit_transform(W))
    Ue = normalize(svd.components_.T)
    svs = []
    for SA in (SA1, SA2):
        C, E = sides(SA)
        svs.append((normalize(C @ Uc) * normalize(E @ Ue)).sum(1))
    feats.append(svs[1] - svs[0])
    X = np.nan_to_num(np.column_stack(feats))
    return StandardScaler().fit_transform(X)


# ===================================================================== members
LIN_MEMBERS = {
    "cwb24": (ANSDIFF["cwb24"], 3.0),
    "c24": (ANSDIFF["c24"], 10.0),
    "cwb14": (ANSDIFF["cwb14"], 3.0),
    "wstem": (ANSDIFF["wstem"], 3.0),
    "dense": (DENSE, 0.03),
}
ASSOC_MEMBERS = {
    "as_char2": dict(mode="char2", min_count=2, dim=100),
    "as_c2s2": dict(mode="char2_stem2", min_count=2, dim=50),
    "as_char23": dict(mode="char23", min_count=2, dim=50),
    "as_stem2u": dict(mode="stem2", min_count=2, dim=50, oriented=False),
    "as_syl": dict(mode="syl", min_count=2, dim=50),
}


def member_scores(tri, ei):
    out = {}
    for k, (X, C) in LIN_MEMBERS.items():
        m = LogisticRegression(C=C, max_iter=1000, solver="liblinear")
        m.fit(X[tri], y[tri])
        out[k] = m.decision_function(X[ei])
    for k, cfg in ASSOC_MEMBERS.items():
        X = assoc_features(tri, **cfg)
        m = LogisticRegression(C=1.0, max_iter=1000)
        m.fit(X[tri], y[tri])
        out[k] = m.decision_function(X[ei])
    return out


LIN_KEYS = list(LIN_MEMBERS)
ASSOC_KEYS = list(ASSOC_MEMBERS)
W_ASSOC = 2.0   # relative weight of the association family (flat CV optimum 1.6-2.5)
THRESH = 0.0


def blend(sc, keys_lin=LIN_KEYS, keys_as=ASSOC_KEYS, w_assoc=W_ASSOC):
    n = len(next(iter(sc.values())))
    tot = np.zeros(n)
    for k in keys_lin:
        s = sc[k]
        tot += s / (s.std() or 1.0) / len(keys_lin)
    for k in keys_as:
        s = sc[k]
        tot += w_assoc * s / (s.std() or 1.0) / len(keys_as)
    return tot


# ===================================================================== main
if DO_CV:
    folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
                 .split(np.zeros(n_tr), y))
    res = Parallel(n_jobs=4, verbose=1)(delayed(member_scores)(tri, vai) for tri, vai in folds)
    print("\n--- members ---")
    for k in LIN_KEYS + ASSOC_KEYS:
        a = [((r[k] > 0).astype(int) == y[v]).mean() for r, (t, v) in zip(res, folds)]
        print(f"{k:12s} {np.mean(a):.4f} +/- {np.std(a)/np.sqrt(len(a)):.4f}")

    def bacc(**kw):
        thr = kw.pop("thr", 0.0)
        a = [((blend(r, **kw) > thr).astype(int) == y[v]).mean() for r, (t, v) in zip(res, folds)]
        return np.mean(a), np.std(a) / np.sqrt(len(a))

    print("\n--- blends ---")
    for w in [0.0, 1.0, 2.0, 2.5, 4.0]:
        m, s = bacc(w_assoc=w)
        print(f"w_assoc={w:<6g} {m:.4f} +/- {s:.4f}")
    m, s = bacc()
    print(f"\nFINAL blend CV accuracy (5x5): {m:.4f} +/- {s:.4f}")

sc = member_scores(np.arange(n_tr), np.arange(n_tr, N))
pred = (blend(sc) > THRESH).astype(int)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
pd.DataFrame({"id": te.id.values, "label": pred}).to_csv(OUT, index=False)
print("wrote", OUT, "| label counts:", np.bincount(pred, minlength=2))

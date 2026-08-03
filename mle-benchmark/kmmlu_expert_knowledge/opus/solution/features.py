"""Feature engineering for KMMLU 4-choice task (t21_kmmlu).

All features are computed from surface properties of the question / options and
from TF-IDF statistics fitted on the *training* split only (fold-safe helpers are
provided separately in model.py).
"""
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

OPTS = ["A", "B", "C", "D"]

NEG_Q = r"옳지\s*않|않은\s*것|아닌\s*것|틀린|거리가\s*먼|부적당|부적합|잘못|해당되지|관계가\s*없"
POS_Q = r"옳은\s*것|맞는\s*것|적절한|적합한|알맞은|올바른"
ABS_W = ["모두", "항상", "반드시", "절대", "전혀", "만", "오직", "무조건"]
HEDGE_W = ["대체로", "일반적", "보통", "경우", "가능", "정도", "수도"]


def _num(s):
    s = str(s).replace(",", "").replace(" ", "")
    s = s.replace("−", "-").replace("–", "-")
    m = re.fullmatch(r"-?\d+(\.\d+)?", s)
    if m:
        return float(s)
    m = re.search(r"-?\d+(\.\d+)?", s)
    return float(m.group()) if m else np.nan


def _row_norm(M):
    n = np.sqrt(np.asarray(M.multiply(M).sum(1)).ravel())
    n[n == 0] = 1.0
    return n


def cos_pairs(A, B):
    """Row-wise cosine similarity between two sparse matrices with equal rows."""
    num = np.asarray(A.multiply(B).sum(1)).ravel()
    return num / (_row_norm(A) * _row_norm(B))


def build_features(df, vec=None):
    """Return (feature DataFrame with 4*len(df) rows, fitted vectorizer).

    Rows are ordered question-major: q0/A, q0/B, q0/C, q0/D, q1/A, ...
    """
    n = len(df)
    q = df["question"].astype(str).values
    O = [df[o].astype(str).values for o in OPTS]

    if vec is None:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                              min_df=3, sublinear_tf=True)
        vec.fit(np.concatenate([q] + O))
    Q = vec.transform(q)
    MO = [vec.transform(O[i]) for i in range(4)]

    # pairwise option similarities
    sim_oo = np.zeros((n, 4, 4))
    for i in range(4):
        for j in range(4):
            if i <= j:
                s = cos_pairs(MO[i], MO[j])
                sim_oo[:, i, j] = s
                sim_oo[:, j, i] = s
    sim_q = np.stack([cos_pairs(MO[i], Q) for i in range(4)], 1)  # n x 4

    off = sim_oo.copy()
    for i in range(4):
        off[:, i, i] = np.nan
    sim_oth_mean = np.nanmean(off, 2)
    sim_oth_max = np.nanmax(off, 2)
    sim_oth_min = np.nanmin(off, 2)

    # lengths
    L = np.stack([np.array([len(x) for x in O[i]]) for i in range(4)], 1).astype(float)
    W = np.stack([np.array([len(x.split()) for x in O[i]]) for i in range(4)], 1).astype(float)
    Lmean = L.mean(1, keepdims=True)
    Lstd = L.std(1, keepdims=True) + 1e-6
    Lrank = np.argsort(np.argsort(L, 1), 1).astype(float)
    Lsum_oth = (L.sum(1, keepdims=True) - L) / 3.0

    # numeric values
    V = np.stack([np.array([_num(x) for x in O[i]]) for i in range(4)], 1)
    allnum = (~np.isnan(V)).all(1)
    Vrank = np.full((n, 4), np.nan)
    Vratio = np.full((n, 4), np.nan)
    if allnum.any():
        sub = V[allnum]
        Vrank[allnum] = np.argsort(np.argsort(sub, 1), 1)
        med = np.median(sub, 1, keepdims=True)
        med = np.where(np.abs(med) < 1e-9, 1.0, med)
        Vratio[allnum] = sub / med
    strict_num = np.stack(
        [np.array([bool(re.fullmatch(r"-?[\d,\.]+\s*[^\d]{0,6}", str(x).strip()))
                   for x in O[i]]) for i in range(4)], 1)

    # question-level
    qlen = np.array([len(x) for x in q], dtype=float)
    isneg = np.array([bool(re.search(NEG_Q, x)) for x in q], dtype=float)
    ispos = np.array([bool(re.search(POS_Q, x)) for x in q], dtype=float)
    q_num = np.array([bool(re.search(r"\d", x)) for x in q], dtype=float)
    q_what = np.array([("무엇" in x) or ("몇" in x) for x in q], dtype=float)
    q_calc = np.array([bool(re.search(r"구하|계산|얼마|=|\?$", x)) for x in q], dtype=float)

    rows = {}

    def put(name, arr):
        rows[name] = np.asarray(arr, dtype=float).reshape(-1)

    pos = np.tile(np.arange(4), (n, 1))

    put("pos", pos)
    for k in range(4):
        put(f"pos_{k}", (pos == k))
    put("len", L)
    put("len_rel", L / Lmean)
    put("len_z", (L - Lmean) / Lstd)
    put("len_rank", Lrank)
    put("len_is_max", Lrank == 3)
    put("len_is_min", Lrank == 0)
    put("len_diff_oth", L - Lsum_oth)
    put("nwords", W)
    put("sim_q", sim_q)
    put("sim_q_rank", np.argsort(np.argsort(sim_q, 1), 1))
    put("sim_q_rel", sim_q - sim_q.mean(1, keepdims=True))
    put("sim_oth_mean", sim_oth_mean)
    put("sim_oth_mean_rank", np.argsort(np.argsort(sim_oth_mean, 1), 1))
    put("sim_oth_max", sim_oth_max)
    put("sim_oth_min", sim_oth_min)
    put("sim_oth_rel", sim_oth_mean - sim_oth_mean.mean(1, keepdims=True))
    put("vrank", np.nan_to_num(Vrank, nan=-1))
    put("v_is_min", (Vrank == 0) & allnum[:, None])
    put("v_is_max", (Vrank == 3) & allnum[:, None])
    put("v_is_mid", ((Vrank == 1) | (Vrank == 2)) & allnum[:, None])
    put("vratio", np.nan_to_num(np.clip(Vratio, -20, 20), nan=0.0))
    put("strict_num", strict_num)

    # lexical cues inside option text
    for w in ABS_W:
        put(f"abs_{w}", np.stack([np.array([w in x for x in O[i]]) for i in range(4)], 1))
    put("abs_any", np.stack([np.array([any(w in x for w in ABS_W) for x in O[i]])
                             for i in range(4)], 1))
    put("hedge_any", np.stack([np.array([any(w in x for w in HEDGE_W) for x in O[i]])
                               for i in range(4)], 1))
    put("o_neg", np.stack([np.array([("않" in x) or ("없" in x) or ("아니" in x)
                                     for x in O[i]]) for i in range(4)], 1))
    put("o_digits", np.stack([np.array([sum(c.isdigit() for c in x) for x in O[i]])
                              for i in range(4)], 1))
    put("o_paren", np.stack([np.array([("(" in x) or ("（" in x) for x in O[i]])
                             for i in range(4)], 1))
    put("o_comma", np.stack([np.array([x.count(",") for x in O[i]]) for i in range(4)], 1))
    put("o_ends_da", np.stack([np.array([x.rstrip().endswith("다") or
                                        x.rstrip().endswith("다.") for x in O[i]])
                               for i in range(4)], 1))
    put("o_in_q", np.stack([np.array([1.0 if (len(O[i][r]) > 1 and O[i][r] in q[r]) else 0.0
                                      for r in range(n)]) for i in range(4)], 1))
    put("o_dup", np.stack([np.array([(sim_oo[r, i] > 0.95).sum() - 1 for r in range(n)])
                           for i in range(4)], 1))

    # question-level broadcast
    for name, arr in [("q_len", qlen), ("q_neg", isneg), ("q_pos", ispos),
                      ("q_hasnum", q_num), ("q_what", q_what), ("q_calc", q_calc)]:
        put(name, np.repeat(arr[:, None], 4, 1))
    put("q_allnum", np.repeat(allnum.astype(float)[:, None], 4, 1))
    put("q_len_mean", np.repeat(Lmean, 4, 1))
    put("q_len_std", np.repeat(Lstd, 4, 1))
    put("q_simoo_mean", np.repeat(sim_oth_mean.mean(1, keepdims=True), 4, 1))

    # interactions with negation phrasing
    put("neg_x_lenz", ((L - Lmean) / Lstd) * isneg[:, None])
    put("neg_x_simoth", sim_oth_mean * isneg[:, None])
    put("neg_x_pos", pos * isneg[:, None])

    F = pd.DataFrame(rows)
    F["qidx"] = np.repeat(np.arange(n), 4)
    F["opt"] = np.tile(np.arange(4), n)
    return F, vec


def option_texts(df):
    """Flatten option texts and question+option texts (question-major order)."""
    q = df["question"].astype(str).values
    O = [df[o].astype(str).values for o in OPTS]
    opt_flat, qo_flat = [], []
    for r in range(len(df)):
        for i in range(4):
            opt_flat.append(O[i][r])
            qo_flat.append(q[r] + " || " + O[i][r])
    return opt_flat, qo_flat

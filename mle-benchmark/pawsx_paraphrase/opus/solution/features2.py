"""Second-stage features: fuzzy token alignment + train-graph structure.

Both are derived exclusively from the provided train.csv/test.csv text (and, for
the graph features, from train labels of the *training folds only*).
"""
import re
from collections import Counter, defaultdict

import numpy as np

import features as F

_DIGIT = re.compile(r"\d")
_LATIN = re.compile(r"[A-Za-z]")


def tri(tok):
    s = "^" + tok + "$"
    if len(s) <= 3:
        return {s}
    return {s[i : i + 3] for i in range(len(s) - 2)}


def jac(a, b):
    if not a or not b:
        return 0.0
    i = len(a & b)
    return i / (len(a) + len(b) - i)


def align_feats(s1, s2):
    """One-to-one greedy fuzzy alignment of whitespace tokens."""
    out = {}
    a = F.tokenize(F.norm(s1))
    b = F.tokenize(F.norm(s2))
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        for k in ALIGN_KEYS:
            out[k] = 0.0
        return out
    ta = [tri(t) for t in a]
    tb = [tri(t) for t in b]
    cand = []
    for i in range(na):
        for j in range(nb):
            s = jac(ta[i], tb[j])
            if s > 0.15:
                cand.append((s, i, j))
    cand.sort(key=lambda x: (-x[0], abs(x[1] - x[2])))
    ua = [False] * na
    ub = [False] * nb
    pairs = []
    for s, i, j in cand:
        if ua[i] or ub[j]:
            continue
        ua[i] = ub[j] = True
        pairs.append((i, j, s))
    pairs.sort()
    k = len(pairs)
    out["al_rate_a"] = k / na
    out["al_rate_b"] = k / nb
    out["al_unmatched_a"] = na - k
    out["al_unmatched_b"] = nb - k
    out["al_unmatched_sum"] = (na - k) + (nb - k)
    sims = np.array([p[2] for p in pairs]) if k else np.zeros(1)
    out["al_sim_mean"] = float(sims.mean())
    out["al_sim_min"] = float(sims.min())
    out["al_sim_std"] = float(sims.std())
    out["al_sim_sum_norm"] = float(sims.sum()) / max(na, nb)
    out["al_n_exact"] = float(sum(1 for p in pairs if p[2] > 0.999))
    out["al_exact_rate"] = out["al_n_exact"] / max(na, nb)
    out["al_n_lowsim"] = float(sum(1 for p in pairs if p[2] < 0.5))
    out["al_lowsim_rate"] = out["al_n_lowsim"] / max(1, k)

    # displacement / order statistics of the alignment
    js = np.array([p[1] for p in pairs], dtype=float)
    isx = np.array([p[0] for p in pairs], dtype=float)
    # normalise index scales when lengths differ
    disp = js - isx
    rel = js / max(1, nb - 1) - isx / max(1, na - 1)
    out["al_disp_absmean"] = float(np.abs(disp).mean()) if k else 0.0
    out["al_disp_absmax"] = float(np.abs(disp).max()) if k else 0.0
    out["al_disp_std"] = float(disp.std()) if k else 0.0
    out["al_reldisp_absmean"] = float(np.abs(rel).mean()) if k else 0.0
    out["al_reldisp_absmax"] = float(np.abs(rel).max()) if k else 0.0
    out["al_n_disp2"] = float((np.abs(disp) >= 2).sum()) if k else 0.0
    out["al_n_disp3"] = float((np.abs(disp) >= 3).sum()) if k else 0.0
    inv = 0
    for x in range(k):
        for yy in range(x + 1, k):
            if js[yy] < js[x]:
                inv += 1
    out["al_inv"] = float(inv)
    out["al_inv_rate"] = inv / (k * (k - 1) / 2) if k > 1 else 0.0
    out["al_tau"] = 1 - 2 * out["al_inv_rate"]
    # a classic PAWS swap: exactly two aligned tokens with opposite big shifts
    big = [(int(isx[x]), int(js[x]), disp[x]) for x in range(k) if abs(disp[x]) >= 1]
    out["al_n_big"] = float(len(big))
    swapish = 0.0
    if len(big) == 2 and big[0][2] * big[1][2] < 0:
        swapish = 1.0
    out["al_swapish"] = swapish

    # properties of the tokens that moved / were substituted
    moved_tok = [a[int(isx[x])] for x in range(k) if abs(disp[x]) >= 1]
    subs = [(a[p[0]], b[p[1]]) for p in pairs if p[2] < 0.999]
    out["al_n_subs"] = float(len(subs))
    out["al_subs_rate"] = len(subs) / max(1, k)
    if subs:
        # same stem, different ending => grammatical variation only
        same_stem = sum(1 for x, yy in subs if F.stem(x) == F.stem(yy))
        pref = sum(1 for x, yy in subs if x[:2] == yy[:2])
        out["al_subs_samestem_rate"] = same_stem / len(subs)
        out["al_subs_pref_rate"] = pref / len(subs)
        out["al_subs_lendiff"] = float(np.mean([abs(len(x) - len(yy)) for x, yy in subs]))
    else:
        out["al_subs_samestem_rate"] = 0.0
        out["al_subs_pref_rate"] = 0.0
        out["al_subs_lendiff"] = 0.0
    out["al_moved_has_digit"] = float(sum(1 for t in moved_tok if _DIGIT.search(t)))
    out["al_moved_has_latin"] = float(sum(1 for t in moved_tok if _LATIN.search(t)))
    unm_a = [a[i] for i in range(na) if not ua[i]]
    unm_b = [b[j] for j in range(nb) if not ub[j]]
    unm = unm_a + unm_b
    out["al_unm_meanlen"] = float(np.mean([len(t) for t in unm])) if unm else 0.0
    out["al_unm_maxlen"] = float(max((len(t) for t in unm), default=0))
    out["al_unm_digit"] = float(sum(1 for t in unm if _DIGIT.search(t)))
    out["al_unm_latin"] = float(sum(1 for t in unm if _LATIN.search(t)))

    # order of numbers and latin tokens (entity swaps)
    for tag, rx in (("num", re.compile(r"\d+")), ("lat", re.compile(r"[A-Za-z]{2,}"))):
        xa = rx.findall(str(s1).lower())
        xb = rx.findall(str(s2).lower())
        out[f"al_{tag}_seq_eq"] = float(xa == xb)
        out[f"al_{tag}_bag_eq"] = float(sorted(xa) == sorted(xb))
        out[f"al_{tag}_swap"] = float(sorted(xa) == sorted(xb) and xa != xb)
        ca, cb = Counter(xa), Counter(xb)
        out[f"al_{tag}_ms_diff"] = float(sum(((ca - cb) + (cb - ca)).values()))
        out[f"al_{tag}_n"] = float(max(len(xa), len(xb)))
    return out


_probe = align_feats("가 나 다", "나 가 다")
ALIGN_KEYS = list(_probe.keys())


def build_align_matrix(df, verbose=True):
    s1 = df["sentence1"].tolist()
    s2 = df["sentence2"].tolist()
    rows = []
    for i in range(len(s1)):
        r = align_feats(s1[i], s2[i])
        rows.append([r.get(k, 0.0) for k in ALIGN_KEYS])
        if verbose and (i + 1) % 5000 == 0:
            print(f"  align {i+1}/{len(s1)}", flush=True)
    return np.asarray(rows, dtype=np.float32)


# --------------------------------------------------------------- graph feats
GRAPH_KEYS = [
    "g_direct", "g_deg_a", "g_deg_b", "g_deg_min", "g_deg_max",
    "g_pos_deg_a", "g_pos_deg_b", "g_neg_deg_a", "g_neg_deg_b",
    "g_vote_pp", "g_vote_pn", "g_vote_nn", "g_ncommon",
    "g_vote_score", "g_has_evidence",
]


def build_graph(pairs_labels):
    """pairs_labels: iterable of (s1, s2, label) from the training split only."""
    adj = defaultdict(dict)
    for a, b, lab in pairs_labels:
        if not a or not b:
            continue
        adj[a][b] = lab
        adj[b][a] = lab
    return adj


def graph_row(adj, a, b):
    o = dict.fromkeys(GRAPH_KEYS, 0.0)
    na = adj.get(a, {})
    nb = adj.get(b, {})
    o["g_direct"] = float(na[b]) if b in na else -1.0
    o["g_deg_a"], o["g_deg_b"] = float(len(na)), float(len(nb))
    o["g_deg_min"] = min(o["g_deg_a"], o["g_deg_b"])
    o["g_deg_max"] = max(o["g_deg_a"], o["g_deg_b"])
    o["g_pos_deg_a"] = float(sum(1 for v in na.values() if v == 1))
    o["g_pos_deg_b"] = float(sum(1 for v in nb.values() if v == 1))
    o["g_neg_deg_a"] = float(sum(1 for v in na.values() if v == 0))
    o["g_neg_deg_b"] = float(sum(1 for v in nb.values() if v == 0))
    pp = pn = nn = 0
    common = set(na) & set(nb)
    common.discard(a)
    common.discard(b)
    for c in common:
        la, lb = na[c], nb[c]
        if la == 1 and lb == 1:
            pp += 1
        elif la == 0 and lb == 0:
            nn += 1
        else:
            pn += 1
    o["g_vote_pp"], o["g_vote_pn"], o["g_vote_nn"] = float(pp), float(pn), float(nn)
    o["g_ncommon"] = float(len(common))
    o["g_vote_score"] = (pp - pn) / len(common) if common else 0.0
    o["g_has_evidence"] = float(bool(common) or b in na)
    return [o[k] for k in GRAPH_KEYS]


def build_graph_matrix(adj, s1_list, s2_list):
    return np.asarray([graph_row(adj, a, b) for a, b in zip(s1_list, s2_list)],
                      dtype=np.float32)

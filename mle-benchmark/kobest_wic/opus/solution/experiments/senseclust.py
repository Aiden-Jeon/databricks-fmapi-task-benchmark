"""Per-word constrained sense clustering (signed Ising / correlation clustering)
plus kernel label-propagation features.

For each target word we gather ALL of its contexts (train + test), build a
similarity graph, inject train pair labels as hard must-link / cannot-link
constraints, and find a 2-way partition maximising sum_{i<j} W_ij * s_i * s_j.
A pair is then predicted "same sense" iff both contexts land in the same part.
"""
import numpy as np


def ising_partition(W, rng, restarts=30, iters=60):
    n = W.shape[0]
    if n <= 1:
        return np.ones(n)
    best, best_s = -np.inf, np.ones(n)
    inits = []
    # spectral init
    try:
        vals, vecs = np.linalg.eigh(W)
        inits.append(np.where(vecs[:, -1] >= 0, 1.0, -1.0))
        inits.append(np.where(vecs[:, -1] >= np.median(vecs[:, -1]), 1.0, -1.0))
    except Exception:
        pass
    for _ in range(restarts):
        inits.append(rng.choice([-1.0, 1.0], size=n))
    for s0 in inits:
        s = s0.copy()
        for _ in range(iters):
            changed = False
            for i in rng.permutation(n):
                h = W[i] @ s
                ns = 1.0 if h >= 0 else -1.0
                if ns != s[i]:
                    s[i] = ns
                    changed = True
            if not changed:
                break
        sc = 0.5 * (s @ W @ s)
        if sc > best:
            best, best_s = sc, s.copy()
    return best_s


def sense_features(words_all, sim_fn, pairs_labeled, pairs_query, seed=0,
                   mu_mode='median', big=50.0, restarts=25):
    """
    words_all: dict word -> list of context ids (global indices)
    sim_fn(idx_array) -> full sim matrix among those ids
    pairs_labeled: list of (word, cid_a, cid_b, label)   -> constraints
    pairs_query:   list of (word, cid_a, cid_b)          -> rows needing features
    returns dict of feature arrays aligned with pairs_query
    """
    rng = np.random.default_rng(seed)
    cons = {}
    for w, a, b, l in pairs_labeled:
        cons.setdefault(w, []).append((a, b, l))

    out_same = np.zeros(len(pairs_query))
    out_h1 = np.zeros(len(pairs_query))
    out_h2 = np.zeros(len(pairs_query))
    out_nc = np.zeros(len(pairs_query))
    out_bal = np.zeros(len(pairs_query))

    # group queries by word
    qidx = {}
    for i, (w, a, b) in enumerate(pairs_query):
        qidx.setdefault(w, []).append(i)

    cache = {}
    for w, qs in qidx.items():
        ids = words_all[w]
        pos = {c: k for k, c in enumerate(ids)}
        S = sim_fn(np.array(ids))
        n = len(ids)
        off = np.median(S[np.triu_indices(n, 1)]) if n > 1 else 0.0
        W = S - off
        np.fill_diagonal(W, 0.0)
        sd = W[np.triu_indices(n, 1)].std() if n > 1 else 1.0
        if sd > 0:
            W = W / sd
        cl = cons.get(w, [])
        for a, b, l in cl:
            if a in pos and b in pos:
                i, j = pos[a], pos[b]
                v = big if l == 1 else -big
                W[i, j] = v
                W[j, i] = v
        key = (w, len(cl))
        s = cache.get(key)
        if s is None:
            s = ising_partition(W, rng, restarts=restarts)
            cache[key] = s
        H = W @ s
        for i in qs:
            _, a, b = pairs_query[i]
            ia, ib = pos[a], pos[b]
            out_same[i] = 1.0 if s[ia] == s[ib] else 0.0
            out_h1[i] = abs(H[ia]) * s[ia] * s[ib]
            out_h2[i] = min(abs(H[ia]), abs(H[ib])) * (1 if s[ia] == s[ib] else -1)
            out_nc[i] = len(cl)
            out_bal[i] = abs(s.mean())
    return {'sc_same': out_same, 'sc_h1': out_h1, 'sc_h2': out_h2,
            'sc_ncons': out_nc, 'sc_bal': out_bal}


def lp_features(words_all, sim_fn, pairs_labeled, pairs_query, temp=0.05):
    """Kernel label propagation: if c1~a and c2~b and (a,b) labelled l, vote l."""
    cons = {}
    for w, a, b, l in pairs_labeled:
        cons.setdefault(w, []).append((a, b, l))
    qidx = {}
    for i, (w, a, b) in enumerate(pairs_query):
        qidx.setdefault(w, []).append(i)
    lp = np.zeros(len(pairs_query))
    lp_hard = np.zeros(len(pairs_query))
    nn_same = np.zeros(len(pairs_query))
    for w, qs in qidx.items():
        ids = words_all[w]
        pos = {c: k for k, c in enumerate(ids)}
        S = sim_fn(np.array(ids))
        cl = [(a, b, l) for a, b, l in cons.get(w, []) if a in pos and b in pos]
        if not cl:
            continue
        for i in qs:
            _, a, b = pairs_query[i]
            ia, ib = pos[a], pos[b]
            num = 0.0
            den = 0.0
            bestw, bestl = -1e9, 0.0
            for (x, y, l) in cl:
                ix, iy = pos[x], pos[y]
                if ix in (ia, ib) or iy in (ia, ib):
                    continue
                sgn = 2.0 * l - 1.0
                for (p, q) in ((ix, iy), (iy, ix)):
                    ww = np.exp((S[ia, p] + S[ib, q]) / temp)
                    num += ww * sgn
                    den += ww
                    m = min(S[ia, p], S[ib, q])
                    if m > bestw:
                        bestw, bestl = m, sgn
            if den > 0:
                lp[i] = num / den
            lp_hard[i] = bestl
            nn_same[i] = bestw
    return {'lp': lp, 'lp_hard': lp_hard, 'lp_nnsim': nn_same}

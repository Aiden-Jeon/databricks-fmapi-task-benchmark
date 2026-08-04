"""Per-word exchangeable count model.

The dataset is built per target word: each word contributes a small number of
pairs whose positive/negative counts are close to balanced (strongly
under-dispersed relative to a binomial).  Estimating the distribution of the
number of positives per word from train.csv therefore gives real information
about the labels of the held-out pairs of the same word.

Model
-----
For a word with T pairs in total (train + test rows of that word, counts are
observable), let m be its total number of positive pairs with prior P_T(m)
estimated from words that lie entirely inside train.csv.  Conditional on m all
C(T, m) label configurations are equally likely (exchangeability).  Given the k
observed positives among the n_tr labelled pairs, the posterior over the vector
z of unknown test labels is

    P(z) ~ P_T(k + sum z) / C(T, k + sum z) * prod_i LR_i^{z_i}

where LR_i = p_i / (1 - p_i) is the likelihood ratio supplied by a text
classifier (optional).
"""
import numpy as np
from math import comb
from collections import defaultdict


def fit_prior(train_df, total_counts, min_words=15, smooth=1.0):
    """Estimate P_T(m) from words fully contained in train_df.

    total_counts: Series word -> T (train + test pair count)
    Returns dict T -> np.array of length T+1.
    """
    g = train_df.groupby('word').label.agg(['sum', 'count'])
    T = total_counts.reindex(g.index)
    full = g[(T.values == g['count'].values)]          # no held-out rows
    by_T = defaultdict(list)
    for m, t in zip(full['sum'].values, full['count'].values):
        by_T[int(t)].append(int(m))

    # pooled distribution of the deviation from perfect balance, used to
    # back off for pair counts with too few observed words
    dev = []
    for t, ms in by_T.items():
        if t >= 3:
            for m in ms:
                dev.append((m - t / 2.0) / np.sqrt(t))
    dev = np.array(dev) if dev else np.array([0.0])

    prior = {}
    Ts = set(int(x) for x in total_counts.dropna().unique() if x >= 1)
    for t in sorted(Ts):
        ms = by_T.get(t, [])
        if len(ms) >= min_words:
            p = np.full(t + 1, smooth, dtype=float)
            for m in ms:
                p[m] += 1.0
        else:
            # back off: map pooled deviation distribution onto this T
            p = np.full(t + 1, 0.5, dtype=float)
            grid = (np.arange(t + 1) - t / 2.0) / np.sqrt(max(t, 1))
            # kernel density of dev evaluated on the grid
            h = 0.35
            for gv in range(t + 1):
                p[gv] += np.exp(-0.5 * ((dev - grid[gv]) / h) ** 2).sum()
        prior[t] = p / p.sum()
    return prior


def _hyper_lik(T, n, k):
    """P(k positives in n draws | m positives out of T) for m = 0..T."""
    out = np.zeros(T + 1)
    den = comb(T, n)
    for m in range(T + 1):
        if k <= m and (n - k) <= (T - m):
            out[m] = comb(m, k) * comb(T - m, n - k) / den
    return out


def fit_prior_em(train_df, total_counts, min_words=30, iters=400,
                 smooth=0.5, pool_min_T=6):
    """Prior over the per-word number of positives, estimated by EM.

    Every word contributes a partial observation (k positives among its n
    labelled pairs out of T total pairs); the likelihood is hypergeometric.
    Words are grouped by T.  Groups with enough words get a non-parametric
    prior, sparse groups fall back to a discretised Gaussian centred on T/2
    whose dispersion factor c (relative to the binomial sd sqrt(T)/2) is fitted
    by pooled maximum likelihood.
    """
    g = train_df.groupby('word').label.agg(['sum', 'count'])
    g['T'] = total_counts.reindex(g.index).values
    g = g.dropna()
    g['T'] = g['T'].astype(int)
    g = g[g['T'] >= g['count']]

    def gauss_prior(T, c):
        m = np.arange(T + 1)
        sd = max(c * np.sqrt(T) / 2.0, 0.25)
        p = np.exp(-0.5 * ((m - T / 2.0) / sd) ** 2)
        return p / p.sum()

    # pooled dispersion factor from the sparse (large T) groups
    pool = g[g['T'] >= pool_min_T]
    best_c, best_ll = 1.0, -np.inf
    if len(pool) >= 5:
        for c in np.arange(0.15, 1.55, 0.05):
            ll = 0.0
            for T, k, n in zip(pool['T'], pool['sum'], pool['count']):
                p = gauss_prior(int(T), c)
                v = float(p @ _hyper_lik(int(T), int(n), int(k)))
                ll += np.log(max(v, 1e-12))
            if ll > best_ll:
                best_ll, best_c = ll, c
    prior = {}
    Ts = sorted(set(int(x) for x in total_counts.dropna().unique() if x >= 1))
    for T in Ts:
        sub = g[g['T'] == T]
        base = gauss_prior(T, best_c)
        if len(sub) >= min_words:
            L = np.array([_hyper_lik(T, int(n), int(k))
                          for k, n in zip(sub['sum'], sub['count'])])
            P = base.copy()
            for _ in range(iters):
                R = L * P
                s = R.sum(1, keepdims=True)
                s[s == 0] = 1.0
                R = R / s
                P = R.sum(0) + smooth * base * len(sub) * 0.1
                P = np.maximum(P, 1e-9)
                P = P / P.sum()
            prior[T] = P
        else:
            prior[T] = base
    prior['_c'] = best_c
    return prior


def posterior_marginals(prior, T, k, n_obs, lr):
    """Marginal P(z_i = 1) for the len(lr) unknown pairs of one word."""
    t = len(lr)
    P = prior.get(int(T))
    if P is None or t == 0:
        return np.full(t, 0.5)
    lr = np.clip(np.asarray(lr, dtype=float), 1e-6, 1e6)
    logw = np.zeros(1 << t)
    zmat = np.zeros((1 << t, t))
    for mask in range(1 << t):
        z = np.array([(mask >> j) & 1 for j in range(t)], dtype=float)
        zmat[mask] = z
        s = int(z.sum())
        m = k + s
        if m > T or m < 0 or m >= len(P) or P[m] <= 0:
            logw[mask] = -np.inf
            continue
        logw[mask] = (np.log(P[m]) - np.log(comb(int(T), int(m)))
                      + float((z * np.log(lr)).sum()))
    if not np.isfinite(logw).any():
        return np.full(t, 0.5)
    logw -= np.nanmax(logw[np.isfinite(logw)])
    w = np.where(np.isfinite(logw), np.exp(logw), 0.0)
    if w.sum() <= 0:
        return np.full(t, 0.5)
    w /= w.sum()
    return (w[:, None] * zmat).sum(axis=0)


def predict(train_df, query_df, total_counts, prior, text_p=None):
    """Return posterior P(label=1) for every row of query_df."""
    obs = train_df.groupby('word').label.agg(['sum', 'count'])
    out = np.full(len(query_df), np.nan)
    if text_p is None:
        text_p = np.full(len(query_df), 0.5)
    text_p = np.clip(np.asarray(text_p, dtype=float), 0.02, 0.98)
    lr_all = text_p / (1.0 - text_p)
    idx_by_word = defaultdict(list)
    for pos_i, w in enumerate(query_df.word.values):
        idx_by_word[w].append(pos_i)
    for w, idxs in idx_by_word.items():
        k = int(obs['sum'].get(w, 0))
        n_obs = int(obs['count'].get(w, 0))
        T = total_counts.get(w, n_obs + len(idxs))
        mg = posterior_marginals(prior, T, k, n_obs, lr_all[idxs])
        for j, i in enumerate(idxs):
            out[i] = mg[j]
    return out

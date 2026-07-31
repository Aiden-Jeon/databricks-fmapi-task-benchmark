"""Per-word label-prior features.

The dataset was built so that a given target word receives a *mix* of
same-sense and different-sense pairs.  Hence, knowing the labels of the other
pairs that share the same target word is informative about the current pair
(negative correlation / finite-population effect).
"""
import numpy as np
import pandas as pd


def neighbour_counts(words, labels, query_words, exclude_self_idx=None):
    """n1/n0 of labelled rows sharing the word (optionally excluding own row)."""
    s1 = {}
    s0 = {}
    for w, l in zip(words, labels):
        if l == 1:
            s1[w] = s1.get(w, 0) + 1
        else:
            s0[w] = s0.get(w, 0) + 1
    n1 = np.array([s1.get(w, 0) for w in query_words], float)
    n0 = np.array([s0.get(w, 0) for w in query_words], float)
    if exclude_self_idx is not None:
        for k, (w, l) in enumerate(zip(query_words, exclude_self_idx)):
            if l == 1:
                n1[k] -= 1
            else:
                n0[k] -= 1
    return n1, n0


def fit_conditional(tot_counts, words, labels, max_tot=8):
    """Empirical distribution of #positives per word, for words whose rows are
    all labelled. Returns dict tot -> np.array over m=0..tot of probability."""
    df = pd.DataFrame({'word': words, 'label': labels})
    g = df.groupby('word').label.agg(['sum', 'count'])
    g['tot'] = [tot_counts.get(w, c) for w, c in zip(g.index, g['count'])]
    full = g[g['count'] == g['tot']]
    dist = {}
    for tot, sub in full.groupby('tot'):
        if tot > max_tot:
            continue
        cnt = np.zeros(int(tot) + 1)
        for m in sub['sum']:
            cnt[int(m)] += 1
        dist[int(tot)] = cnt
    return dist


def _binom_prior(tot, rho=0.0):
    """fallback prior on #positives: symmetric, slightly concentrated."""
    from math import comb
    p = np.array([comb(tot, m) for m in range(tot + 1)], float)
    p = p ** (1.0 + rho)
    return p / p.sum()


def conditional_prob(dist, tot, k, n1, alpha=2.0):
    """P(label=1 | tot rows for the word, k labelled ones of which n1 positive)."""
    tot = int(tot); k = int(k); n1 = int(n1)
    if k >= tot or tot < 1:
        tot = max(tot, k + 1)
    n0 = k - n1
    base = dist.get(tot)
    prior = _binom_prior(tot, 0.35)
    if base is not None and base.sum() > 0:
        prior = (base + alpha * prior * base.sum() / max(1, len(base))) 
        prior = prior / prior.sum()
    from math import comb
    num = 0.0
    den = 0.0
    for m in range(tot + 1):
        if m < n1 or (tot - m) < n0:
            continue
        # hypergeometric likelihood of observing n1 positives among k draws
        lik = comb(m, n1) * comb(tot - m, n0)
        wgt = prior[m] * lik
        den += wgt
        rem = tot - k
        if rem > 0:
            num += wgt * (m - n1) / rem
    if den <= 0:
        return 0.5
    return num / den


def prior_features(tot_counts, lab_words, lab_labels, q_words, q_self_labels=None,
                   dist=None):
    n1, n0 = neighbour_counts(lab_words, lab_labels, q_words, q_self_labels)
    tot = np.array([tot_counts.get(w, 1) for w in q_words], float)
    k = n1 + n0
    if dist is None:
        dist = {}
    cp = np.array([conditional_prob(dist, t, kk, nn1)
                   for t, kk, nn1 in zip(tot, k, n1)])
    F = pd.DataFrame({
        'pr_n1': n1, 'pr_n0': n0, 'pr_k': k, 'pr_tot': tot,
        'pr_diff': n1 - n0,
        'pr_rate': (n1 + 1.0) / (k + 2.0),
        'pr_remain': tot - k,
        'pr_cond': cp,
    })
    return F


def empirical_table(tot_counts, lab_words, lab_labels, kcap=4, smooth=25.0):
    """Empirical P(label=1 | k, n1) estimated leave-one-out on the labelled rows."""
    import numpy as np
    words = np.asarray(lab_words)
    labels = np.asarray(lab_labels)
    n1, n0 = neighbour_counts(words, labels, words, labels)
    k = n1 + n0
    kk = np.minimum(k, kcap)
    nn = np.minimum(n1, kcap)
    gm = labels.mean()
    tab = {}
    for key in set(zip(kk.tolist(), nn.tolist())):
        m = (kk == key[0]) & (nn == key[1])
        c = m.sum()
        tab[key] = (labels[m].sum() + smooth * gm) / (c + smooth)
    return {'tab': tab, 'gm': gm, 'kcap': kcap}


def empirical_apply(emp, F):
    import numpy as np
    kcap = emp['kcap']
    kk = np.minimum(F['pr_k'].values, kcap)
    nn = np.minimum(F['pr_n1'].values, kcap)
    return np.array([emp['tab'].get((a, b), emp['gm']) for a, b in zip(kk, nn)])

"""Structural prior from the premise grouping of MultiNLI/KorNLI.

In (Multi)NLI every premise was shown to annotators who wrote one
entailed, one neutral and one contradicting hypothesis, so the labels of
rows sharing a premise are (almost) always *distinct*.  Measured on
train.csv: 97.1% of premise groups with >=2 rows have all-distinct labels.

24% of the test rows share their premise with at least one training row,
so we can turn the observed sibling labels into a prior that is combined
with the model probabilities.  The prior is estimated from train.csv only.
"""
import numpy as np

NC = 3


def estimate_prior(labels, groups, n_sib_max=2):
    """P(y | multiset of sibling labels) estimated by leave-one-out on train.

    Returns dict: sorted tuple of sibling labels -> probability vector.
    """
    from collections import defaultdict
    idx = defaultdict(list)
    for i, g in enumerate(groups):
        idx[g].append(i)
    counts = {}
    for g, rows in idx.items():
        if len(rows) < 2:
            continue
        for i in rows:
            sib = tuple(sorted(labels[j] for j in rows if j != i))[:n_sib_max]
            counts.setdefault(sib, np.zeros(NC))[labels[i]] += 1
    prior = {}
    for k, v in counts.items():
        prior[k] = (v + 0.5) / (v.sum() + 0.5 * NC)
    return prior


def apply_prior(prob, sib_lists, prior, weight=1.0):
    """Multiply model probabilities by the sibling prior (in log space)."""
    out = np.log(np.clip(prob, 1e-9, None)).copy()
    base = np.log(np.full(NC, 1.0 / NC))
    for i, sib in enumerate(sib_lists):
        if not sib:
            continue
        k = tuple(sorted(sib))[:2]
        p = prior.get(k)
        if p is None:                     # unseen combination -> generic rule
            p = np.full(NC, 0.5 / NC)
            for c in set(k):
                p[c] = 0.03
            p = p / p.sum()
        out[i] += weight * (np.log(p) - base)
    e = np.exp(out - out.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def sibling_labels(query_groups, ref_groups, ref_labels, exclude_self=None):
    """For each query row, the labels of reference rows with the same premise."""
    from collections import defaultdict
    m = defaultdict(list)
    for j, g in enumerate(ref_groups):
        m[g].append(j)
    out = []
    for i, g in enumerate(query_groups):
        rows = m.get(g, [])
        if exclude_self is not None:
            rows = [j for j in rows if j != exclude_self[i]]
        out.append([ref_labels[j] for j in rows])
    return out

"""Premise-group structured decoder.

KLUE-NLI was built by showing an annotator one premise and asking for three
hypotheses: one entailment, one neutral and one contradiction.  Rows that share a
premise therefore almost always carry *distinct* labels, and 4027 of the 5000
test premises also occur in train.csv with their labels.

Instead of predicting each row independently we decode every premise group
jointly:

    argmax_{y_unknown}  sum_i log p(y_i | premise, hypothesis_i)
                        + w * log Prior(multiset(y_unknown + y_known))

`Prior` is the empirical distribution of label multisets over training premise
groups of the same size (see `fit_prior`), normalised by the number of
arrangements of the multiset so that it is a distribution over label *sequences*.
It is estimated from train.csv only.  It encodes both "all three labels are
distinct" (90% of size-3 groups) and the asymmetries of the rarer patterns
(e.g. a group whose two known labels are both `entailment` is completed by
`contradiction` 152 times vs `neutral` 11 times in train).

`w > 1` compensates for the fact that the multiclass log-probabilities are
under-confident relative to this prior; w = 3 was selected on held-out splits.
"""
import itertools
from collections import Counter
from math import factorial, log
import numpy as np

NLAB = 3


def _multiset(labels):
    c = [0] * NLAB
    for l in labels:
        c[l] += 1
    return tuple(c)


def _n_arrangements(cnt):
    d = 1
    for c in cnt:
        d *= factorial(c)
    return factorial(sum(cnt)) // d


def fit_prior(premises, labels, alpha=0.5):
    """group size -> {label multiset -> log P(multiset) - log #arrangements}."""
    by = {}
    for p, l in zip(premises, labels):
        by.setdefault(p, []).append(l)
    per_size = {}
    for ls in by.values():
        per_size.setdefault(len(ls), Counter())[_multiset(ls)] += 1
    out = {}
    for k, cnt in per_size.items():
        all_ms = [c for c in itertools.product(range(k + 1), repeat=NLAB) if sum(c) == k]
        tot = sum(cnt.values()) + alpha * len(all_ms)
        out[k] = {ms: log((cnt.get(ms, 0) + alpha) / tot) - log(_n_arrangements(ms))
                  for ms in all_ms}
    return out


def decode(prem_unk, logp, prem_known, y_known, prior, w_prior=3.0, pen=6.0):
    """prem_unk/logp: premises and (n,3) log-probabilities of the rows to predict.
    prem_known/y_known: premises and labels of the labelled (training) rows.
    Returns integer labels."""
    pred = np.argmax(logp, axis=1).copy()
    known_by_prem = {}
    for p, l in zip(prem_known, y_known):
        known_by_prem.setdefault(p, []).append(l)
    unk_by_prem = {}
    for i, p in enumerate(prem_unk):
        unk_by_prem.setdefault(p, []).append(i)

    for p, idx in unk_by_prem.items():
        klab = known_by_prem.get(p, [])
        if not klab and len(idx) == 1:
            continue  # no structural information: keep the argmax
        tbl = prior.get(len(klab) + len(idx))
        best, best_s = None, -np.inf
        for cand in itertools.product(range(NLAB), repeat=len(idx)):
            s = sum(logp[idx[j], c] for j, c in enumerate(cand))
            ms = _multiset(list(cand) + klab)
            if tbl is not None:
                s += w_prior * tbl[ms]
            else:  # unseen group size -> plain duplicate penalty
                s -= pen * sum(max(0, c - 1) for c in ms)
            if s > best_s:
                best_s, best = s, cand
        for j, c in enumerate(best):
            pred[idx[j]] = c
    return pred

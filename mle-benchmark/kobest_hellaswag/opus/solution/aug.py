"""Augmentation using only train.csv.

The context of a row is itself a chain of consecutive events s1..sk, followed by
the gold ending e, followed (in unknown order) by the three distractor endings.
Hence for any j < k we can synthesise a new example:
    context'    = s1..sj
    gold        = s_{j+1}
    distractors = three events strictly later than s_{j+1}
                  (taken from s_{j+2}..sk, e, then the other endings)
This mirrors the real task construction (plausible but later events) and multiplies
the amount of supervision available.
"""
import numpy as np
import pandas as pd
from feats import split_sents, END_COLS


def augment(df, rng, min_ctx=2, max_per_row=3, use_endings=True):
    rows, src = [], []
    for ridx, r in enumerate(df.itertuples(index=False)):
        s = split_sents(r.context)
        k = len(s)
        gold = getattr(r, END_COLS[r.label]) if hasattr(r, 'label') else None
        ends = [getattr(r, c) for c in END_COLS]
        others = [ends[i] for i in range(4) if i != r.label]
        rng.shuffle(others)
        chain = s + [gold] + others          # full event chain (tail order arbitrary)
        cands_pool = []
        # positions j+1 we may use as gold: need j>=min_ctx-1 sentences of context
        for gpos in range(min_ctx, k):       # gold = chain[gpos] (0-based), context = chain[:gpos]
            later = chain[gpos + 1:gpos + 1 + 3]
            if len(later) < 3 or not use_endings and gpos + 3 >= k:
                continue
            cands_pool.append((gpos, later))
        if not cands_pool:
            continue
        rng.shuffle(cands_pool)
        for gpos, later in cands_pool[:max_per_row]:
            cands = [chain[gpos]] + list(later)
            perm = rng.permutation(4)
            lab = int(np.where(perm == 0)[0][0])
            rows.append(dict(id='aug_%d_%d' % (ridx, gpos),
                             context=' '.join(chain[:gpos]),
                             **{END_COLS[i]: cands[perm[i]] for i in range(4)},
                             label=lab))
            src.append(ridx)
    out = pd.DataFrame(rows)
    return out, np.array(src, dtype=int)
